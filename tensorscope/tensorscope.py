# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Find what GPU kernels should be tuned for which tensors to significantly reduce 
inference/training time of any Tensorflow model. Results are presented as 
interactive, hierarchical pie chart of GPU/CPU ops with concrete tensor shapes 
and relative time contribution. 

How to:
- If training uses estimator API:
  
  1) Import TensorscopeRunHook to your main training file, for example like this:

        import sys
        # path to cloned git repo https://github.com/MikalaiDrabovich/TensorScope
        path_to_tensorscope = '/home/ndr/TensorScope'

        sys.path.append(path_to_tensorscope)
        from tensorscope import TensorscopeRunHook

  2) Add before main training loop:
  
        session_hook = TensorscopeRunHook(output_dir=path_to_ts_results,
          save_steps=10, show_memory=True, show_dataflow=True)

  3) In main training loop add/append session_hook to estimator.train() call:
  
        mnist_classifier.train(input_fn=train_input_fn, hooks=[session_hook])
     

- If training explicitly uses session API: 

  1) Import Tensorscope to your main training file,
     for example like this:

        import sys
        # path to cloned git repo https://github.com/MikalaiDrabovich/TensorScope
        path_to_tensorscope = '/home/ndr/TensorScope'

        sys.path.append(path_to_tensorscope)
        from tensorscope import Tensorscope
    
  2) Initialize Tensorscope object before main training loop, for example: 
 
        ts = Tensorscope(num_steps=10, output_dir='/tmp/tensorscope', session=session)

  3) Add options and run_metadata parameters to session.run() (in the main training loop)
 
        _ = session.run(target,
                        options=ts.options,
                        run_metadata=ts.metadata())  
                        
        # finally, add right after sesion.run():
        ts.characterize_model(graph=session.graph) 
        
        
Run training/inference as usual, it will stop after specified number of steps
(10 by default) and launch Chrome brpwser to display interactive chart

Take a look at brief summary in terminal. Details are saved to 
data_*.tsv file in output_dir 


Some of the addressed issues:
- models may not provide tensor shapes of inputs or outputs;
- some of collected metadata 'streams' may not have info about tensor inputs
or outputs (for example, 'stream:all')
- Some streams may have only partial timings, we rely on the sum of all minus  
'stream:all' and the ones similar to "/job:localhost/replica:0/task:0/device:GPU:0 Compute"
to get correct total time.
- op names in model and metadata may be slightly different:
e.g. MatMul_17:MatMul  while exact name is necessary to keep to find correct inputs
- chart data format for node path required info about op placement device to be 
inserted before tensor shapes in order to get meaningful hierarchy in a chart.

Mikalai Drabovich, nick.drabovich@amd.com

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import subprocess
import time
import csv
import numpy as np
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import graph_io

from tensorflow.python.training import session_run_hook
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tensorflow.core.protobuf import config_pb2

from enum import Enum

# Find data type names for tensors from int value as defined in RunMetadata proto
# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
class TensorDataType(Enum):
  invalid = 0
  flt = 1 # float
  double = 2
  int32 = 3
  uint8 = 4
  int16 = 5
  int8 = 6
  string = 7
  complex64 = 8  # single-precision complex
  int64 = 9
  boolean = 10 # bool
  qint8 = 11     # quantized int8
  quint8 = 12    # quantized uint8
  qint32 = 13    # quantized int32
  bfloat16 = 14  # float32 truncated to 16 bits.  only for cast ops.
  qint16 = 15    # quantized int16
  quint16 = 16   # quantized uint16
  uint16 = 17
  complex128 = 18  # double-precision complex
  half = 19
  tf_resource = 20
  variant = 21  # arbitrary c++ data types
  uint32 = 22
  uint64 = 23
  float_ref = 101
  double_ref = 102
  int32_ref = 103
  uint8_ref = 104
  int16_ref = 105
  int8_ref = 106
  string_ref = 107
  complex64_ref = 108
  int64_ref = 109
  bool_ref = 110
  qint8_ref = 111
  quint8_ref = 112
  qint32_ref = 113
  bfloat16_ref = 114
  qint16_ref = 115
  quint16_ref = 116
  uint16_ref = 117
  complex128_ref = 118
  half_ref = 119
  tf_resource_ref = 120
  variant_ref = 121
  uint32_ref = 122
  uint64_ref = 123


class Tensorscope(object):
    
    def __init__(self, num_steps=10, output_dir="/tmp/tensorscope/", session=tf.get_default_session()):
    
        if not session:
            print("session is None, exiting.")
            sys.exit()
        else:
            self.session = session
        
        # directory for saving results
        self.temp_path = output_dir if output_dir[-1] == '/' else output_dir+('/')
        self.temp_path = "/tmp/tensorscope/" if self.temp_path.strip() == '/' else self.temp_path
           
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
            
        self.num_steps_to_analyze = num_steps
        
        # 'burn in' - skip 2 runs at start and 2 runs at the end
        self.total_num_sessions = self.num_steps_to_analyze + 4
        
        # timeline may be useful for sanity checks of results,
        # to view it, launch Chrome browser, go to chrome://tracing
        # and load generated timeline_at_step_*.json file
        self.step_to_save_timeline_at = 2
        
        # full trace option should gather both time and memory data 
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                
        self.node_input_names = dict()
        self.node_output_shapes = dict()
        self.final_io_shapes = dict()
        self.op_time_total = dict()
        self.cf_temp = dict()
        
        self.node_dict = {}
        self.current_node_dict = {}
        self.sess_node_dict_out_raw = {}
        
        self.sess_node_dict_out = {}
        self.profiling_result = None
        
        self.current_session = 0
        self.num_steps_analyzed = 0
        self.total_time_analyzed = 0
        self.tracing_graph_parsed = False

        self.session_results = []
        
        # path, device str, duration, all duration, outpu str, timelabel_str
        self.current_raw_nodes = []
        self.current_nodes_generic = []
        self.node_list_all_sessions = []
        
    def make_generic_node_name(self, name_from_metadata):

        current_node_path_parts = name_from_metadata.split('/')
        for i, node in enumerate(current_node_path_parts):
            node_loc = node+''
            node_loc = node_loc.split(':')
            for j, nd in enumerate(node_loc):
                nd = nd.split('_')  
                nd = [n if not n.isdigit() else 'numberN'  for n in nd ]
                node_loc[j] = '_'.join(nd)
            node_loc = ':'.join(node_loc)
            current_node_path_parts[i] = node_loc
            
        return '/'.join(current_node_path_parts)      

    def get_common_shapes(self, device_id, in_node, sess_node_dict_out_raw):
        fin_shape = '?'
        in_node_orig = in_node+''
        
        in_node = in_node.split('/')
        leaf_node_parts = in_node[-1].split('_')
        leaf_node = [p for p in leaf_node_parts if not p.isdigit()]
        leaf_node = '_'.join(leaf_node)
        in_node[-1] = leaf_node
        in_node = '/'.join(in_node)
                                            
        if in_node in sess_node_dict_out_raw:
            loc_dev_dict = sess_node_dict_out_raw[in_node][1]
            if device_id in loc_dev_dict: 
                return loc_dev_dict[device_id][1]
            else:
                for devid, devval in loc_dev_dict.items():
                    if 'replica' in devid:
                        if in_node in loc_dev_dict[devid]:
                            if loc_dev_dict[devid][1][0] != '(no output)':
                                # what if there is also an exact output slot for the value?
                                return loc_dev_dict[devid][1][0]
         
    
        if leaf_node == 'ConcatOffset':
            return '(2x1)'
       
        if leaf_node in ['Shape', 'shape', 'ExpandDims', 'reduction_indices']:
            return '(1|2|3x1)'
                            
        if leaf_node in ['begin', 'Begin', 'Enter' , 'enter', 'end', 'End', 'size', 'Size', 'mul', 'alpha', 'beta', 'gamma', 'delta', 'concat_dim', 'ToInt32', 'ToInt64', 'axis', 'Const', 'const', 'Squeeze', 'mod', 'Identity', 'range', 'Fill', 'BroadcastGradientArgs', 'control_dependency', 'Merge', 'Switch']:
            return '(1)'
    
        if leaf_node in ['add', 'Reshape', 'RealDiv',  'zeros', 'zeros_like', 'ones_like', 'ones', 'y', 'stack', 'ExponentialDecay']:
            return '(1|2|3x1)'   
        
        """
        if in_node_orig.endswith('TransposeNHWCToNCHW-LayoutOptimizer'):
            temp_workaround = in_node_orig.replace('NHWCToNCHW', 'NCHWToNHWC')
            temp_workaround = temp_workaround.replace('-0-', '-0-0-')
            
            if temp_workaround in sess_node_dict_out_raw:
                loc_dev_dict = sess_node_dict_out_raw[temp_workaround][1]
                if device_id in loc_dev_dict: 
                    return loc_dev_dict[device_id][1]
                else:
                    for devid, devval in loc_dev_dict.items():
                        if 'replica' in devid:
                            if in_node in loc_dev_dict[devid]:
                                if loc_dev_dict[devid][1][0] != '(no output)':
                                    # what if there is also an exact output slot for the value?
                                    return loc_dev_dict[devid][1][0]
        """                        
                                
        return fin_shape

    
    def consolidate_node_data(self, sess_node_dict_out_raw):
        result_dict = {}
        for k,v in sess_node_dict_out_raw.items():
            all_streams_dict = v[1]
           
            device_type = 'unknown'
            device_number = -1
 
            op_device_dict = {}
            op_finalized_dict = {}
                       
            timings_saved = False
            
            op_start_replica = -1
            op_end_replica = -1
            all_end_replica = -1
            replica_key = ''
 
                   
            for k1,v1 in all_streams_dict.items():
                 
                device_path = k1.split('/')
                for dev_path in device_path:
                    dev_path_parts = dev_path.split(':')
                    if dev_path_parts[0] == 'device':
                       device_type = dev_path_parts[1]
                       device_number = dev_path_parts[2]
                       break
                
                new_key = device_type + '-' + device_number
                
                if not new_key in op_finalized_dict:
                     
                    op_finalized_dict[new_key] = {'timings_saved':False, 'op_start_replica':-1, 'op_end_replica': -1, 'all_end_replica':-1, 'replica_key':''}
                          
                if ('stream:all' in k1): # this stream will have the highest priority in timing data, if it's not available - get time from memcpy or replica;
                    if new_key not in op_device_dict:
                        op_device_dict[new_key] = {'op_start':v1[2],'op_end':v1[3],'all_end':v1[4], 'io_shapes':''.join(v1[0])+'->'+''.join(v1[1]), 'num_calls_stream_all': 1, 'num_calls_replica': 0, 'num_calls_memcpy': 0}
                    else:
                     
                        op_device_dict[new_key]['op_start'] = v1[2]
                        op_device_dict[new_key]['op_end'] = v1[3]
                        op_device_dict[new_key]['all_end'] = v1[4]
                        op_device_dict[new_key]['num_calls_stream_all'] += 1
                        
                    op_finalized_dict[new_key]['timings_saved'] = True
                        
                        
                if ('replica' in k1):
                
                    if new_key not in op_device_dict:
                        op_device_dict[new_key] = {'io_shapes':''.join(v1[0])+'->'+''.join(v1[1]),  'num_calls_stream_all': 0, 'num_calls_replica': 1, 'num_calls_memcpy': 0}
                    else:
                        op_device_dict[new_key]['io_shapes'] = ''.join(v1[0])+'->'+''.join(v1[1])
                        op_device_dict[new_key]['num_calls_replica'] += 1
                    
                    if not op_finalized_dict[new_key]['timings_saved']:
              
                        op_finalized_dict[new_key]['op_start_replica'] = v1[2]
                        op_finalized_dict[new_key]['op_end_replica'] = v1[3]
                        op_finalized_dict[new_key]['all_end_replica'] = v1[4]
                    
                        op_finalized_dict[new_key]['replica_key'] = new_key+'' 
                        
                
                if  ('memcpy' in k1):
                
                    if new_key not in op_device_dict:
                        op_device_dict[new_key] = {'io_shapes':''.join(v1[0])+'->'+''.join(v1[1]),  'num_calls_stream_all': 0, 'num_calls_replica': 0, 'num_calls_memcpy': 1}
                    else:
                        op_device_dict[new_key]['io_shapes'] = ''.join(v1[0])+'->'+''.join(v1[1])
                        op_device_dict[new_key]['num_calls_memcpy'] +=1
                        
                    op_device_dict[new_key]['op_start'] = v1[2]
                    op_device_dict[new_key]['op_end'] = v1[3]
                    op_device_dict[new_key]['all_end'] = v1[4]
                    
                    op_finalized_dict[new_key]['timings_saved'] = True
                                  
            
            for k2,v2 in op_finalized_dict.items():
                if not v2['timings_saved']:
                    op_device_dict[v2['replica_key']]['op_start'] = v2['op_start_replica']
                    op_device_dict[v2['replica_key']]['op_end'] = v2['op_end_replica']
                    op_device_dict[v2['replica_key']]['all_end'] = v2['all_end_replica']
                    
            for k1,v1 in op_device_dict.items():
                new_key =  k + '@' + k1 + '@' + v1['io_shapes']
                result_dict[new_key] = [k, v[0], k1, v1['op_start'], v1['op_end'], v1['all_end'], v1['io_shapes'], v1['num_calls_stream_all'], v1['num_calls_replica'], v1['num_calls_memcpy']]
       
        return result_dict
    
    def aggregate_session_by_node_path(self, sess_node_list_in):
        total_lines = len(sess_node_list_in)
        current_line = 0
        sess_node_dict_out_raw = {}
                      
        mem_nodes = ['MEMCPYDtoH', 'MEMCPYHtoD', 'MEMCPYDtoD', 'MEMCPYHtoH']
        mem_nodes_hcc  = ['hcMemcpyDeviceToHost', 'hcMemcpyHostToDevice', 'hcMemcpyDeviceToDevice', 'hcMemcpyHostToHost']    
        mem_nodes.extend(mem_nodes_hcc)
                
        while current_line < total_lines:
 
            current_key = sess_node_list_in[current_line][0]
            current_key = current_key.split('/')
            current_key_last = current_key[-1] + ''
            current_key_last = current_key_last.split(':')
            
            if len(current_key_last) > 1 and current_key_last[-1] in mem_nodes:
                shortname = current_key_last[-1]
                current_key[-1] =  current_key_last[0]
                current_key.append(shortname)
            else:
                shortname = current_key_last[0]
                current_key[-1] = shortname+ ''
                
            current_key = '/'.join(current_key)
            current_device = sess_node_list_in[current_line][1]
                                          
            if not current_key in sess_node_dict_out_raw:
                device_dict = {}
                device_dict[current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                sess_node_dict_out_raw[current_key] = [shortname, device_dict]
            
            if current_device not in  sess_node_dict_out_raw[current_key][1]:
                sess_node_dict_out_raw[current_key][1][current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                                        
            device_data = []
                  
            # we can get input names, short op name, output size, but not time, time will be from stream:all or sum of other streams minus stream:all and replica  
            if 'replica' in current_device: 
                timeline_string = sess_node_list_in[current_line][-1]
                input_names = ''
                if timeline_string != '':
                    shortname = timeline_string.split('=')
                    shortname = shortname[-1]
                    shortname = shortname.split('(')
                    shortname = shortname[0].strip()
                    sess_node_dict_out_raw[current_key][0] = shortname + ''
                   
                    timeline_string = timeline_string.split('(')
                    inputs_from_timeline_string = timeline_string[-1]
                    if len(inputs_from_timeline_string)==1: # if just closing bracket, then no inputs
                        input_names = ['(no input)']
                        if inputs_from_timeline_string[0] != ')':
                            sys.exit('timeline_label string didnt have closing bracket: ', sess_node_list_in[current_line][-1], timeline_string)
                    else:
                        inputs_from_timeline_string = inputs_from_timeline_string[:-1]
                        timeline_input_names = inputs_from_timeline_string.split(',')
                        input_names = [p.strip() for p in timeline_input_names]
                
                device_data = [input_names, sess_node_list_in[current_line][-2], sess_node_list_in[current_line][2], sess_node_list_in[current_line][3], sess_node_list_in[current_line][4]]
            
            if 'stream' in current_device: # we can get times from stream:all amd stream:0..n 
                device_data = [['no inputs saved in stream'], ['no outputs saved in stream'], sess_node_list_in[current_line][2], sess_node_list_in[current_line][3], sess_node_list_in[current_line][4]]
 
            if 'memcpy' in current_device:
                timeline_string = sess_node_list_in[current_line][-1]
                
                lb = timeline_string
                mem_cand = lb.split(' ')[0]
                
                output_shapes_list = []
                if mem_cand in mem_nodes:
                    memsz = timeline_string.split(' ')
                    output_shapes_list.append('(' + memsz[1]+' bytes)')
                    
                    """
                    current_key = current_key + '/' + mem_cand
                    
                    if not current_key in sess_node_dict_out_raw:
                        device_dict = {}
                        device_dict[current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                        sess_node_dict_out_raw[current_key] = [shortname, device_dict]
                    
                    if current_device not in  sess_node_dict_out_raw[current_key][1]:
                        sess_node_dict_out_raw[current_key][1][current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                
                    """          
                else:
                    output_shapes_list.append('(no output info)')
                                   
                device_data = [['(Transfer)'], output_shapes_list, sess_node_list_in[current_line][2], sess_node_list_in[current_line][3], sess_node_list_in[current_line][4]]

            current_data = sess_node_dict_out_raw[current_key][1][current_device]            
            sess_node_dict_out_raw[current_key][1][current_device] = [device_data[0], device_data[1], current_data[2], int(current_data[3]) + device_data[3] - device_data[2], int(current_data[4]) + device_data[4]]            
            current_line +=1
                     
        
        count = 0
        #now replace names of inputs with actual sizes 
        for key, val in sess_node_dict_out_raw.items():
            device_dict = val[1]
            for k,v in device_dict.items():
                if 'replica' in k: # find input names to look for later
                    input_names_from_stream = device_dict[k][0]
                    if input_names_from_stream[0] != '(no input)':
                        inp_with_shape = []
                        for in_node in input_names_from_stream:
                            if in_node[0] !='^': # skip control nodes
                            
                                if in_node in sess_node_dict_out_raw:  #for each input name candidate find shape if recorded
                                    loc_dev_dict = sess_node_dict_out_raw[in_node][1]
                                    if k in loc_dev_dict:
                                        inp_with_shape.extend(loc_dev_dict[k][1])
                                    else:
                                    
                                        leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                                
                                        if leaf_shape != '?':
                                            inp_with_shape.extend(leaf_shape)
                                        else:
                                            inp_with_shape.extend(['(?)'])
                                            print('Attention: this node is referenced as input in timeline_label but is absent from RunMetadata: ', in_node, ' for device ', k, ', code ref 1')
                                        
                                else:
                                     in_node = in_node.split('/')
                                     leaf_node_parts = in_node[-1].split('_')
                                     leaf_node = [p for p in leaf_node_parts if not p.isdigit()]
                                     leaf_node = '_'.join(leaf_node)
                                     
                                     if leaf_node == '': # if in_node ends in for example /_42
                                         in_node = '/'.join(in_node[:-1])
                                         if in_node in sess_node_dict_out_raw:  #for each input name candidate find shape if recorded
                                            loc_dev_dict = sess_node_dict_out_raw[in_node][1]
                                            if k in loc_dev_dict:
                                                inp_with_shape.extend(loc_dev_dict[k][1])
                                            else:
                                                new_res = []
                                                
                                                k_alt = k.replace('CPU', 'GPU')
                                                if k_alt in loc_dev_dict:
                                                    new_res = loc_dev_dict[k_alt][1]
                                                else:
                                                    k_alt = k.replace('GPU', 'CPU')
                                                    if k_alt in loc_dev_dict:
                                                        new_res = loc_dev_dict[k_alt][1]
                                                    """
                                                    else:
                                                        k_alt = k.replace('XLA_GPU', 'CPU')
                                                        if k_alt in loc_dev_dict:
                                                            new_res = loc_dev_dict[k_alt][1]
                                                            # k_alt = k.replace('GPU', 'CPU') xla
                                                    """
         
                                                if len(new_res) > 0:
                                                    inp_with_shape.extend(new_res)
                                                else:
                                                
                                                    leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                            
                                                    if leaf_shape != '?':
                                                        inp_with_shape.extend(leaf_shape)
                                                    else:                                        
                                                        inp_with_shape.extend(['(?)'])
                                                        print('2 ? - for node ', in_node, ' for device ', k)
                                         else:
                                         
                                  
                                            leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                            
                                            if leaf_shape != '?':
                                                inp_with_shape.extend(leaf_shape)
                                            else:
                                                inp_with_shape.extend(['(?)'])
                                                print('Attention: this node is referenced as input in timeline_label but is absent from RunMetadata: ', in_node, ' for device ', k, ', code ref 3')

                                     else:
                                        # try exact slot number 
                                        leaf_node_parts = in_node[-1].split(':')
                                        slot_number = 'not defined'
                                        if leaf_node_parts[-1].isdigit():
                                            slot_number = int(leaf_node_parts[-1])
                                                                              
                                        if slot_number != 'not defined':
                                            in_node[-1] = leaf_node_parts[0] 
                                            in_node = '/'.join(in_node)
                                            
                                            
                                            if in_node in sess_node_dict_out_raw:  #for each input name candidate find shape if recorded
                                                loc_dev_dict = sess_node_dict_out_raw[in_node][1]
                                                if k in loc_dev_dict:
                                                
                                                    outputs_maybe_with_slots = loc_dev_dict[k][1]
                                                    
                                                    for outp in outputs_maybe_with_slots:
                                                        spl_outp = outp.split(' slot ')
                                                        if len(spl_outp) > 1:
                                                            sln = int(spl_outp[-2])
                                                            if sln == slot_number:
                                                                outputs_maybe_with_slots = [spl_outp[0] + ')']
                                                                break
                                                            
                                                    inp_with_shape.extend(outputs_maybe_with_slots)
                                                else:
                                                    new_res = []
                                                    
                                                    k_alt = k.replace('CPU', 'GPU')
                                                    if k_alt in loc_dev_dict:
                                                        new_res = loc_dev_dict[k_alt][1][tensor_number_in_output]
                                                    else:
                                                        k_alt = k.replace('GPU', 'CPU')
                                                        if k_alt in loc_dev_dict:
                                                            new_res = loc_dev_dict[k_alt][1][tensor_number_in_output]
                                                        """
                                                        else:
                                                            k_alt = k.replace('XLA_GPU', 'CPU')
                                                            if k_alt in loc_dev_dict:
                                                                new_res = loc_dev_dict[k_alt][1]
                                                                # k_alt = k.replace('GPU', 'CPU') xla
                                                        """
                                                        
                                                        
                                                    if len(new_res) > 0:
                                                        inp_with_shape.extend(new_res)
                                                    else:
                                                        leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                                
                                                        if leaf_shape != '?':
                                                            inp_with_shape.extend(leaf_shape)
                                                        else:
                                                            inp_with_shape.extend(['(?)'])
                                                            print('Attention: this node is referenced as input in timeline_label but is absent from RunMetadata: ', in_node, ' for device ', k, ', code ref 4')
                                            else:                                            
                                            
                                                leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                                
                                                if leaf_shape != '?':
                                                    inp_with_shape.extend(leaf_shape)
                                                else:
                                                    inp_with_shape.extend(['(?)'])
                                                    print('Attention: this node is referenced as input in timeline_label but is absent from RunMetadata: ', in_node, ' for device ', k, ', code ref 5')

                                        else:
                                                                                  
                                            in_node = '/'.join(in_node)
                                            leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                            
                                            if leaf_shape != '?':
                                                inp_with_shape.extend(leaf_shape)
                                            else:
                                                inp_with_shape.extend(['(?)'])
                                                print('Attention: this node is referenced as input in timeline_label but is absent from RunMetadata: ', in_node, ' for device ', k, ', code ref 6')

                                  
                        if len(inp_with_shape) == 0: # if all inputs where control nodes
                            device_dict[k][0] = ['(no input)']
                        else:
                            # replace names of inputs with shapes found
                            device_dict[k][0] = inp_with_shape # ref correct?
                 
                    break # since other non replica will not have input names at all
                    
      
        """
        filename_ops_metadata = self.temp_path + "before consolidation"+str(time.time())+".tsv"
        file_ops_metadata = open(filename_ops_metadata,'w')


        for k,v in sess_node_dict_out_raw.items():
            
            op_per_device = []
        
            dev_dict = v[1]
            for dev, op_data in dev_dict.items():
                op_per_device.append(dev)
                op_per_device.append(op_data)
            line_out = [k, v[0], op_per_device]
            file_ops_metadata.write('\t'.join([str(p) for p in line_out]))
            file_ops_metadata.write('\n')
            
        file_ops_metadata.close()
        """

        result_dict = self.consolidate_node_data(sess_node_dict_out_raw)
       
        return result_dict
            

    def fill_raw_data(self, step_stats, current_raw_nodes):
    
        # save RunMetadata from session
        #tf.train.write_graph(step_stats, self.temp_path, 'metadata_'+str(time.time()) +'.txt')
        
        all_current_raw_nodes = []
        
        mem_nodes = ['MEMCPYDtoH', 'MEMCPYHtoD', 'MEMCPYDtoD', 'MEMCPYHtoH']
        mem_nodes_hcc  = ['hcMemcpyDeviceToHost', 'hcMemcpyHostToDevice', 'hcMemcpyDeviceToDevice', 'hcMemcpyHostToHost']    
        mem_nodes.extend(mem_nodes_hcc)
        
        for dev_stats in step_stats.dev_stats:
            for node_stat in dev_stats.node_stats:
            
                output_shapes_list = []
                if node_stat.output:
                    for (i, node_stat_out) in enumerate(node_stat.output):
                        
                        node_stat_dims = node_stat_out.tensor_description.shape.dim

                        valid_tensor_shapes = [str(d.size) for d in node_stat_dims if (d.size and str(d.size)!='')]
                        if len(valid_tensor_shapes) == 1:
                            valid_tensor_shapes.append('1')
                        
                        tensor_shape_result = 'x'.join(valid_tensor_shapes)

                        if tensor_shape_result=='':
                            tensor_shape_result = '1x1'

                        # prepend data type if it is not 32b float
                        if node_stat_out.tensor_description.dtype != 1:
                            # convert int description to data type name
                            tensor_data_type = TensorDataType(node_stat_out.tensor_description.dtype)
                            result_shape_with_type = tensor_data_type.name + ' ' + tensor_shape_result
                        else:
                            result_shape_with_type = tensor_shape_result
                        
                        if result_shape_with_type=='':
                            result_shape_with_type = '? unknown type'
                        
                        if node_stat_out.slot:
                            output_shapes_list.append('('+result_shape_with_type+' slot ' + str(node_stat_out.slot) + ' slot ' +')')
                        else:
                            output_shapes_list.append('('+result_shape_with_type+')')
                else:
                
                    if node_stat.timeline_label:
                    
                        lb = node_stat.timeline_label
                        mem_cand = lb.split(' ')[0]
                        
                        
                        if mem_cand in mem_nodes:
                            memsz = node_stat.timeline_label.split(' ')
                            output_shapes_list.append('(' + memsz[1]+' bytes)')
                        else:
                            output_shapes_list.append('(no output info)')
                    else:
                        output_shapes_list.append('(no output)')
                        
                output_shapes = output_shapes_list#''.join(output_shapes_list) 
                all_current_raw_nodes.append([node_stat.node_name, dev_stats.device, node_stat.op_start_rel_micros, node_stat.op_end_rel_micros, node_stat.all_end_rel_micros, output_shapes, node_stat.timeline_label])
        
                
        self.current_raw_nodes = all_current_raw_nodes      
        self.node_list_all_sessions.extend(all_current_raw_nodes)
        
        # save unaggregated data from each session
        """
        filename_ops_metadata = self.temp_path + "node_list_one_session_"+str(time.time())+".tsv"
        file_ops_metadata = open(filename_ops_metadata,'w')
        sorted_all_node_names = sorted(all_current_raw_nodes)
        for opname in sorted_all_node_names:
            file_ops_metadata.write('\t'.join([str(p) for p in opname]))
            file_ops_metadata.write('\n')
            
        file_ops_metadata.close()
        """
       
 
    def print_distribution_summary(self, sorted_times, denom_const):
        # Distribution of time among top k ops
        # Value of k is equally spaced in log10
        
        num_unique_ops = len(sorted_times)
        k_values = np.round(np.power(10, np.linspace(0, np.log10(num_unique_ops), num=10, endpoint=True)).astype(np.double))
        top_k_time = np.zeros(len(k_values))
        
        bin_count = 0
        op_count = 0
        total_ops_time = 0
           
        for op_name, op_time in sorted_times:
          op_count = op_count + 1
          total_ops_time = total_ops_time + op_time[0]
          
          if op_count >= k_values[bin_count]:
            top_k_time[bin_count] = total_ops_time
            bin_count = bin_count + 1
            
        for k_count in range(len(top_k_time)):
          print("Top-%d ops (%.1f%%)\t%.1f ms (%.1f%%)" % (k_values[k_count], 100*k_values[k_count]/float(num_unique_ops),
                top_k_time[k_count]/(1000*self.num_steps_analyzed),
                top_k_time[k_count]/denom_const))
                        
                        
    def metadata(self):

          self.session_start_time = time.time()
          self.current_session += 1
          self.session_metadata = tf.RunMetadata()
          return self.session_metadata
 
 
    def save_timeline(self):
    
        _timeline = timeline.Timeline(self.session_metadata.step_stats)
        _chrome_trace = _timeline.generate_chrome_trace_format(show_memory=True)

        _timeline_file_name = self.temp_path + "timeline_at_step_%d.json" % (self.current_session) #str(time.time())[:10]      
        with open(_timeline_file_name,'w') as _timeline_file:
            _timeline_file.write(_chrome_trace) 
            print('Timeline saved in', _timeline_file.name)
   
   
    def show_results(self, tsv_file_name_for_chart):
  
        # immediatly break outside loop of training/inference
        sys.exit(0)
         
    
    def generate_results(self):
       
        # op_end timings
        #sorted_times = ( (key, value) for key, value in sorted(self.profiling_result.items(), key=lambda x: list(x[1])[4], reverse=True))
        
        # all_end timings
        sorted_times = ( (key, value) for key, value in sorted(self.profiling_result.items(), key=lambda x: list(x[1])[5], reverse=True))
        
        sorted_times = list(sorted_times)
 
        total_ops_time = 0.0
        num_unique_ops = len(sorted_times)
        for op_name, op_time in sorted_times:
          # op_end timings
          #total_ops_time = total_ops_time + op_time[4]
          # all_end timings
          total_ops_time = total_ops_time + op_time[5]
                
        mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
        mean_all_ops_time_per_step = float(total_ops_time)/self.num_steps_analyzed
        denom_const = 0.01*self.num_steps_analyzed*mean_all_ops_time_per_step

        #print("\nSanity check: total time for %d steps: %.3f sec., 1 step avg. time:  %.1f microsec., 1 step avg. time captured in metadata: %.1f microsec., number of unique ops: %d" % 
        #    (self.num_steps_analyzed, self.total_time_analyzed, 1000000.0*mean_time_per_step, mean_all_ops_time_per_step, num_unique_ops))
        
        # *** extract tensor shapes, create input file for chart generation ***
        
        # this is where we store timings of ops grouped by device+op_type+tensors
        results_for_op_device_shape = {}
        
        op_count = 0
        cumul_time = 0.0
        
        # prepare files for saving results
        tsv_file_name = self.temp_path + "data.tsv"
        tsv_file_name_for_chart = self.temp_path + "data_for_pie_chart.tsv"
        
        tsv_file_obj = open(tsv_file_name,'w')
        tsv_file_obj_for_chart = open(tsv_file_name_for_chart,'w')

        tsv_file_writer = csv.writer(tsv_file_obj, delimiter='\t')
        tsv_file_writer_for_chart = csv.writer(tsv_file_obj_for_chart, delimiter='\t')

        header_tsv = ['Op rank by total time', 'Time of 1 call, microseconds', 'Number of calls in 1 run', 'Total time in 1 run', '% of total time', 'Cumulative time, microseconds', '% of total cumulative time', 'Op name', 'Device', 'Input/output tensors']
        header_tsv_for_chart = ['Node','Time']

        tsv_file_writer.writerow(header_tsv)
        tsv_file_writer_for_chart.writerow(header_tsv_for_chart)
        
        
        # finalize data for the chart
        for times_key, times_value in sorted_times:
 
          # remove slot info from shapes
          spl = times_value[6].split(' slot ')
          if len(spl) > 1:
              no_spl_res = [p for p in spl if not p.isdigit()]
              times_value[6] = ''.join(no_spl_res)
          
          opname_short = times_value[1]
          shape_str = times_value[6]
            
          num_calls_all_runs =  max(times_value[7],times_value[8],times_value[9])
          num_calls_per_run =  (num_calls_all_runs/float(self.num_steps_analyzed))#int(times_value[1]/self.num_steps_analyzed)
          
          # rounding fix 99/100
          if num_calls_per_run == 0:
              num_calls_per_run = 1

          op_time_all_runs = times_value[5]
          

          # keep in mind that same nodes will likely have different io shapes between session for models like nmt (seq2seq, lstm)
          # alternatively, don't divide total time by number of session runs, to get actual time spent in each op
          op_time_per_run_microsec = op_time_all_runs/(float(self.num_steps_analyzed))
          #op_time_per_run_microsec = op_time_all_runs 
           
          op_time_per_call_microsec = op_time_per_run_microsec/float(num_calls_per_run)
          
          cumul_time += op_time_per_run_microsec
          op_count = op_count + 1
                   
          opname_for_summation = '/'.join([opname_short, times_value[2], times_value[6]]) # op, device, i/o shapes
          #opname_for_summation = opname_for_summation[0].upper() + opname_for_summation[1:]
           
          if opname_for_summation in results_for_op_device_shape:
              prev_val = results_for_op_device_shape[opname_for_summation] 
              results_for_op_device_shape[opname_for_summation] = [prev_val[0] + op_time_all_runs, prev_val[1] + num_calls_all_runs]
          else:
              results_for_op_device_shape[opname_for_summation] = [op_time_all_runs, num_calls_all_runs]

          current_row_output_for_chart_tsv = [op_time_per_run_microsec, times_value[2], opname_short]#, times_value[0]]
          chart_nodes = times_value[0].split('/')
          current_row_output_for_chart_tsv.extend(chart_nodes[:-1])
          current_row_output_for_chart_tsv.append(times_value[6])  
          tsv_file_writer_for_chart.writerow(current_row_output_for_chart_tsv)
          
        tsv_file_obj_for_chart.close()
        
        # finalize stats for op+params+device 
        fin_sorted_times = ( (key, value) for key, value in sorted(results_for_op_device_shape.items(), key=lambda x: list(x[1])[0], reverse=True))
        sorted_times = list(fin_sorted_times)
        
        total_ops_time = 0
        op_count = 0
        cumul_time = 0.0

        num_unique_ops = len(sorted_times)
        for op_name, op_time in sorted_times:
          op_count = op_count + 1
          total_ops_time = total_ops_time + op_time[0]

        mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
        mean_all_ops_time_per_step = float(total_ops_time)/self.num_steps_analyzed  
        denom_const = 0.01*self.num_steps_analyzed*mean_all_ops_time_per_step

        print("\nSanity check: total time for %d steps: %.3f sec., 1 step avg. time:  %.1f microsec., 1 step avg. time captured in metadata: %.1f microsec., number of unique ops: %d" % 
            (self.num_steps_analyzed, self.total_time_analyzed, 1000000.0*mean_time_per_step, mean_all_ops_time_per_step, num_unique_ops))

        op_rank = 0
        for times_key, times_value in sorted_times:
          num_calls_all_runs =  times_value[1]
          num_calls_per_run =  (times_value[1]/float(self.num_steps_analyzed)) #int
          
          # rounding fix 99/100
          if num_calls_per_run == 0:
              num_calls_per_run = 1

          op_time_all_runs = times_value[0]
          op_time_per_run_microsec = op_time_all_runs/float(self.num_steps_analyzed)
          op_time_per_call_microsec = op_time_per_run_microsec/float(num_calls_per_run)
          
          cumul_time += op_time_per_run_microsec
          op_rank = op_rank + 1
                   
          current_row_output = [op_rank, 
                                "%9.1f" % op_time_per_call_microsec,
                                "%9.1f" % num_calls_per_run, #%d
                                "%9.1f" % (op_time_per_run_microsec),
                                "%9.1f" % (100*op_time_all_runs/float(total_ops_time)),
                                "%9.1f" % (cumul_time),
                                "%9.1f" % (100*cumul_time/mean_all_ops_time_per_step)]
                                
          nodepathparts = times_key.split('/')                                 
          current_row_output.extend(nodepathparts)
          tsv_file_writer.writerow(current_row_output)

        tsv_file_obj.close()

        print('\n*** Brief summary ***\n')
        self.print_distribution_summary(sorted_times, denom_const)
        print('\n*** I/O tensor shapes are saved to %s ***\n' % tsv_file_name)
       
        self.show_results(tsv_file_name_for_chart)
        
            
    def characterize_model(self, graph=tf.get_default_graph()):
    
        if not graph:
            print("session is None, exiting.")
            sys.exit()
        else:
            assert self.session.graph is graph
            self.model_graph = graph
         
        if self.session_metadata is not None:
            
            # assuming that the default graph is the one of interest
            #self.model_graph = agraph#tf.get_default_graph()
            self.current_session_time = time.time() - self.session_start_time 
                    
            # skip two first and two last session runs
            if self.current_session > 1 and self.current_session < (self.total_num_sessions - 2) : 
                self.total_time_analyzed += self.current_session_time
                self.num_steps_analyzed += 1
                
                if True:
                    parsing_start_time = time.time()
                    
                    self.fill_raw_data(self.session_metadata.step_stats, self.current_raw_nodes)#input_names, output_shapes):
                    self.sess_node_dict_out = self.aggregate_session_by_node_path(self.current_raw_nodes)
                     
                    # save raw data
                    """
                    filename_ops_metadata = self.temp_path + "final_session_"+str(time.time())+".tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')

                    
                    for k,v in self.sess_node_dict_out.items():
                        file_ops_metadata.write('\t'.join([str(p) for p in v]))
                        file_ops_metadata.write('\n')
                        
                    file_ops_metadata.close()
                    """
                    
                    # aggregate by type + parameters + device
                    if not self.profiling_result:
                        self.profiling_result = self.sess_node_dict_out.copy()
                    else:
                        for k,v in self.sess_node_dict_out.items():
                            if k in self.profiling_result:
                                self.profiling_result[k][3] = self.profiling_result[k][3] + v[3]
                                self.profiling_result[k][4] = self.profiling_result[k][4] + v[4]
                                self.profiling_result[k][5] = self.profiling_result[k][5] + v[5]
                                
                                if v[6] != self.profiling_result[k][6]:
                                    print('Warning: shape differ between sessions for node:', k, v[6], self.profiling_result[k][6])
                                     
                                self.profiling_result[k][7] = self.profiling_result[k][7] + v[7]
                                self.profiling_result[k][8] = self.profiling_result[k][8]+ v[8]
                                self.profiling_result[k][9] =  self.profiling_result[k][9] + v[9]
                                
                            else:
                                self.profiling_result[k] = v
                                

                    parsing_end_time = time.time()
                    self.tracing_graph_parsed = True
                    print("Session graph and metadata parsed in %4.4f seconds" % (parsing_end_time - parsing_start_time))                    

        else:
            print("Metadata was not collected, check calls to session.run()")
            sys.exit(0)

        print("Step %d/%d(%d) completed in %4.4f seconds." % 
        (self.num_steps_analyzed, self.num_steps_to_analyze, self.total_num_sessions, self.current_session_time))
        

        # save timeline at a specific step (number 2 by default)
        if self.current_session == self.step_to_save_timeline_at:
            self.save_timeline()        
                
        # after we captured required number of steps, 
        # start analysis and show results
        if self.num_steps_analyzed == self.num_steps_to_analyze:
            #if not self.session._closed:
            #    self.session.close()
            try:
                session.close()
            except:
                pass
                
            print('Session is closed')
            
            filename_ops_metadata = self.temp_path + "data_all_sessions_unaggregated" + ".tsv"
            file_ops_metadata = open(filename_ops_metadata,'w')
 
            for k,v in self.profiling_result.items():
                file_ops_metadata.write('\t'.join([str(p) for p in v]))
                file_ops_metadata.write('\n')
                
            file_ops_metadata.close()
                    
            self.generate_results()


class TensorscopeRunHook(session_run_hook.SessionRunHook):
 
  def __init__(self,
               save_steps=None,
               save_secs=None,
               output_dir='/tmp',
               show_dataflow=True,
               show_memory=False):

    self._ts = None 
    self._output_file = os.path.join(output_dir, "timeline-{}.json")
    self._file_writer = SummaryWriterCache.get(output_dir)
    self._output_dir = output_dir
    self._show_dataflow = show_dataflow
    self._show_memory = show_memory
    self._timer = tf.train.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)

  def begin(self):
    self._next_step = None
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfilerHook.")

  def before_run(self, run_context):
    self._request_summary = (
        self._next_step is None or
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    opts = (config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            if self._request_summary else None)
    
    if not self._ts:
        self._ts = Tensorscope(num_steps=10, output_dir=self._output_dir, session=run_context.session)
    
    self._ts.session_start_time = time.time()
    self._ts.current_session += 1
          
    return SessionRunArgs(requests, options=opts)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results["global_step"]
    global_step = stale_global_step + 1
    
    self._ts.session_metadata = run_values.run_metadata
    self._ts.characterize_model(graph=run_context.session.graph) 
    self._next_step = global_step + 1
