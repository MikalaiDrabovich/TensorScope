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
Quickly find bottleneck ops and their exact input/output tensor shapes for 
models in Tensorflow.

Visualize model end-to-end training on a pie chart diagram. Standard timeline 
from tfprof does not allow to aggregate ops like this, and can not provide 
i/o tensor shapes for some computationally expensive ops, especially in 
case of RNNs. While it could be possible to extract i/o shapes from Tensorflow   
manually and/or after changing model source code, 
for complex models with thousand of ops this may take days.

With Tensorscope, you should get actionable information within minutes 
by following 5 step how-to (see below). For some popular model 
setups, bundled with TensorScope, you'll get results within seconds:
 
cd reproduce_results
./run_me.sh

---

How to get results for your own model setup - 5 steps:

1) git clone https://github.com/MikalaiDrabovich/TensorScope.git

2) Find among your training source files the one with a main loop 
over data batches. This will be unique for your training 
setup and may take quite some time to figure out (you may want to start with 
grep -rE "sess.*\.run\(|\.train\(" . )

As an example, for official.resnet main training loop is located in 
official/resnet/resnet_run_loop.py, and for NMT it will be in nmt/nmt/model.py

3) In the file from previous step:

  a) Import TensorScope module. 
     For example, add after 'import tensorflow as tf' :
          
        import sys
        sys.path.append('full path to TensorScope/tensorscope/ from clone git repo')
      
   - if you use estimator API:
        from tensorscope import TensorScopeRunHook
      
   - if you use session API:
        from tensorscope import TensorScope    

  b) add before the loop from step 2:
   
   - if you use estimator API:
      session_hook = TensorScopeRunHook(num_steps_to_warmup=10,
                                        num_steps_to_measure=100,
                                        model_name='model_name')   
   - if you use session API:
      # ts may need to be defined as global object
      # (see this case in nmt/nmt/model.py)
      ts = TensorScope(num_steps_to_warmup=10,
                       num_steps_to_measure=100,
                       model_name='model_name',
                       session=session)
                         
  c) add in the loop from step 2:
      
   - if you use session API, add 'options' and 'run_metadata' to session.run() and
     also call tensorscope.characterize_model(my_session) after it, for example:
     
      _ = session.run(target,
                      options=ts.options,
                      run_metadata=ts.metadata())  
      ts.characterize_model(graph=session.graph) 
              
   - if you use estimator API, you only need to add/append 
     session_hook to estimator.train(), for example:
        
      my_estimator.train(input_fn=train_input_fn, hooks=[session_hook])


4) Now run training/inference as usual.
TensorScope will generate results and stop training after 
num_steps_to_warmup+num_steps_to_measure steps (10+100 by default)

5) See generated pie_chart.html and data.tsv for detailed results.

Also, take a look at printed out log-scale distribution of top-k ops vs 
cumulative time (under '*** Top-K ops vs total time ***' and duplicated 
in /results/model_name/model_name_log.txt). If you see that, for example, 
top-10 ops (out of total 1000 ops) take 90% of time, it is quite likely that 
focusing optimization/tuning efforts on that 1% of ops could be more 
advantageous. The assumption of course is that full utilization of 
compute device is still far from being reached. 
It is possible to verify this assumption for a small number of ops 
(which have RegisterStatistics('flops') implemented) - see data 
in column 'G(FL)OP/S achieved'.

6) To compare to some other system, copy its 'data.tsv' to 
results/model_name directory, rename 'data.tsv' to 'data_2.tsv'.
See results in 'data_compared.tsv', unmatched ops will be saved
to 'data_unmatched_ops.tsv'


---

If you have questions/comments please open an issue at 
https://github.com/MikalaiDrabovich/TensorScope or send e-mail to nick.drabovich@amd.com 

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
import errno

from tensorflow.python.client import timeline
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs


class TensorScope(object):
    
    def __init__(self, num_steps_to_warmup=10, num_steps_to_measure=100, model_name='model_name', session=tf.get_default_session()):
    
        # number of steps to skip in the beginning
        self.num_steps_to_warm_up = num_steps_to_warmup
        
        # total number of measurements  
        self.num_steps_to_analyze = num_steps_to_measure
        
        # Step number when to generate timeline. You may use it to 
        # double check that TensorScope didn't miss any ops - 
        # total times should be similar.
        # Also, you can manually verfy shapes for some 
        # tensors (if they are not unknown on the timeline).
        # As can be expected, time for timeline generation (as well
        # as RunMetaData analysis) will be excluded from total wall time.
        self.step_to_save_timeline_at = self.num_steps_to_warm_up + int(self.num_steps_to_analyze/2) + 1
        
        # set to -1 to disable, otherwise set to some step number after warm-up, for which to dump all intermediate results
        self.debug_dump_step = self.step_to_save_timeline_at # -1
        
        # To estimate profiler overhead, set to NO_TRACE
        # possible values: NO_TRACE, SOFTWARE_TRACE, HARDWARE_TRACE
        self.options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
        #self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True, allow_soft_placement=False, report_tensor_allocations_upon_oom=False)
       
        self.num_steps_analyzed = 0
        self.total_time_analyzed = 0
        self.current_step = 0
        
        
        self.node_dict = {}
        self.current_node_dict = {}
        self.sess_node_dict_out_raw = {}
        self.sess_node_dict_out = {}
        self.profiling_result = {}
        self.available_flops_estimate = {}
        self.flops_retrieved = False
        
        self.session_results = []
        self.current_raw_nodes = []
        self.current_nodes_with_const = {}
        self.node_list_all_sessions = []
        
        self.use_only_all_end_micros = False
        
        if not session:
            print("tf.session is None, exiting.")
            sys.exit(0)
        self.session = session
        
        # create directory for saving results
        output_dir = os.path.abspath(os.path.join(__file__, '..', '..', 'results', model_name))

        self.temp_path = output_dir if output_dir[-1] == os.sep else output_dir+os.sep
        if (model_name == '') or (''.join(set(model_name.strip())) == os.sep):
            os.model_name='model_name'
        
        if (''.join(set(self.temp_path.strip())) == os.sep):
            self.temp_path = os.path.abspath(os.path.join('tmp', 'tensorscope', 'results', model_name))
            
        try:
            os.makedirs(self.temp_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(e)
                sys.exit(0)
        

    def aggregate_node_data(self, sess_node_dict_out_raw):
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
                    op_finalized_dict[new_key] = {'timings_saved':False, 'op_start_replica':-1, 'op_end_replica': -1, 'all_end_replica':-1, 'replica_key':new_key}
                          
                if ('stream:all' in k1): # 'stream:all' is the first place to search for timing data, if it's not there - we'll get time from memcpy or replica
                    if new_key not in op_device_dict:
                        op_device_dict[new_key] = {'op_start':v1[2],'op_end':v1[3],'all_end':v1[4], 'io_shapes':''.join(v1[0])+'->'+''.join(v1[1]), 'num_calls_stream_all': 1, 'num_calls_replica': 0, 'num_calls_memcpy': 0}
                    else:
                        op_device_dict[new_key]['op_start'] = v1[2]
                        op_device_dict[new_key]['op_end'] = v1[3]
                        op_device_dict[new_key]['all_end'] = v1[4]
                        op_device_dict[new_key]['num_calls_stream_all'] += 1  
                    
                    # set to False to also add timing from replica:
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

            """
            for k2,v2 in op_finalized_dict.items():
                if not v2['timings_saved']:
                    op_device_dict[v2['replica_key']]['op_start'] = v2['op_start_replica']
                    op_device_dict[v2['replica_key']]['op_end'] = v2['op_end_replica']
                    op_device_dict[v2['replica_key']]['all_end'] = v2['all_end_replica']
            """
            # add time from replica
            for k2,v2 in op_finalized_dict.items():
                #print(v2['replica_key'])
                if not v2['timings_saved']:
                    op_device_dict[v2['replica_key']]['op_start'] = v2['op_start_replica']
                    prev_op_end = 0
                    if 'op_end' in op_device_dict[v2['replica_key']]:
                        prev_op_end = op_device_dict[v2['replica_key']]['op_end']
                    
                    prev_all_end = 0
                    if 'all_end' in op_device_dict[v2['replica_key']]:
                        prev_all_end = op_device_dict[v2['replica_key']]['all_end']
                        
                    op_device_dict[v2['replica_key']]['op_end'] = prev_op_end + v2['op_end_replica']
                    op_device_dict[v2['replica_key']]['all_end'] = prev_all_end + v2['all_end_replica']
                            
            for k1,v1 in op_device_dict.items():
                # the final aggregation by type + device + i/o shapes
                new_key =  k + '@' + k1 + '@' + v1['io_shapes']
                result_dict[new_key] = [k, v[0], k1, v1['op_start'], v1['op_end'], v1['all_end'], v1['io_shapes'], v1['num_calls_stream_all'], v1['num_calls_replica'], v1['num_calls_memcpy']]
                if (v1['num_calls_stream_all']==0 and v1['num_calls_replica']==0 and v1['num_calls_memcpy']==0):
                    print('Warning: v1[\'num_calls_stream_all\']==0 and v1[\'num_calls_replica\']==0 and v1[\'num_calls_memcpy\']==0')
                    
        return result_dict
        
        
    def get_common_shapes(self, device_id, in_node, sess_node_dict_out_raw):
        
        if 'LayoutOptimizer' in in_node:
            const_cand = self.add_path_with_const(in_node)
            if const_cand in self.current_nodes_with_const:
                loc_dev_dict = self.current_nodes_with_const[const_cand][1]
                if device_id in loc_dev_dict: 
                    return loc_dev_dict[device_id][1]
                else:
                    for devid, devval in loc_dev_dict.items():
                        if 'replica' in devid:
                            if in_node in loc_dev_dict[devid]:
                                if loc_dev_dict[devid][1][0] != '(no output)':
                                    # TODO: handle case when there is also an exact output slot for the value
                                    return loc_dev_dict[devid][1][0]
        
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
                                # TODO: handle case when there is also an exact output slot for the value
                                return loc_dev_dict[devid][1][0]
         
        # TODO: verify ops below and add more ops with fixed shapes
        if leaf_node == 'ConcatOffset':
            return '(2x1)'
       
        if leaf_node in ['Shape', 'shape', 'ExpandDims', 'reduction_indices']:
            return '(1|2|3x1)'
                            
        if leaf_node in ['begin', 'Begin', 'Enter' , 'enter', 'end', 'End', 'size', 'Size', 'mul', 'alpha', 'beta', 'gamma', 'delta', 'concat_dim', 'ToInt32', 'ToInt64', 'axis', 'Const', 'const', 'Squeeze', 'mod', 'Identity', 'range', 'Fill', 'BroadcastGradientArgs', 'control_dependency', 'Merge', 'Switch']:
            return '(1)'
    
        if leaf_node in ['add', 'Reshape', 'RealDiv',  'zeros', 'zeros_like', 'ones_like', 'ones', 'y', 'stack', 'ExponentialDecay']:
            return '(1|2|3x1)'   

     
 
        """
        # TODO: workaround for missing nodes, if they are still present in ROCm TensorFlow
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

    
    def extract_session_data(self, sess_node_list_in):
        # Returns dictionary. Key - based on path.
        # For example, for /path_part1/part2:part3 key will be /path_part1/part2/part3
        # for /path_part1/part2 key will be /path_part1/part2
        # Values are the following:
        # [shortname, dict['device']]
        # shortname - for example for /path_part1/part2:part3 short name will be 'part3'
        # (TODO: make sure this is ok with any future TF changes)
        # dict['device'] = [input size, output size, times - op start, op end, all end ]
        
        # input: sess_node_list_in = [op name, device, op_start_rel_micros,  op_end_rel_micros,  all_end_rel_micros, output_shapes_list, timeline_label]
        
        total_lines = len(sess_node_list_in)
        current_line = 0
        sess_node_dict_out_raw = {}
                      
        mem_nodes = ['MEMCPYDtoH', 'MEMCPYHtoD', 'MEMCPYDtoD', 'MEMCPYHtoH']
        mem_nodes_hcc  = ['hcMemcpyDeviceToHost', 'hcMemcpyHostToDevice', 'hcMemcpyDeviceToDevice', 'hcMemcpyHostToHost']    
        mem_nodes.extend(mem_nodes_hcc)
                
       
        self.current_nodes_with_const = {}
       
        while current_line < total_lines:
            # aggregate op name in current_key
            current_key = sess_node_list_in[current_line][0]
            current_key = current_key.split('/')
 
            current_key_last = current_key[-1].split(':')
            
            if len(current_key_last) > 1 and current_key_last[-1] in mem_nodes:
                shortname = current_key_last[-1]
                current_key[-1] =  current_key_last[0]
                current_key.append(shortname)         
            else:
                shortname = current_key_last[0]
                current_key[-1] =  current_key_last[0]
                
            current_key = '/'.join(current_key)
            current_device = sess_node_list_in[current_line][1]
             
            # add new key                             
            if current_key not in sess_node_dict_out_raw:
                device_dict = {}
                device_dict[current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                sess_node_dict_out_raw[current_key] = [shortname, device_dict]
            
            # add new device
            if current_device not in sess_node_dict_out_raw[current_key][1]:
                sess_node_dict_out_raw[current_key][1][current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
            
            
            # add new key with generic name for '-LayoutOptimizer'
            if 'LayoutOptimizer' in current_key:
                current_key_const = self.add_path_with_const(current_key)
                if current_key_const not in self.current_nodes_with_const:
                    device_dict = {}
                    device_dict[current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                    self.current_nodes_with_const[current_key_const] = [shortname, device_dict]
                
                # add new device
                if current_device not in self.current_nodes_with_const[current_key_const][1]:
                    self.current_nodes_with_const[current_key_const][1][current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
         
                                   
            device_data = []
                  
           
            # Time data can be extracted from (stream:all + replica) or from (streams:0..N + replica) or just memcpy 
            # 'replica' stream may have input names, short op name, output size.           
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
                            sys.exit('timeline_label string did not have closing bracket: ', sess_node_list_in[current_line][-1], timeline_string)
                    else:
                        inputs_from_timeline_string = inputs_from_timeline_string[:-1]
                        timeline_input_names = inputs_from_timeline_string.split(',')
                        input_names = [p.strip() for p in timeline_input_names]
                
                #sess_node_list_in has [op name, device, op_start_rel_micros,  op_end_rel_micros,  all_end_rel_micros, output_shapes_list, timeline_label]
                device_data = [input_names, sess_node_list_in[current_line][-2], sess_node_list_in[current_line][2], sess_node_list_in[current_line][3], sess_node_list_in[current_line][4]]
            
            if 'stream' in current_device:
                # stream:* doesn't have io tensor shapes info
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
                    # TODO: probably remove this completely
                    current_key = current_key + '/' + mem_cand
                    
                    if not current_key in sess_node_dict_out_raw:
                        device_dict = {}
                        device_dict[current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                        sess_node_dict_out_raw[current_key] = [shortname, device_dict]
                    
                    if current_device not in sess_node_dict_out_raw[current_key][1]:
                        sess_node_dict_out_raw[current_key][1][current_device] = [['(no input)'], ['(no output)'], 0, 0, 0]
                
                    """          
                else:
                    output_shapes_list.append('(no output info)')
                                   
                device_data = [['(Transfer)'], output_shapes_list, sess_node_list_in[current_line][2], sess_node_list_in[current_line][3], sess_node_list_in[current_line][4]]

            current_data = sess_node_dict_out_raw[current_key][1][current_device]
           
            # aggregate op data for every device: preprocess io shapes, sum up time
            # dict[op] = [tensorflow_op_name=shortname, dict[current_device]], 
            # where dict[current_device] = [input tensor names, output shapes,  op_start_rel_micros, op_end_rel_micros, all_end_rel_micros]
            # summig up timings
            time_op_end_sum = int(current_data[3]) + (device_data[3] - device_data[2])
            time_all_end_sum = int(current_data[4]) + device_data[4]
            if time_op_end_sum > 100000000000000:
                print('Warning: overflow in time_op_end_sum =  int(current_data[3]) + (device_data[3] - device_data[2])', current_data[3] + device_data[3] - device_data[2], current_data[3], device_data[3], device_data[2])
                time_op_end_sum = int(current_data[3])
                
            if time_all_end_sum > 100000000000000:
                print('Warning: overflow in time_all_end_sum = int(current_data[4]) + device_data[4]',  current_data[4] + device_data[4],  current_data[4], device_data[4])
                time_all_end_sum = int(current_data[4])
                  
            sess_node_dict_out_raw[current_key][1][current_device] = [device_data[0], device_data[1], device_data[2], time_op_end_sum, time_all_end_sum]
            
            
            # ensure that op_end_rel_micros >= op_start_rel_micros
            if  device_data[3] < device_data[2]:
                print('Error in RunMetadata: op_end_rel_micros < op_start_rel_micros', current_key, current_device, sess_node_dict_out_raw[current_key][1][current_device])
                sys.exit(0)

            current_line +=1
                     
        #replace names of inputs with actual sizes 
        for key, val in sess_node_dict_out_raw.items():
            device_dict = val[1]
            for k,v in device_dict.items():
                # find input names to look for later
                if 'replica' in k:
                    input_names_from_stream = device_dict[k][0]
                    if input_names_from_stream[0] != '(no input)':
                        inp_with_shape = []
                        for in_node in input_names_from_stream:
                            # skip control nodes
                            if in_node[0] !='^':
                                #for each input name candidate find shape if recorded
                                if in_node in sess_node_dict_out_raw:
                                    loc_dev_dict = sess_node_dict_out_raw[in_node][1]
                                    if k in loc_dev_dict:
                                        inp_with_shape.extend(loc_dev_dict[k][1])
                                    else:
                                        leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                        if leaf_shape != '?':
                                            inp_with_shape.extend(leaf_shape)
                                        else:
                                            inp_with_shape.extend(['(?)'])
                                            print('Missing node in RunMetadata: ', in_node, ' for device ', k, ', code ref 1')
                                else:
                                     in_node = in_node.split('/')
                                     leaf_node_parts = in_node[-1].split('_')
                                     leaf_node = [p for p in leaf_node_parts if not p.isdigit()]
                                     leaf_node = '_'.join(leaf_node)
                                     
                                     # if in_node ends in for example /_42
                                     if leaf_node == '': 
                                         in_node = '/'.join(in_node[:-1])
                                         #find shape
                                         if in_node in sess_node_dict_out_raw:
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
                                                    # TODO: check handling of XLA* devices
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
                                                print('Node referenced but missing from RunMetadata: ', in_node, ' for device ', k, ', code ref 3')
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
                                                        # TODO: check handling of XLA* devices
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
                                                            print('Node referenced but missing from RunMetadata: ', in_node, ' for device ', k, ', code ref 4')
                                            else:                                            
                                            
                                                leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                                
                                                if leaf_shape != '?':
                                                    inp_with_shape.extend(leaf_shape)
                                                else:
                                                    inp_with_shape.extend(['(?)'])
                                                    print('Node referenced but missing from RunMetadata: ', in_node, ' for device ', k, ', code ref 5')
                                        else:                                      
                                            in_node = '/'.join(in_node)
                                            leaf_shape = self.get_common_shapes(k, in_node, sess_node_dict_out_raw)
                                            if leaf_shape != '?':
                                                inp_with_shape.extend(leaf_shape)
                                            else:
                                                """
                                                if in_node.endswith('-LayoutOptimizer'):
                                                    # try for generic name
                                                    in_node_const = add_path_with_const(in_node)
                                                    
                                                    
                                                    if in_node_const in sess_node_dict_out_raw:
                                                        loc_dev_dict = sess_node_dict_out_raw[in_node_const][1]
                                                        if k in loc_dev_dict:
                                                            inp_with_shape.extend(loc_dev_dict[k][1])
                                                            print('Shape for missing node from RunMetadata: ', in_node, ' for device ', k, 'inferred from ', in_node_const, ' with result: ', loc_dev_dict[k][1])
                                                        else:
                                                            leaf_shape = self.get_common_shapes(k, in_node_const, sess_node_dict_out_raw)
                                                            if leaf_shape != '?':
                                                                inp_with_shape.extend(leaf_shape)
                                                            else:
                                                                inp_with_shape.extend(['(?)'])
                                                                print('Node missing from RunMetadata: ', in_node, ' for device ', k, 'also tried variant in sess_node_dict_out_raw: ', in_node_const, ', code ref 6')
                                                else:
                                                """
                                                inp_with_shape.extend(['(?)'])
                                                print('Node missing from RunMetadata: ', in_node, ' for device ', k, 'also tried variant in sess_node_dict_out_raw: ', in_node_const, ', code ref 7')
                                                    
                        if len(inp_with_shape) == 0: # if all inputs where control nodes
                            device_dict[k][0] = ['(no input)']
                        else:
                            # replace names of inputs with shapes
                            # TODO: also probably check if ref is correct
                            device_dict[k][0] = inp_with_shape                    
                    # we done, since another (non-replica) streams will not have input names at all:
                    break 

        return sess_node_dict_out_raw
            

    def process_step_stats(self, step_stats, current_raw_nodes):
        # extract [op name, device, time (op start, end, all end), output size ]
        # for memcpy, output size is extracted from timeline_label
        
        all_current_raw_nodes = []
        
        mem_nodes = ['MEMCPYDtoH', 'MEMCPYHtoD', 'MEMCPYDtoD', 'MEMCPYHtoH']
        mem_nodes_hcc  = ['hcMemcpyDeviceToHost', 'hcMemcpyHostToDevice', 'hcMemcpyDeviceToDevice', 'hcMemcpyHostToHost']    
        mem_nodes.extend(mem_nodes_hcc)
        
        # map data type id in RunMetadata to type name (as defined in tensorflow/core/framework/types.proto)
        tensor_data_types = { '0':'invalid',      '1':'float',            '2':'double',           '3':'int32',            '4':'uint8', 
                              '5':'int16',        '6':'int8',             '7':'string',           '8':'complex64',        '9':'int64', 
                              '10':'boolean',     '11':'qint8',           '12':'quint8',         '13':'qint32',           '14':'bfloat16',
                              '15':'qint16',      '16':'quint16',         '17':'uint16',         '18':'complex128',       '19':'half',
                              '20':'tf_resource', '21':'variant',         '22':'uint32',         '23':'uint64',           '101':'float_ref',
                              '102':'double_ref', '103':'int32_ref',      '104':'uint8_ref',    '105':'int16_ref',        '106':'int8_ref',
                              '107':'string_ref', '108':'complex64_ref',  '109':'int64_ref',    '110':'bool_ref',         '111':'qint8_ref',
                              '112':'quint8_ref', '113':'qint32_ref',     '114':'bfloat16_ref', '115':'qint16_ref',       '116':'quint16_ref',
                              '117':'uint16_ref', '118':'complex128_ref', '119':'half_ref',     '120':'tf_resource_ref',  '121':'variant_ref',
                              '122':'uint32_ref', '123':'uint64_ref'}
                              

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
                        # add data type name (if it is not 32b float)
                        if node_stat_out.tensor_description.dtype != 1:
                            # convert int description to data type name
                            result_shape_with_type = tensor_data_types[str(node_stat_out.tensor_description.dtype)] + ' ' + tensor_shape_result
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
                        
                all_current_raw_nodes.append([node_stat.node_name, dev_stats.device, node_stat.op_start_rel_micros, node_stat.op_end_rel_micros, node_stat.all_end_rel_micros, output_shapes_list, node_stat.timeline_label])
        
        self.current_raw_nodes = all_current_raw_nodes      
        self.node_list_all_sessions.extend(all_current_raw_nodes)
   
        
    # TODO: may be used later, for adding workarounds for missing nodes in hipTensorFlow
    """
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
    """
    def add_path_with_const(self, name_from_metadata):
        current_node_path_parts = name_from_metadata.split('/')
        for i, node in enumerate(current_node_path_parts):
            node_loc = node+''
            node_loc = node_loc.split(':')
            for j, nd in enumerate(node_loc):
                nd = nd.split('_')  
                nd = [n if not n.isdigit() else 'tensorscopeN'  for n in nd ]
                node_loc[j] = '_'.join(nd)
            node_loc = ':'.join(node_loc)
            current_node_path_parts[i] = node_loc
            
        return '/'.join(current_node_path_parts)
    
    def print_distribution_summary(self, sorted_times, denom_const):
        # Distribution of time among Top-k ops (op type + io tensor shape + device)
        # Value of k is equally spaced in log10
        num_unique_ops = len(sorted_times)
        if num_unique_ops < 1:
            print("No ops data found. See data dump to debug - set self.debug_dump_step from -1 to self.step_to_save_timeline_at")
            sys.exit(0)
        else:       
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
              print("Top-%d\t(%.1f%% of all ops):\t%.1f ms(%.1f%% of all time)" % (k_values[k_count], 100*k_values[k_count]/float(num_unique_ops),
                    top_k_time[k_count]/(1000*self.num_steps_analyzed),
                    top_k_time[k_count]/denom_const))
                        
                        
    def metadata(self):
          # For session-based setups (without SessionRunHook) we need to increment number of steps here.
          # For estimator API with SessionRunHook, increment is done at the end of before_run()
          self.current_step_start_time = time.time()
          self.current_step += 1
          self.session_metadata = tf.RunMetadata()
          return self.session_metadata
 
 
    def save_timeline(self):
        _timeline = timeline.Timeline(self.session_metadata.step_stats)
        _chrome_trace = _timeline.generate_chrome_trace_format(show_memory=True)

        _timeline_file_name = self.temp_path + "timeline_at_step_%d.json" % self.current_step 
        with open(_timeline_file_name,'w') as _timeline_file:
            _timeline_file.write(_chrome_trace)
            tf.logging.info('Timeline saved in %s', _timeline_file.name)
    
    def compare_systems(data_file_1, data_file_2, result_file, unmatched_ops_file):
 
        tsv_file_obj_1 = open(data_file_1,'r')
        tsv_file_obj_2 = open(data_file_2,'r')
        tsv_file_obj_result = open(result_file,'w')
        tsv_file_obj_unmatched = open(unmatched_ops_file,'w')
        
        tsv_file_writer = csv.writer(tsv_file_obj_result, delimiter='\t')
        tsv_file_writer_unmatched = csv.writer(tsv_file_obj_unmatched, delimiter='\t')
    
        header_tsv = ['Op', 'System_1 time per call', 'System_2 time per call', 'Time ratio Sys1/Sys2 (one call)', 'System_1 total op time', 
        'System_2 total op time', 'Time ratio Sys1/Sys2 (all calls)', 'System_1 occurence', 'System_2 occurence', 'Occurence Ratio', 
        'System1 cumulative time', 'System1 % of total time ', 'System1 cumulative % of total time ',
        'System2 cumulative time', 'System2 % of total time ', 'System2 cumulative % of total time ']
        
        header_tsv_unmatched = ['System number', 'Op rank by total time', 'Time of 1 call, microseconds', 'Number of calls in 1 run', 'Total time in 1 run',
                      '% of total time', 'Cumulative time, microseconds', '% of total cumulative time',
                      '(FL)OPs in RegisterStatistics', 'GFLOP/S achieved',
                      'Op name', 'Device', 'Input/output tensors']
                      
        tsv_file_writer.writerow(header_tsv)
        tsv_file_writer_unmatched.writerow(header_tsv_unmatched)
        
        sys1_op_dict = {}
        sys2_op_dict = {}
        
        sys_op_intersection_dict = {}
        sys_op_difference_dict = {}
        
        sys1 = tsv_file_obj_1.readlines()
        sys2 = tsv_file_obj_2.readlines()
 
        sys1 = sys1[1:]
        sys2 = sys2[1:]
 
        sys1 = [p.strip() for p in sys1] 
        sys2 = [p.strip() for p in sys2]
        
        for p in sys1:
            p_parts = p.split('\t')
            p_parts = [p1.strip() for p1 in p_parts] 
            unique_op_description = '@'.join(p_parts[-3:])
            # op, time 1 call, occ, total time 
            sys1_op_dict[unique_op_description] = [p_parts, p_parts[1], p_parts[2], p_parts[3]]
             
        for p in sys2:
            p_parts = p.split('\t')
            p_parts = [p1.strip() for p1 in p_parts] 
            unique_op_description = '@'.join(p_parts[-3:])
            # op, time 1 call, occ, total time
            sys2_op_dict[unique_op_description] = [p_parts, p_parts[1], p_parts[2], p_parts[3]]
             
        for k,v in sys1_op_dict.items():
            if float(v[1]) <= 0.0:
                print('Warning: v[1] == 0', k, v)
                continue
                
            if float(v[2]) <= 0.0:
                print('Warning: v[1] == 0', k, v)
                continue
                    
            if k in sys2_op_dict:
                sys2_row = sys2_op_dict[k]
                
                if float(sys2_row[1]) <= 0.0:
                    print('Warning: sys2_row[1] == 0', k, sys2_row)
                    continue
                
                if float(sys2_row[2]) <= 0.0:
                    print('Warning: sys2_row[2] == 0', k, sys2_row)
                    continue

                sys_op_intersection_dict[k] = [k, v[1], sys2_row[1], float(v[1])/float(sys2_row[1]), v[3], sys2_row[3], float(v[3])/float(sys2_row[3]), v[2], sys2_row[2], float(v[2])/float(sys2_row[2])]
        
        missing_in_1 = list(set(sys2_op_dict.keys())-set(sys1_op_dict.keys()))
        missing_in_2 = list(set(sys1_op_dict.keys())-set(sys2_op_dict.keys()))
        
        for k in missing_in_1:
            sys1_row = ['System 1']
            sys1_row.extend(sys2_op_dict[k][0])
            tsv_file_writer_unmatched.writerow(sys1_row)
            
        for k in missing_in_2:
            sys2_row = ['System 2']
            sys2_row.extend(sys1_op_dict[k][0])
            tsv_file_writer_unmatched.writerow(sys2_row)
        
        tsv_file_obj_unmatched.close()
        
        # sort ops by total time in sys1
        sorted_times = ( (key, value) for key, value in sorted(sys_op_intersection_dict.items(), key=lambda x: float(list(x[1])[4]), reverse=True))
        sorted_times = list(sorted_times)
        
        # add cumulative stats to find cut offs for the ops to optimize
        fin_sorted_times = []
        total_time_sys1 = 0
        total_time_sys2 = 0
        
        for k,v in sorted_times:
           total_time_sys1 += float(v[4])
           total_time_sys2 += float(v[5])
        
        # absolute total time, cumulative total time, absolute total time percentage, cumultative total time percentage
        stats1 = [0, 0, 0, 0]
        stats2 = [0, 0, 0, 0]
        
        for k,v in sorted_times:
            # v contains ['Op', 'System_1 time per call', 'System_2 time per call', 'Ratio (per occurence)', 'System_1 total op time', 
            #              'System_2 total op time', 'Ratio (per all calls)', 'System_1 occurence', 'System_2 occurence', 'Occurence Ratio']
            t1 = float(v[4])
            t2 = float(v[5])
            
            stats1 = [t1, stats1[1] + t1, t1/total_time_sys1, stats1[3] + t1/total_time_sys1]
            stats2 = [t2, stats2[1] + t2, t2/total_time_sys2, stats2[3] + t2/total_time_sys2]
          
            v.extend(stats1[1:])
            v.extend(stats2[1:])
            
            fin_sorted_times.append((k,v))
        
        for op_name, op_time in sorted_times:
            tsv_file_obj_result.write('\t'.join([str(p) for p in op_time]))
            tsv_file_obj_result.write('\n') 
              
        tsv_file_obj_result.close()
        
        print('\n\n***Comparison completed***')
        print('See data_compared.tsv to select candidates for optimization in system1.')
        print('- See column D \'Time ratio Sys1/Sys2 (one call)\'. This is how many times an op is faster in system2 compared to system1.')
        print('- Ops are sorted by total time in the 1st (slower) system.')
        print('- To cover, for example, 80% of time spent in system1, see values in column M (\'System1 cumulative % of total time\')')
        print('Good candidates for further optimizations will be ops from the first row to the row with value of about 0.8 (80%) in column M')
        print('- Unmatched ops saved to data_unmatched_ops.tsv. Take a look there to see if some time consuming ops are actually unique to system1 and system2.')
        print('- Baselines for system1 are in column B (\'Time of 1 call, microseconds\'). By default, these values are averaged over 100 runs (aka steps, batches) and 10 warm-up steps.')
        


    def generate_results(self): 

        if self.use_only_all_end_micros:
            # if op time is calculated as all_end_rel - 0
            sorted_times = ( (key, value) for key, value in sorted(self.profiling_result.items(), key=lambda x: list(x[1])[5], reverse=True))
        else:
            # if op time is calculated as op_end_rel - op_start_rel 
            sorted_times = ( (key, value) for key, value in sorted(self.profiling_result.items(), key=lambda x: list(x[1])[4], reverse=True))
         
        sorted_times = list(sorted_times)
        
        # if requested, dump sorted results before grouping by type + io shapes + device
        if self.debug_dump_step > 0:
            filename_ops_metadata = self.temp_path + "debug_7_all_sessions_profiling_result.tsv"
            file_ops_metadata = open(filename_ops_metadata,'w')

            for op_name, op_time in sorted_times:
                file_ops_metadata.write('\t'.join([str(p) for p in op_time]))
                file_ops_metadata.write('\t')
                file_ops_metadata.write(op_name)
                file_ops_metadata.write('\n') 
            file_ops_metadata.close()
                    

        """
        # if requested, dump sorted results before grouping by type + io shapes + device
        if self.debug_dump_step > 0:
            filename_ops_metadata = self.temp_path + "data_extracted.tsv"
            file_ops_metadata = open(filename_ops_metadata,'w')
            
            header_data_extracted = ['Op', 'Total Time', 'Occurence', 'Time of 1 call', 'Tensor Shapes', Path in model', 'Time of 1 call, microseconds', 'Number of calls in 1 run', 'Total time in 1 run', '% of total time', 'Cumulative time, microseconds', '% of total cumulative time', 'Op name', 'Device', 'Input/output tensors']
            
            for op_key, op_data in sorted_times:
                file_ops_metadata.write('\t'.join([str(op_data[1]), str(op_data[6]), str(op_data[0]), str(op_data[2]), str(op_data[5]), str(op_data[3]), str(op_data[4]), str(op_data) ]))
                file_ops_metadata.write('\n') 
            file_ops_metadata.close()
        """                        
                    
        total_ops_time = 0.0
        num_unique_ops = len(sorted_times)
        for op_name, op_time in sorted_times:
          if self.use_only_all_end_micros:
              # if op time is calculated as all_end_rel - 0
              total_ops_time = total_ops_time + op_time[5]
          else:
              # if op time is calculated as op_end_rel - op_start_rel 
              total_ops_time = total_ops_time + op_time[4]
                
        mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
        mean_all_ops_time_per_step = float(total_ops_time)/self.num_steps_analyzed

        print("\nSanity check before aggregation:\n total wall time for %d steps: \t\t %.3f sec. (%.6f sec./batch) \n op time extracted from RunMetadata: \t %.3f sec. (%.6f sec./batch) \n number of ops: %d" % 
            (self.num_steps_analyzed, self.total_time_analyzed, mean_time_per_step, total_ops_time/1000000.0, mean_all_ops_time_per_step/1000000.0, num_unique_ops))
               
        # Extract tensor shapes, create input file for chart generation
        # timings grouped by device + op_type + i/o tensors
        results_for_op_device_shape = {}
        op_count = 0
        cumul_time = 0.0
        
        # create files for saving results
        tsv_file_name = self.temp_path + "data.tsv"
        tsv_file_name_for_chart = self.temp_path + "data_for_pie_chart.tsv"
        
        tsv_file_obj = open(tsv_file_name,'w')
        tsv_file_obj_for_chart = open(tsv_file_name_for_chart,'w')
        
        tsv_file_writer = csv.writer(tsv_file_obj, delimiter='\t')
        tsv_file_writer_for_chart = csv.writer(tsv_file_obj_for_chart, delimiter='\t')

        header_tsv = ['Op rank by total time', 'Time of 1 call, microseconds', 'Number of calls in 1 run', 'Total time in 1 run',
                      '% of total time', 'Cumulative time, microseconds', '% of total cumulative time',
                      '(FL)OPs in RegisterStatistics', 'GFLOP/S achieved',
                      'Op name', 'Device', 'Input/output tensors']
                      
        header_tsv_for_chart = ['Node','Time']

        tsv_file_writer.writerow(header_tsv)
        tsv_file_writer_for_chart.writerow(header_tsv_for_chart)

        """
        compare_to_other_system_on_chart = False
        if compare_to_other_system:
            tsv_file_obj_other_system = open(self.temp_path + "data_for_pie_chart_2.tsv",'r')
            other_system_results = tsv_file_obj_other_system.readlines()
            other_system_results = [p.strip() for p in other_system_results]
        """
         
        # finalize data for the chart
        for times_key, times_value in sorted_times:

          # remove slot info from shapes
          spl = times_value[6].split(' slot ')
          if len(spl) > 1:
              no_spl_res = [p for p in spl if not p.isdigit()]
              times_value[6] = ''.join(no_spl_res)
          
          
          shape_str = times_value[6]
            
          num_calls_all_steps =  max(times_value[7],times_value[8],times_value[9])          
          if num_calls_all_steps <=0.0001:
              print('Warning: num_calls_all_steps <=0.0001')
              
          #num_calls_per_step should be float to spot ops that do not run in every session, as could be expected
          num_calls_per_step =  (num_calls_all_steps/float(self.num_steps_analyzed))
          if num_calls_per_step <=0.0001:
              print('Warning: num_calls_per_step <=0.0001')
          
          
          if self.use_only_all_end_micros:
              # all end
              op_time_all_runs = times_value[5]
          else:
              # op end
              op_time_all_runs = times_value[4]
          
          if op_time_all_runs < 1.0:
              print(times_key, 'op_end_all_micros was 0: ', times_value[4], ' using all_end_micros instead : ', times_value[5], 'times_value: ', times_value)
              op_time_all_runs = times_value[5]
              if op_time_all_runs < 1.0:
                  print(times_key, 'both all_end_micros and op_end_all_micros were 0, forcing to be 1 microsecond instead.')
                  op_time_all_runs = 1
          
          
          num_flops_per_step = -1
          times_key_part = times_key.split('@')
          times_key_part = times_key_part[0]
          if times_key_part in self.available_flops_estimate:
              num_flops_per_step = self.available_flops_estimate[times_key_part]
          #else:
          #    print(times_key_part, 'not in RegisterStatistics(flops)')
          
          
          # For RNN models nodes in graph may have different io shapes between sessions.
          # To get actual time spent in each op, don't divide total time by number of session runs.
          # op_time_per_run_microsec = op_time_all_runs
          
          op_time_per_run_microsec = op_time_all_runs/(float(self.num_steps_analyzed))
          
          cumul_time += op_time_per_run_microsec
          op_count = op_count + 1
                   
          opname_short = times_value[1]
          
          opname_for_summation = '/'.join([opname_short, times_value[2], times_value[6]]) # op, device, i/o shapes
          #opname_for_summation = opname_for_summation[0].upper() + opname_for_summation[1:]
           
          if opname_for_summation in results_for_op_device_shape:
              prev_val = results_for_op_device_shape[opname_for_summation]
              flops_est = num_flops_per_step
              if prev_val[2] > 0 and num_flops_per_step > 0:
                  if (prev_val[2] - flops_est) > 0.1:
                      print('Warning: varying estimates for flops found for the same opname_for_summation: ', opname_for_summation, times_key, prev_val[2], flops_est)
                      flops_est = max(flops_est, prev_val[2])  
              results_for_op_device_shape[opname_for_summation] = [prev_val[0] + op_time_all_runs, prev_val[1] + num_calls_all_steps, flops_est]
          else:
              results_for_op_device_shape[opname_for_summation] = [op_time_all_runs, num_calls_all_steps, num_flops_per_step]

          current_row_output_for_chart_tsv = [op_time_per_run_microsec, times_value[2], opname_short]
          chart_nodes = times_value[0].split('/')
          current_row_output_for_chart_tsv.extend(chart_nodes[:-1])
          current_row_output_for_chart_tsv.append(times_value[6])
          
          if num_flops_per_step > 0:
              current_row_output_for_chart_tsv.insert(3, ('%.1f' % (num_flops_per_step/op_time_per_run_microsec/1000000.0)) +' TFLOPS' )
          else:
              current_row_output_for_chart_tsv.insert(3, '|')
                       
         
          # suppress long op paths
          current_row_output_for_chart_tsv = [current_row_output_for_chart_tsv[0], current_row_output_for_chart_tsv[1], current_row_output_for_chart_tsv[2], current_row_output_for_chart_tsv[3],  current_row_output_for_chart_tsv[-1]]
          
          # suppress 'rays' of io shapes on a chart
          current_row_output_for_chart_tsv.append(' ')
          
          # suppress displaying device number
          #del current_row_output_for_chart_tsv[1]
          
          """
          if compare_to_other_system:
              current_row_output_for_chart_tsv.insert(-1, 'red')
              temp1 = current_row_output_for_chart_tsv[2]
              del  current_row_output_for_chart_tsv[2]
              current_row_output_for_chart_tsv.insert(-1, temp1)
          """
              
          tsv_file_writer_for_chart.writerow(current_row_output_for_chart_tsv)
 
        """
        if compare_to_other_system:
            for p in other_system_results[1:]:
                other_row = p.split('\t')
                other_row.insert(-1, 'green')
                temp1 = other_row[2]
                del other_row[2]
                other_row.insert(-1, temp1)
                tsv_file_writer_for_chart.writerow(other_row)
        """
        tsv_file_obj_for_chart.close()
        
        # finalize stats for op+params+device
        fin_sorted_times = ( (key, value) for key, value in sorted(results_for_op_device_shape.items(), key=lambda x: list(x[1])[0], reverse=True))
        fin_sorted_times = list(fin_sorted_times)
        
        # if requested, dump sorted results before grouping by type + io shapes + device
        if self.debug_dump_step > 0:
            filename_ops_metadata = self.temp_path + "debug_8_results_dict.tsv"
            file_ops_metadata = open(filename_ops_metadata,'w')
            for opdata in fin_sorted_times:
                file_ops_metadata.write('\t'.join([str(p) for p in opdata]))
                file_ops_metadata.write('\n') 
            file_ops_metadata.close()    
            
        total_ops_time = 0
        op_count = 0
        cumul_time = 0.0

        num_unique_ops = len(fin_sorted_times)
        for op_name, op_time in fin_sorted_times:
          op_count = op_count + 1
          total_ops_time = total_ops_time + op_time[0]

        mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
        mean_all_ops_time_per_step = float(total_ops_time)/self.num_steps_analyzed  
        denom_const = 0.01*self.num_steps_analyzed*mean_all_ops_time_per_step

        print("\nSanity check after aggregation:\n total wall time for %d steps: \t\t %.3f sec. (%.6f sec./batch) \n op time extracted from RunMetadata: \t %.3f sec. (%.6f sec./batch) \n number of unique ops: %d" % 
            (self.num_steps_analyzed, self.total_time_analyzed, mean_time_per_step, total_ops_time/1000000.0, mean_all_ops_time_per_step/1000000.0, num_unique_ops))
            
        op_rank = 0
        for times_key, times_value in fin_sorted_times:
          num_calls_all_steps =  times_value[1]
          num_calls_per_step =  (times_value[1]/float(self.num_steps_analyzed)) #not int but float (to highlight ops that are not run in every session)
          op_time_all_runs = times_value[0]
          
          # For RNN models nodes in graph may have different io shapes between sessions.
          # To get actual time spent in each op, don't divide total time by number of session runs.
          # op_time_per_run_microsec = op_time_all_runs
          op_time_per_run_microsec = op_time_all_runs/float(self.num_steps_analyzed)
          
          op_time_per_call_microsec = op_time_per_run_microsec/float(num_calls_per_step)
          cumul_time += op_time_per_run_microsec
          op_rank = op_rank + 1
          
          gflops1 = float((times_value[2] if times_value[2] > 0 else 0))
          
          gflops = gflops1/float(op_time_per_call_microsec)
          gflops/=1000.0 #(because op time is not in seconds but in microseconds)
          
          current_row_output = [op_rank, 
                                "%9.2f" % op_time_per_call_microsec,
                                "%9.2f" % num_calls_per_step,
                                "%9.2f" % (op_time_per_run_microsec),
                                "%9.2f" % (100*op_time_all_runs/float(total_ops_time)),
                                "%9.2f" % (cumul_time),
                                "%9.2f" % (100*cumul_time/mean_all_ops_time_per_step),
                                "%9.2f" % (num_calls_per_step*times_value[2] if times_value[2] > 0 else 0),
                                "%9.2f" % gflops]
                                                
          nodepathparts = times_key.split('/')                                 
          current_row_output.extend(nodepathparts)
          tsv_file_writer.writerow(current_row_output)

        tsv_file_obj.close()

        print('\n\n*** Top-K ops vs total time ***\n')
        self.print_distribution_summary(fin_sorted_times, denom_const)
        print("\n*** See data.tsv, pie_chart.html for details ***\n")
        
        if os.path.exists(self.temp_path + "data_2.tsv"):
                TensorScope.compare_systems(self.temp_path + "data.tsv",
                                 self.temp_path + "data_2.tsv",
                                 self.temp_path + "data_compared.tsv",
                                 self.temp_path + "data_unmatched_ops.tsv")
            
        # session should be closed by now
        sys.exit(0)
        
            
    def characterize_model(self, graph=tf.get_default_graph()): 
        if not graph:
            print("session is None, exiting.")
            sys.exit()
        
        if not self.session.graph is graph:
            print("self.session.graph is not graph, exiting.")
            sys.exit()
        
        if not self.flops_retrieved:
            tot_ops_num = 0
            flops_calc = 0
            total_flops = 0
            for op in graph.get_operations():
                flops = -1
                tot_ops_num +=1
                try:
                    flops = ops.get_stats_for_node_def(graph, op.node_def, "flops").value
                    if flops is not None:
                        flops_calc+=1
                        total_flops+=flops
                        self.available_flops_estimate[op.node_def.name] = flops
                except Exception as e: 
                    continue  

            print('Total ops: ', tot_ops_num)
            print('Percentage of nodes with flops stats available: %2.1f%%. Sanity check - sum of number of operations (per single occurence) in these ops: %.3f GFLOP' % (100.0*(flops_calc/tot_ops_num), total_flops/1000000000.0))
            self.flops_retrieved = True 

        
        if self.session_metadata is not None:
            self.current_step_time = time.time() - self.current_step_start_time 
            
            if self.current_step <= self.num_steps_to_warm_up:
                tf.logging.info('Warm-up step %d/%d completed in %4.4f seconds.',
                        self.current_step,self.num_steps_to_warm_up, self.current_step_time)
            else:                       
                self.total_time_analyzed += self.current_step_time
                self.num_steps_analyzed += 1
                parsing_start_time = time.time()
                
                # if requested, dump raw data for debug
                if self.debug_dump_step == self.current_step:
                    with open(os.path.join(self.temp_path,'debug_1_RunMetadata.txt'),'w') as f:
                        f.write(str(self.session_metadata))
                    
                    with open(os.path.join(self.temp_path,'debug_1_RunMetadata_step_stats.txt'),'w') as f:
                        f.write(str(self.session_metadata.step_stats))
                    
                    if hasattr(self.session_metadata, 'partition_graphs'):
                        pg_count = 0
                        for pg in self.session_metadata.partition_graphs:
                            tf.train.write_graph(pg, self.temp_path,'debug_1_RunMetadata_partition_graph_' + str(pg_count) + '.txt')
                            pg_count+=1

                self.process_step_stats(self.session_metadata.step_stats, self.current_raw_nodes)
                
                # if requested, dump extracted nodes from RunMetadata.step_stats
                if self.debug_dump_step == self.current_step:
                    filename_ops_metadata = self.temp_path + "debug_2_nodes_after_process_step_stats.tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')
                    sorted_all_node_names = sorted(self.current_raw_nodes)
                    for opname in sorted_all_node_names:
                        file_ops_metadata.write('\t'.join([str(p) for p in opname]))
                        file_ops_metadata.write('\n') 
                    file_ops_metadata.close()
            
                extracted_session_data = self.extract_session_data(self.current_raw_nodes)
                    
                # if requested, dump consolidated results for verification
                if self.debug_dump_step == self.current_step:
                    filename_ops_metadata = self.temp_path + "debug_3_nodes_after_extract_session_data.tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')
                    for k, v in extracted_session_data.items():
                        file_ops_metadata.write(k)
                        file_ops_metadata.write('\t')
                        file_ops_metadata.write(v[0])
                        file_ops_metadata.write('\t')
                        file_ops_metadata.write(str(v[1]))
                        file_ops_metadata.write('\n')
                    file_ops_metadata.close()
                    
                self.sess_node_dict_out = self.aggregate_node_data(extracted_session_data)
                
                # if requested, dump consolidated results for verification
                if self.debug_dump_step == self.current_step:
                    filename_ops_metadata = self.temp_path + "debug_4_current_step_profiling_result.tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')
                    for k,v in self.sess_node_dict_out.items():
                        file_ops_metadata.write(k)
                        file_ops_metadata.write('\t')
                        file_ops_metadata.write('\t'.join([str(p) for p in v]))
                        file_ops_metadata.write('\n')
                    file_ops_metadata.close()
                    

                # if requested, dump final dict before adding session results
                if self.debug_dump_step == self.current_step:
                    filename_ops_metadata = self.temp_path + "debug_5_previous_step_profiling_result.tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')
         
                    for k,v in self.profiling_result.items():
                        file_ops_metadata.write('\t'.join([str(p) for p in v]))
                        file_ops_metadata.write('\n') 
                    file_ops_metadata.close()
                    
                # add one session results to the dict of all sessions
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
                
                # if requested, dump final dict after adding session results
                if self.debug_dump_step == self.current_step:
                    filename_ops_metadata = self.temp_path + "debug_6_merged_profiling_result.tsv"
                    file_ops_metadata = open(filename_ops_metadata,'w')
         
                    for k,v in self.profiling_result.items():
                        file_ops_metadata.write('\t'.join([str(p) for p in v]))
                        file_ops_metadata.write('\t')
                        file_ops_metadata.write(k)
                        file_ops_metadata.write('\n')
                    file_ops_metadata.close()
                                                  
                parsing_end_time = time.time()
                tf.logging.info('Step %d/%d completed in %4.4f seconds, RunMetadata parsed in %4.4f seconds',
                                self.num_steps_analyzed, self.num_steps_to_analyze, self.current_step_time, parsing_end_time - parsing_start_time)
   
        else:
            print("self.session_metadata is None, check calls to session.run()")
            sys.exit(0)

        # save timeline at a specific step
        if self.step_to_save_timeline_at == self.current_step:
            self.save_timeline()  
                
        # all requested steps are measured, start analysis and generate results
        if self.num_steps_analyzed >= self.num_steps_to_analyze:
            try:
                session.close()
            except:
                pass   

            print('Session is closed')
            self.generate_results()


# SessionRunHook to use with esitmator API
class TensorScopeRunHook(session_run_hook.SessionRunHook):
 
  def __init__(self, num_steps_to_warmup=10, num_steps_to_measure=100, model_name='model_name'):
    self._ts = None
    self._num_steps_to_warmup = num_steps_to_warmup  
    self._num_steps_to_measure = num_steps_to_measure
    self._model_name = model_name
    
  def before_run(self, run_context):
    if not self._ts:
        self._ts = TensorScope(num_steps_to_warmup=self._num_steps_to_warmup,
                               num_steps_to_measure = self._num_steps_to_measure,
                               model_name=self._model_name,
                               session=run_context.session)
                               
    self._ts.current_step_start_time = time.time()
    self._ts.current_step += 1
    return SessionRunArgs(None, options=self._ts.options)

  def after_run(self, run_context, run_values):
    self._ts.session_metadata = run_values.run_metadata
    self._ts.characterize_model(graph=run_context.session.graph)
    
if __name__=="__main__":
    if len(sys.argv) < 2:
        TensorScope.compare_systems("data.tsv", "data_2.tsv", "data_compared.tsv", "data_unmatched_ops.tsv")
    else:
        TensorScope.compare_systems(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    
