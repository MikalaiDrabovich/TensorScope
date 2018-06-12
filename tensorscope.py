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

Some of the addressed issues:
- models may not provide tensor shapes of inputs or outputs;
- some of collected metadata 'streams' may not have info about tensor inputs
or outputs (for example, 'stream:all')
- Some streams may have only partial timings, we rely on the sum of all except 
'stream:all' to get op time.
- op names in model and metadata may be slightly different:
e.g. MatMul_17:MatMul  while exact name is necessary to keep to find correct inputs
- chart data format for node path required info about op placement device to be 
inserted before tensor shapes in order to get meaningful hierarchy in a chart.

The goal was to minimize any necessary code change to existing setup 
for model training/inference.

How to:
- import tensorscope to your file with the main training/inference loop
- initialize Tensorscope object before main training loop
  ts = Tensorscope(num_steps=10, output_dir='/tmp/tensorscope')

- add two parameters to session.run() (only to the main training loop)
    _ = session.run(target,
                    options=ts.options,
                    run_metadata=ts.metadata())  
                    
- add call right after sesion.run():
    ts.characterize_model() 
    
- run training/inference as usual, it will stop after specified number of steps
(10 by default) and launch Chrome brpwser to display interactive chart

- take a look at brief summary in terminal. Details are saved to 
data_*.tsv file in output_dir 

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
        
        self.current_session = 0
        self.num_steps_analyzed = 0
        self.total_time_analyzed = 0
        self.tracing_graph_parsed = False

    
    def are_op_part_names_equal(self, str1, str2):
        paths1 = str1.split('_')        # ['strided', 'slice', '1']
        op_stems1 = [p for p in paths1 if not p.isdigit()] # ['strided', 'slice']
        variant1 =  (''.join(op_stems1  )).lower()  # stridedslice
        
        paths2 = str2.split('_')        # ['strided', 'slice', '2']
        op_stems2 = [p for p in paths2 if not p.isdigit()] # ['strided', 'slice']
        variant2 =  (''.join(op_stems2)).lower()  # stridedslice
      
        return variant1 == variant2
      
    def cleanup_op_path_from_metadata(self, name_from_metadata):
        # transform input similar to 'strided_slice_1:strided_slice_1:StridedSlice' -> 'strided_slice_1'
        #name_from_metadata = name_from_metadata.lower()
        
        current_node_path_parts = name_from_metadata.split('/')
        possible_duplicates = current_node_path_parts[-1].split(':')
        # fill actual __cf__ node names
        cf_cand_num = current_node_path_parts[-1].split('_')
        cf_cand = [cf for cf in cf_cand_num if not cf.isdigit()]
        cf_cand = ''.join(cf_cand)
        
        duplicates_removed = []
        if len(possible_duplicates) > 1:
            unique_parts = [possible_duplicates[0]]
            # assumption: StridedSlice goes after strided_slice_1
            for i, dup_cand in enumerate(possible_duplicates):
                #if dup_cand == 'MEMCPYDtoD':
                #    import pdb
                #    pdb.set_trace()
                    
                if not self.are_op_part_names_equal(dup_cand, unique_parts[-1]):
                    unique_parts.append(dup_cand)

            possible_duplicates = ':'.join(unique_parts)
            current_node_path_parts[-1] = possible_duplicates

        cleaned_up_path = '/'.join(current_node_path_parts)
        cleaned_up_path_no_cf = '/'.join(current_node_path_parts[:-1])
        
        if ''.join(cf_cand)=='cf':
            self.cf_temp[cleaned_up_path_no_cf] = cleaned_up_path
        
        return cleaned_up_path
      
                    
    def find_input_names_from_model(self, model_graph, input_names_from_model):
        ''' 
        Find names of input nodes from model graph,
            The names then will be used to find input parameters 
            from metadata graph, which sometimes doesn't have info about
            input node parameters
        Args:
            (in)  model_graph (tensorflow.Graph): model of interest
            (out) input_names_from_model (dict): dictionary of node name and its inputs
        '''
        all_ops = model_graph.get_operations()
        print("Total number of ops in a model graph: ", len(all_ops))
        
        # this file may be quite big
        tf.train.write_graph(model_graph, self.temp_path,'model_graph.json')
        
        all_node_names = set()
        all_node_names_cleaned = set()
        
        for node in all_ops:
            #if not "Variable" in node.op:
            #    continue
                
            all_node_names.add(node.name)
            node_name_cleaned = node.name
            
            #node_name_cleaned = node_name_cleaned.lower()
            
            all_node_names_cleaned.add(node_name_cleaned)
            input_names = []
                
            for i, inp in enumerate(node.inputs):
                input_names.append(inp.name)
                
            input_names_from_model[node_name_cleaned] =  input_names
            
        # save op names for any additional inspection
        filename_ops_model = self.temp_path + "node_names_model_original_"+str(time.time())[:10]+".txt"
        file_ops_model = open(filename_ops_model,'w')
        sorted_all_node_names = sorted(all_node_names)
        
        for opname in sorted_all_node_names:
            file_ops_model.write(opname+'\n')
        file_ops_model.close()
        
        filename_ops_model = self.temp_path + "node_names_model_cleaned_"+str(time.time())[:10]+".txt"
        file_ops_model = open(filename_ops_model,'w')
        sorted_all_node_names_cleaned = sorted(all_node_names_cleaned)
        
        for opname in sorted(sorted_all_node_names_cleaned):
            file_ops_model.write(opname+'\n')
        file_ops_model.close()      
            
                        
    def find_output_shape_from_metadata(self, step_stats, output_shape_device):
        ''' 
        Find node output parameters (tensor shapes)
        Args:
            (in)  step_stats (tensorflow.RunMetadata): run_metadata from session.run()
            (out) output_shape (dict): dictionary of node name and output tensor shape
            
        Assumption is that all recorded streams have the same tensor shapes for identical ops,
        steam:all doesn't have output tensor shapes info at all, so we have to use data from
        other streams.
        
        '''
        
        tf.train.write_graph(step_stats, self.temp_path, 'metadata.json')
      
        num_nodes_total = 0
        
        all_node_names = set()
        all_node_names_orig = set() 
        
        for dev_stats in step_stats.dev_stats:
            # assume device naming is consistent
            device_path = dev_stats.device.split('/')
            device_type = ''
            device_number = 0
            
            for dev_path in device_path:
                dev_path_parts = dev_path.split(':')
                if dev_path_parts[0] == 'device':
                   device_type = dev_path_parts[1]
                   device_number = dev_path_parts[2]
            
            print('New device info extracted from metadata:',  dev_stats.device, device_type, device_number)
 
            #'stream:all' may not have info about output nodes, the other streams may have it,
            # however, another streams may have only partial timings, go figure..
            for node_stat in dev_stats.node_stats:
                num_nodes_total+=1
                output_shapes = []
                _device_type = ''
                _device_number = ''
                
                cleanedup_op_path = self.cleanup_op_path_from_metadata(node_stat.node_name)
                
                all_node_names.add(cleanedup_op_path)
                all_node_names_orig.add(node_stat.node_name)
                 
                if cleanedup_op_path in output_shape_device:
                    [output_shapes, _device_type, _device_number] = output_shape_device[cleanedup_op_path]
                
                 # iterate over all outputs, if there are any
                if node_stat.output:
                    for (i, node_stat_out) in enumerate(node_stat.output):
                        node_stat_dims = node_stat_out.tensor_description.shape.dim

                        valid_tensor_shapes = [str(d.size) for d in node_stat_dims if (d.size and str(d.size)!='')]
                        if len(valid_tensor_shapes) == 1:
                            valid_tensor_shapes.append('1')
                        
                        tensor_shape_result = 'x'.join(valid_tensor_shapes)
                        if tensor_shape_result == 'x':
                            import pdb
                            pdb.set_trace()
                            
                        if tensor_shape_result=='':
                            tensor_shape_result = '1x1'
                        # indicate data type only if it is not he usual DT_FLOAT = 1
                        if node_stat_out.tensor_description.dtype != 1: # todo find type number : str correspondence  
                            # (TODO) add switch case from notes
                            if node_stat_out.tensor_description.dtype == 3:
                                result_shape_with_type = '(int32) '
                            else:
                                if node_stat_out.tensor_description.dtype == 9:
                                    result_shape_with_type = '(int64) '
                                else:
                                    result_shape_with_type = '(type '+str(node_stat_out.tensor_description.dtype) + ') '
                                
                            result_shape_with_type += tensor_shape_result
                        else:
                            result_shape_with_type = tensor_shape_result
                        
                        if result_shape_with_type=='': # if a scalar value
                            result_shape_with_type = '1x1'
                        output_shapes.append('('+result_shape_with_type+')')
                        
                new_device = device_type
                if _device_type != '':
                    if  new_device != _device_type:
                        new_device = new_device + '+' + _device_type
                
                new_device_number = device_number
                if _device_number != '':
                    if  new_device_number != _device_number:
                        new_device_number = new_device_number + '+' + _device_number
                     
                output_shape_device[cleanedup_op_path] = [output_shapes, new_device , new_device_number]
                            
        print("Total number of nodes in a metadata graph: ", num_nodes_total)

        filename_ops_metadata = self.temp_path + "node_names_metadata_cleaned_"+str(time.time())[:10]+".txt"
        file_ops_metadata = open(filename_ops_metadata,'w')
        sorted_all_node_names = sorted(all_node_names)
        for opname in sorted_all_node_names:
            file_ops_metadata.write(opname+'\n')
        file_ops_metadata.close()
        
        filename_ops_metadata = self.temp_path + "node_names_metadata_original_"+str(time.time())[:10]+".txt"      
        file_ops_metadata = open(filename_ops_metadata,'w')
        sorted_all_node_names_orig = sorted(all_node_names_orig)
        for opname in sorted_all_node_names_orig:
            file_ops_metadata.write(opname+'\n')
        file_ops_metadata.close()      
 
 
    def find_input_names_output_shapes(self, step_stats, input_names, io_shapes):
        ''' 
        Find node output parameters (tensor shapes)
        Args:
            (in)  step_stats (tensorflow.RunMetadata): run_metadata from session.run()
            (out) output_shape (dict): dictionary of node name and output tensor shape
            
        Assumption is that all recorded streams have the same tensor shapes for identical ops,
        steam:all doesn't have output tensor shapes info at all, so we have to use data from
        other streams.
        
        '''
        
        #io_shapes = {}
        
         
        tf.train.write_graph(step_stats, self.temp_path, 'metadata.txt')
        tf.train.write_graph(self.model_graph, self.temp_path,'model_graph.txt')
    
        num_nodes_total = 0
        
        all_node_names = set()
        all_node_names_orig = set() 
        
        for dev_stats in step_stats.dev_stats:
            # assume device naming is consistent
            device_path = dev_stats.device.split('/')
            device_type = ''
            device_number = 0
            
            for dev_path in device_path:
                dev_path_parts = dev_path.split(':')
                if dev_path_parts[0] == 'device':
                   device_type = dev_path_parts[1]
                   device_number = dev_path_parts[2]
            
            print('New device info extracted from metadata:',  dev_stats.device, device_type, device_number)
 
            #'stream:all' may not have info about output nodes, the other streams may have it,
            # however, another streams may have only partial timings, go figure..
            for node_stat in dev_stats.node_stats:
                num_nodes_total+=1
                
                _device_type = ''
                _device_number = ''
                
                cleanedup_op_path = node_stat.node_name#self.cleanup_op_path_from_metadata(node_stat.node_name)
                
                all_node_names.add(cleanedup_op_path)
                all_node_names_orig.add(node_stat.node_name)
                 
                #if cleanedup_op_path in io_shapes:
                #    [output_shapes_list, _device_type, _device_number] = io_shapes[cleanedup_op_path]

                input_names_list = []
                lable_cand = ''
                # iterate over all outputs, if there are any
                if node_stat.timeline_label:
                    lb = node_stat.timeline_label
                    lb1 = lb.split('(')
                    lb2 = lb1[-1]
                    lable_cand = lb2[:-1]
                    
                    if lable_cand in ['Pinned to Device', 'Device to Pinned', 'Device to Device', 'Pageable to Device']:
                        input_names_list.append('(no input)')
                    else: 
                        if lable_cand=='':
                            input_names_list.append('(1x1 Const)')
                        else:
                            input_names_list.append(lable_cand)#node_stat.timeline_label.split('(')[-1][:-1]
                else:
                    input_names_list.append('(no input)')
                          
                # iterate over all outputs, if there are any
                output_shapes_list = []
                if node_stat.output:
                    for (i, node_stat_out) in enumerate(node_stat.output):
                        node_stat_dims = node_stat_out.tensor_description.shape.dim

                        valid_tensor_shapes = [str(d.size) for d in node_stat_dims if (d.size and str(d.size)!='')]
                        if len(valid_tensor_shapes) == 1:
                            valid_tensor_shapes.append('1')
                        
                        tensor_shape_result = 'x'.join(valid_tensor_shapes)
                        if tensor_shape_result == 'x':
                            import pdb
                            pdb.set_trace()
                            
                        if tensor_shape_result=='':
                            tensor_shape_result = '1x1'
                        # indicate data type only if it is not he usual DT_FLOAT = 1
                        if node_stat_out.tensor_description.dtype != 1: # todo find type number : str correspondence  
                            # (TODO) add switch case from notes
                            if node_stat_out.tensor_description.dtype == 3:
                                result_shape_with_type = '(int32) '
                            else:
                                if node_stat_out.tensor_description.dtype == 9:
                                    result_shape_with_type = '(int64) '
                                else:
                                    result_shape_with_type = '(type '+str(node_stat_out.tensor_description.dtype) + ') '
                                
                            result_shape_with_type += tensor_shape_result
                        else:
                            result_shape_with_type = tensor_shape_result
                        
                        if result_shape_with_type=='': # if a scalar value
                            result_shape_with_type = '1x1'
                            print('if result_shape_with_type==')
                            import pdb
                            pdb.set_trace()
                            
                        output_shapes_list.append('('+result_shape_with_type+')')
                else:
                    if lable_cand in ['Pinned to Device', 'Device to Pinned', 'Device to Device', 'Pageable to Device']:
                        if node_stat.timeline_label:
                            memsz = node_stat.timeline_label.split(' ')
                            memsz = memsz[1]
                            output_shapes_list.append('transfer '+memsz+' bytes')
                        else:
                            output_shapes_list.append('no info about transfer size')
                    else:
                        output_shapes_list.append('(no output)')
                 
                        
                new_device = device_type
                if _device_type != '':
                    if  new_device != _device_type:
                        new_device = new_device + '+' + _device_type
                
                new_device_number = device_number
                if _device_number != '':
                    if  new_device_number != _device_number:
                        new_device_number = new_device_number + '+' + _device_number
                     
                io_shapes[cleanedup_op_path] = [input_names_list,[output_shapes_list, new_device , new_device_number]]
                 
        for node_name, io_info in io_shapes.items():
            # find inputs shapes from input names
            input_cand_names_list = io_info[0]
            input_cand_shapes_list = []
            
            input_cand_names_list1 = input_cand_names_list[0].split(',')
            input_cand_names_list1 = [p.strip(' ^$') for p in input_cand_names_list1]
            #input_cand_names_list1 = [p.strip() for p in input_cand_names_list1]
           
            for icand_name in input_cand_names_list1:
            
                const_like_ops = icand_name.split('/')[-1]
                const_like_ops = const_like_ops.split('_')
                const_like_ops = [p for p in const_like_ops if not p.isdigit()]
               
                if len(const_like_ops)==0:
                    const_like_ops = icand_name.split('/')[-2]
                    
                if const_like_ops[0] == '':
                    const_like_ops = icand_name.split('/')[-2]
                else:
                
                    const_like_ops = '_'.join(const_like_ops)
                    c1 = const_like_ops.split(':')
                    if len(c1) > 1:
                        const_like_ops = ':'.join(c1[:-1])
                    
                if const_like_ops in ['ConcatOffset', 'Shape', 'axis', 'Const', 'Squeeze', 'ExpandDims', 'mod', 'range', 'Fill', 'reduction_indices', 'BroadcastGradientArgs']:
                    if const_like_ops == 'ConcatOffset':
                        input_cand_shapes_list.extend('(2x1)')
                   
                    if const_like_ops in ['Shape', 'ExpandDims', 'reduction_indices']:
                        strided_slice_name = icand_name.split('/')[-2]
                        ssnc = strided_slice_name.split('_')
                        if ssnc[0]=='strided' and ssnc[1]=='slice': 
                            input_cand_shapes_list.extend('(1x3)')
                        else:
                            input_cand_shapes_list.extend(['(1|2|3x1)'])
                                        
                    if const_like_ops in ['axis', 'Const', 'Squeeze', 'mod', 'range', 'Fill', 'BroadcastGradientArgs']:
                        input_cand_shapes_list.extend('(1x1)')
                  
                else:                
                    if icand_name not in ['(no input)', '(1x1 Const)']:
                        if icand_name in io_shapes:
                            input_cand_shapes_list.extend(io_shapes[icand_name][1][0])
                        else:
                            opn = icand_name.split(':')
                            if len(opn)>1:
                                opn = opn[:-1]
                            opn = ':'.join(opn)
                            
                            if opn in io_shapes:
                                input_cand_shapes_list.extend(io_shapes[opn][1][0])
                            else:
              
                                opns = opn.split('/')
                                if len(opns)>1:
                                    if opn.split('/')[-2] == 'ToInt32':
                                        input_cand_shapes_list.extend('(1x1)')
                                      
                                    else:
                                        opn = icand_name + '_1'
                                        if opn in io_shapes:
                                            input_cand_shapes_list.extend(io_shapes[opn][1][0])
                                        else:           
                                            opn = icand_name.split('_')
                                            if len(opn)>1:
                                                opn = opn[:-1]
                                                opn = '_'.join(opn)
                                                if opn[-1] == '/':
                                                    opn = opn[:-1]
                                        
                                                if opn in io_shapes:
                                                    input_cand_shapes_list.extend(io_shapes[opn][1][0])
                                                else:
                                                    opn = opn + '_1'
                    
                                                    if opn in io_shapes:
                                                        input_cand_shapes_list.extend(io_shapes[opn][1][0])
                                                    else:
                                                        if const_like_ops in ['add', 'Reshape', 'RealDiv', 'shape', 'ones_like','y', 'stack', 'ExponentialDecay']:
                                                            input_cand_shapes_list.extend(['(1|2|3x1)'])
                                                        else:
                                                            input_cand_shapes_list.extend(['(?)'])
                                                            #input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name + ' also tried: ' + opn])
                                                            
                                                            """
                                                            tensor_from_model = self.model_graph.get_tensor_by_name(icand_name)
                                                            if tensor_from_model is not None:
                                                                if tensor_from_model.shape.dims is None:
                                                                    tensor_shape_result = '1x1'
                                                                else:
                                                                    valid_tensor_shapes = [d.__str__() for d in tensor_from_model.shape.dims if d.__str__() != '']
                                                                    tensor_shape_result = 'x'.join(valid_tensor_shapes)
                                                                input_cand_shapes_list.extend([tensor_shape_result])
                                                            else:      
                                                                input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name + ' also tried: ' + opn])
                                                            """                                           
                                            
                                            else:
                                                if const_like_ops in ['add', 'Reshape', 'RealDiv', 'shape', 'ones_like','y','stack']:
                                                    input_cand_shapes_list.extend(['(1|2|3x1)'])
                                                else:
                                                    input_cand_shapes_list.extend(['(?)'])
                                                    #input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name])
                                                    """
                                                    tensor_from_model = self.model_graph.get_tensor_by_name(icand_name)
                                                    if tensor_from_model is not None:
                                                        if tensor_from_model.shape.dims is None:
                                                            tensor_shape_result = '1x1'
                                                        else:
                                                            valid_tensor_shapes = [d.__str__() for d in tensor_from_model.shape.dims if d.__str__() != '']
                                                            tensor_shape_result = 'x'.join(valid_tensor_shapes)
                                                        input_cand_shapes_list.extend([tensor_shape_result])
                                                    else:      
                                                        input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name])
                                                    """
                                                    
                                else:
                                    input_cand_shapes_list.extend(['(?)'])
                                    #input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name])
                                    
                                    """
                                    tensor_from_model = self.model_graph.get_tensor_by_name(icand_name)
                                    if tensor_from_model is not None:
                                        if tensor_from_model.shape.dims is None:
                                            tensor_shape_result = '1x1'
                                        else:
                                            valid_tensor_shapes = [d.__str__() for d in tensor_from_model.shape.dims if d.__str__() != '']
                                            tensor_shape_result = 'x'.join(valid_tensor_shapes)
                                        input_cand_shapes_list.extend([tensor_shape_result])
                                    else:      
                                        input_cand_shapes_list.extend(['UNKNOWN input for node: '+ node_name + ' input not found: ' + icand_name])
                                    """
                        
                    else:
                        input_cand_shapes_list.extend([icand_name])
                            
               
                io_shapes[node_name] = [input_cand_shapes_list, io_shapes[node_name][1]]
        
        filename_ops_metadata = self.temp_path + "io_shapes_"+str(time.time())[:10]+".txt"
        file_ops_metadata = open(filename_ops_metadata,'w')
        sorted_all_node_names = sorted(all_node_names)
        for opname in sorted_all_node_names:
            file_ops_metadata.write(opname+'\n')
        file_ops_metadata.close()
        
        """
        filename_ops_metadata = self.temp_path + "node_names_metadata_original_"+str(time.time())[:10]+".txt"      
        file_ops_metadata = open(filename_ops_metadata,'w')
        sorted_all_node_names_orig = sorted(all_node_names_orig)
        for opname in sorted_all_node_names_orig:
            file_ops_metadata.write(opname+'\n')
        file_ops_metadata.close()      
        """
 
 
    def find_final_shapes(self, node_output_shape, input_names_from_model, io_shapes):
        ''' 
        Find input tensor shapes using names from model and output shapes from 
        metadata graph. We need this because gathered metadata may not 
        contain information about input tensor shapes.
        
        Args:
            (in)  node_output_shape (dict): dictionary of node name and output tensor shape
            (in)  input_names_from_model (dict): dictionary of node name and its input names
            (out) io_shapes (dict): final result with op parameters
        '''         
        
        ops_not_in_model = set()
        ops_not_in_metadata =  set()
        
        for k, v in node_output_shape.items():
            result_shapes = []
            
            #if k == 'GradientDescent/update_embeddings/encoder/embedding_encoder/mul':
            #    import pdb
            #    pdb.set_trace()
                            
            if k in input_names_from_model:
                all_input_names = input_names_from_model[k]
                if not all_input_names:
                    result_shapes.extend(['(no input)'])
                else:
                    for inp in all_input_names:
                        if inp in node_output_shape:
                            _tensor_shape, _device_type, _device_number = node_output_shape[inp]
                            if not _tensor_shape:
                                _tensor_shape = ['(None)']   
                            result_shapes.extend(_tensor_shape)
                        else:
                            tensor_from_model = self.model_graph.get_tensor_by_name(inp)
                            if tensor_from_model is not None:
                                tensor_shape_result = '(not in metadata %s)' % inp.replace('/','|')
                                correct_short_name_used = False
                                if tensor_from_model.shape.dims is None:
                                    tensor_shape_result = '1x1'
                                else:
                                    valid_tensor_shapes = [d.__str__() for d in tensor_from_model.shape.dims if d.__str__() != '']
                                    
                                    if '?' in valid_tensor_shapes:
                                        cand1 =  inp.split(':')
                                        cand = cand1[:-1]
                                        t1 = ':'.join(cand)
                                         
                                        
                                        if t1 in node_output_shape:
                                            correct_short_name_used = True
                                            _tensor_shape, _device_type, _device_number = node_output_shape[t1]
                                            
                                            if not _tensor_shape:
                                                _tensor_shape = ['(None)']   
                                            valid_tensor_shapes = _tensor_shape
                                        else:   
                                            if t1.split('/')[-1] == 'BroadcastGradientArgs':
                                                tensor_shape_result = '1x1'
                                            else:
                                            
                                                if t1 in self.cf_temp:
                                                    cf_cand = self.cf_temp[t1]
                                                    
                                                    if cf_cand in node_output_shape:
                                                      
                                                        correct_short_name_used = True
                                                        _tensor_shape, _device_type, _device_number = node_output_shape[cf_cand]
                                                    
                                                        if not _tensor_shape:
                                                            _tensor_shape = ['(None)']   
                                                        valid_tensor_shapes = _tensor_shape
                                                    else:
                                                        ops_not_in_metadata.add(t1)
                                   
                                                else:
                                                    ops_not_in_metadata.add(t1)
                                    else:     
                                        if len(valid_tensor_shapes) == 1:
                                            valid_tensor_shapes.append('1')
                                    if not correct_short_name_used:
                                        tensor_shape_result = 'x'.join(valid_tensor_shapes)
                                    else:
                                        tensor_shape_result = ''.join(valid_tensor_shapes)
                            
                                    if tensor_shape_result=='':
                                        tensor_shape_result='1x1'
                                        
                                if not (tensor_shape_result[0] == '(' and tensor_shape_result[-1]==')'):
                                    result_shapes.append('('+tensor_shape_result+')')
                                else:
                                    result_shapes.append(tensor_shape_result)  
                            else:
                                result_shapes.append('(not in metadata %s)' % inp.replace('/','|'))
                                ops_not_in_metadata.add(inp)
                           
            else:
                
                cand1 =  k.split(':')
                if cand1[-1] == 'MEMCPYDtoH' or cand1[-1] == 'MEMCPYHtoD' or  cand1[-1] == 'MEMCPYDtoD':
                    result_shapes.append('(no input)')
                    ops_not_in_model.add(k)
                else:
                    cand = cand1[:-1]
                    t1 = ':'.join(cand)
                    
                    if t1 in node_output_shape:
                        _tensor_shape, _device_type, _device_number = node_output_shape[t1]
                        if not _tensor_shape:
                            _tensor_shape = ['(None)']   
                        result_shapes.extend(_tensor_shape)
                    else:
                        result_shapes.append('(no input)')
                        ops_not_in_model.add(k)
               

            if len(result_shapes) > 0:
                if result_shapes[0] == 'MEMCPYDtoH' or  result_shapes[0] == 'MEMCPYHtoD' or result_shapes[0] == 'MEMCPYDtoD':
                    io_shapes[k] = '(no input)'
            
            result_shapes.extend(['->'])
            result_shapes.extend(v)
            io_shapes[k] = result_shapes
        
        # save data for additional verification
        filename_ops_not_in_model = self.temp_path + "node_names_not_in_model_"+str(time.time())[:10]+".txt"
        filename_ops_not_in_metadata = self.temp_path + "node_names_not_in_metadata_"+str(time.time())[:10]+".txt"
        filename_final_shapes = self.temp_path + "node_final_shapes_"+str(time.time())[:10]+".txt"
        
        file_ops_not_in_model = open(filename_ops_not_in_model,'w')
        file_ops_not_in_metadata = open(filename_ops_not_in_metadata,'w')
        file_final_shapes = open(filename_final_shapes,'w')

        sorted_ops_not_in_model = sorted(ops_not_in_model)
        for opname in sorted_ops_not_in_model:
            file_ops_not_in_model.write(opname+'\n')
       
        sorted_ops_not_in_metadata = sorted(ops_not_in_metadata)
        for opname in sorted_ops_not_in_metadata:
            file_ops_not_in_metadata.write(opname+'\n')
        
        sorted_times = ( (key, value) for key, value in sorted(io_shapes.items(), key=lambda x: x[0], reverse=True))
        sorted_times = list(sorted_times)
 
        for opname in sorted_times:
            file_final_shapes.write(opname.__str__())
            file_final_shapes.write('\n')
               
        file_ops_not_in_model.close()
        file_ops_not_in_metadata.close()
        file_final_shapes.close()


    def update_op_total_time(self, step_stats, total_op_time):

        '''Adds op time from current step to total time for op
        Args:
            step_stats (RunMetadata): timings for current step
            total_op_time (dict): dictionary of op name and its total time
            for all measured sessions      
        '''
        
        for dev_stats in step_stats.dev_stats:
            # assume naming was consistent
            device_path = dev_stats.device.split('/')
            device_type = ''
            device_number = 0
            
            for dev_path in device_path:
                dev_path_parts = dev_path.split(':')
                if dev_path_parts[0] == 'device':
                   device_type = dev_path_parts[1]
                   device_number = dev_path_parts[2]
                
            # for each op, add additional information about device type and number
            # so we can group timings by op + parameters and compare each device performance
            for node_stat in dev_stats.node_stats:
                op_time = node_stat.op_end_rel_micros - node_stat.op_start_rel_micros
                # CPU/GPU/XLA_CPU/XLA_GPU
                if (device_type[-3:] == 'GPU' and device_path[-1] != 'stream:all') or device_type[-3:] == 'CPU':   
                    final_op_path = self.cleanup_op_path_from_metadata(node_stat.node_name)
                    if final_op_path in total_op_time:
                        total_op_time[final_op_path][0] += op_time
                        total_op_time[final_op_path][1] += 1
                    else:
                        total_op_time[final_op_path] = [op_time, 1]


    def group_by_op_type_and_params1(self, op_timings):
        grouped_timings = {}
        for k,v in op_timings.items():
            full_op_path = self.cleanup_op_path(k, group_by_type=True, invert=False)
            full_op_path_parts = full_op_path.split('/')
            # keep only device and i/o tensor shapes 
            ops_with_params_without_group_num = full_op_path_parts[-2:]
            ops_with_params_without_group_num = '/'.join(ops_with_params_without_group_num)
            if ops_with_params_without_group_num in grouped_timings:
                prev_values =  grouped_timings[ops_with_params_without_group_num]
                grouped_timings[ops_with_params_without_group_num] = [sum(x) for x in zip(prev_values, v)]
            else:
                grouped_timings[ops_with_params_without_group_num] = v
        return grouped_timings


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

        _timeline_file_name = self.temp_path + "timeline_at_step_%d_%s.json" % (self.current_session, str(time.time())[:10])              
        with open(_timeline_file_name,'w') as _timeline_file:
            _timeline_file.write(_chrome_trace) 
            print('Timeline saved in', _timeline_file.name)
   
   
    def show_results(self, tsv_file_name_for_chart):
    
        # *** Automate generation of data in krona format, launch Chrome to draw the final chart ***
        chart_tools_path = os.path.dirname(os.path.realpath(__file__)) + "/chart_tools/scripts/"
        result_file = self.temp_path + "chart_" + str(time.time())[:10] + ".html"
       
        bash_cmd_generate_chart_data = 'cd "%s" && ./ImportText.pl "%s" -o "%s"' % (chart_tools_path, tsv_file_name_for_chart, result_file)
        
        #subprocess.call(bash_cmd_generate_chart_data, shell=True)
        os.system(bash_cmd_generate_chart_data)

        print('\n*** Interactive pie chart with details is saved to %s\n' % result_file)

        # if training was launched with sudo for some reason, --no-sandbox option will be required here
        bash_cmd_launch_browser = 'google-chrome --no-sandbox "%s"' % (result_file)
        
        # will this help to keep chrome's sandbox on, even if training was launched with sudo?
        #bash_cmd_launch_browser = 'sudo -H -u $SUDO_USER google-chrome "%s"' % (result_file)
        
        print('Launching the browser to show generated chart: %s' % bash_cmd_launch_browser)
        subprocess.Popen(bash_cmd_launch_browser, shell=True)
        #os.system(bash_cmd_launch_browser)
        
        # immediatly break outside loop of training/inference
        sys.exit(0)
         
    
    def generate_results(self):
       
        sorted_times = ( (key, value) for key, value in sorted(self.op_time_total.items(), key=lambda x: list(x[1])[0], reverse=True))
        sorted_times = list(sorted_times)
 
        total_ops_time = 0.0
        num_unique_ops = len(sorted_times)
        for op_name, op_time in sorted_times:
          total_ops_time = total_ops_time + op_time[0]
            
        mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
        mean_all_ops_time_per_step = float(total_ops_time)/self.num_steps_analyzed
        denom_const = 0.01*self.num_steps_analyzed*mean_all_ops_time_per_step

        print("\nSanity check: total time for %d steps: %.3f sec., 1 step avg. time:  %.1f microsec., 1 step avg. time captured in metadata: %.1f microsec., number of unique ops: %d" % 
            (self.num_steps_analyzed, self.total_time_analyzed, 1000000.0*mean_time_per_step, mean_all_ops_time_per_step, num_unique_ops))
        
        # *** extract tensor shapes, create input file for chart generation ***
        
        # this is where we store timings of ops grouped by device+op_type+tensors
        results_for_op_device_shape = {}
        
        op_count = 0
        cumul_time = 0.0
        
        # prepare files for saving results
        tsv_file_name = self.temp_path + "data_"+str(time.time())[:10]+".tsv"
        tsv_file_name_for_chart = self.temp_path + "data_krona_format_"+str(time.time())[:10]+".tsv"
        
        tsv_file_obj = open(tsv_file_name,'w')
        tsv_file_obj_for_chart = open(tsv_file_name_for_chart,'w')

        tsv_file_writer = csv.writer(tsv_file_obj, delimiter='\t')
        tsv_file_writer_for_chart = csv.writer(tsv_file_obj_for_chart, delimiter='\t')

        header_tsv = ['Op rank by total time', 'Time of 1 call, microseconds', 'Number of calls in 1 run', 'Total time in 1 run', '% of total time', 'Cumulative time, microseconds', '% of total cumulative time', 'Op name', 'Device', 'Input/output tensor dimensions']
        header_tsv_for_chart = ['Node','Time']

        tsv_file_writer.writerow(header_tsv)
        tsv_file_writer_for_chart.writerow(header_tsv_for_chart)
        
        
        # finalize data for the chart
        for times_key, times_value in sorted_times:

          # extract short op name,
          # append tensor shape and device from 
          # model and/or metadata
          
          oppath = times_key.split('/')[::-1]
          opname_short = oppath[0]
          opname_short_bak = opname_short + ''
          times_value_bak = times_value[:]
          
          shape_str = ['N-A']
          if times_key in self.final_io_shapes:
              shape_str = self.final_io_shapes[times_key]
              shape_str = [shape_str[1][1] + '-' + shape_str[1][2], ''.join(shape_str[0]) + '->'  + ''.join(shape_str[1][0])]
          else:
            shape_str = ['GPU-0', '(no input)->(no output)']  
            print('This node was not found in self.final_io_shapes, however it existed in metadata graph: ',  times_key)
         
          
            
          """
          if times_key in self.final_io_shapes:
              shape_str = self.final_io_shapes[times_key]
              shape_str_bak = shape_str[:]
              # reformat node path to get device info on a chart before parameters
              # (TODO) simplify
              if len(shape_str) > 2:     
                  if (not (shape_str[-3])):
                        
                      cand1 =  times_key.split(':')
                      cand = cand1[:-1]
                      t1 = ':'.join(cand)

                      if t1 in self.final_io_shapes:
                          shape_str = self.final_io_shapes[t1]   
                          if len(shape_str[-3]) == 0:
                              shape_str[-3] = '(no output)'
                          else:
                              shape_str_orig = shape_str[:]
                              if type(shape_str[-3]) is list:
                                  shape_str[-3] = shape_str[-3][0]
                      else:
                          shape_str[-3] = '(no output)' 
                  else:
                      if len(shape_str[-3]) == 0:
                          shape_str[-3] = '(no output)'
                      else:
                          if type(shape_str[-3]) is list:
                              shape_str[-3] = shape_str[-3][0]
                  shape_str = [shape_str[-2] + '-' + shape_str[-1], ''.join(shape_str[:-2])]
          else:
              shape_str = ['GPU-0', '(no input)->(no output)'] # quick fix for MEMCPYDtoD 
              print('This node was not found in self.final_io_shapes, however it existed in metadata graph: ',  times_key)
          """
            
          num_calls_all_runs =  times_value[1]
          num_calls_per_run =  int(times_value[1]/self.num_steps_analyzed)
          
          op_time_all_runs = times_value[0]
          op_time_per_run_microsec = op_time_all_runs/(float(self.num_steps_analyzed))
          op_time_per_call_microsec = op_time_per_run_microsec/float(num_calls_per_run)
          
          cumul_time += op_time_per_run_microsec
          op_count = op_count + 1
                   
          opname_short_parts = opname_short.split('_')
          opname_short_parts = [p for p in opname_short_parts if not p.isdigit()]
          if len(opname_short_parts) == 0:
              opname_short_parts = [oppath[1]]
              
          opname_short = '_'.join(opname_short_parts)
          
          if opname_short=='':
              opname_short = oppath[1]
              opname_short_parts = opname_short.split('_')
              opname_short_parts = [p for p in opname_short_parts if not p.isdigit()]
              if len(opname_short_parts) == 0:
                  opname_short_parts = oppath[1]
              opname_short = '_'.join(opname_short_parts)
                            
          opname_short_cand1 = opname_short.split(':')
          
          if len(opname_short_cand1)>1:
              opname_short = opname_short_cand1[-1]
          
          # quick fix for MEMCPYDtoD 
          if len(shape_str) == 0:
              shape_str.append('GPU-0')
              shape_str.append('(no input)->(no output)')
          elif len(shape_str) == 1:
              shape_str.append('(no input)->(no output)')
          

          opname_for_summation = '/'.join([opname_short, shape_str[0], shape_str[1]]) # op, device, i/o shapes
          #opname_for_summation = opname_for_summation[0].upper() + opname_for_summation[1:]
           
          if opname_for_summation in results_for_op_device_shape:
              prev_val = results_for_op_device_shape[opname_for_summation] 
              results_for_op_device_shape[opname_for_summation] = [prev_val[0] + op_time_all_runs, prev_val[1] + num_calls_all_runs]
          else:
              results_for_op_device_shape[opname_for_summation] = [op_time_all_runs, num_calls_all_runs]

          current_row_output_for_chart_tsv = [op_time_per_run_microsec, shape_str[0], opname_short]
          current_row_output_for_chart_tsv.extend(oppath[1:])
          current_row_output_for_chart_tsv.append(shape_str[1])  
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

        print("\n2 - Total time for %d steps: %.3f sec., 1 step avg. time:  %.1f microsec., 1 step avg. time captured in metadata: %.1f microsec., number of unique ops: %d" % 
            (self.num_steps_analyzed, self.total_time_analyzed, 1000000.0*mean_time_per_step, mean_all_ops_time_per_step, num_unique_ops))

        op_rank = 0
        for times_key, times_value in sorted_times:
          num_calls_all_runs =  times_value[1]
          num_calls_per_run =  int(times_value[1]/self.num_steps_analyzed)
          
          op_time_all_runs = times_value[0]
          op_time_per_run_microsec = op_time_all_runs/float(self.num_steps_analyzed)
          op_time_per_call_microsec = op_time_per_run_microsec/float(num_calls_per_run)
          
          cumul_time += op_time_per_run_microsec
          op_rank = op_rank + 1
                   
          current_row_output = [op_rank, 
                                "%9.1f" % op_time_per_call_microsec,
                                "%d" % num_calls_per_run,
                                "%9.1f" % (op_time_per_run_microsec),
                                "%9.1f" % (100*op_time_all_runs/float(total_ops_time)),
                                "%9.1f" % (cumul_time),
                                "%9.1f" % (100*cumul_time/mean_all_ops_time_per_step)]
                                
          nodepathparts = times_key.split('/')                                 
          current_row_output.extend(nodepathparts)
          tsv_file_writer.writerow(current_row_output)

        tsv_file_obj.close()

        print('\n*** Brief summary of findings ***\n')
        self.print_distribution_summary(sorted_times, denom_const)
        print('\n*** I/O tensor shapes for kernel optimizations (for this specific mo5 del and batch size) are saved to %s ***\n' % tsv_file_name)
       
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
                self.update_op_total_time(self.session_metadata.step_stats, self.op_time_total)
                self.num_steps_analyzed += 1
                
                if not self.tracing_graph_parsed:
                    # If output shapes are intially unknown in a dynamic graph,
                    # we need to parse both session metadata and graph to extract this info
                    parsing_start_time = time.time()
                    #self.find_input_names_from_model(self.model_graph, self.node_input_names)
                    #self.find_output_shape_from_metadata(self.session_metadata.step_stats, self.node_output_shapes)
                    #self.find_final_shapes(self.node_output_shapes, self.node_input_names, self.final_io_shapes)
                    
                    self.find_input_names_output_shapes(self.session_metadata.step_stats, self.node_input_names, self.final_io_shapes)#input_names, output_shapes):
                    
                    
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
                
            self.generate_results()
            
"""
Some things to account for later on

add V2, V3 postfix candidates if _kernel_label_map defined:
 def _kernel_label_map(self, op_to_kernel_label_map):
    #EXPERIMENTAL: A context manager for setting kernel labels.
    This context manager can be used to select particular
    implementations of kernels within the scope of the context.
    For example:
        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
 // Metadata output (i.e., non-Tensor) for a single Run() call.
message RunMetadata {
  // Statistics traced for this step. Populated if tracing is turned on via the
  // "RunOptions" proto.
  // EXPERIMENTAL: The format and set of events may change in future versions.
  StepStats step_stats = 1;

  // The cost graph for the computation defined by the run call.
  CostGraphDef cost_graph = 2;

  // Graphs of the partitions executed by executors.
  repeated GraphDef partition_graphs = 3;
}

Main issues is that there is no info about input tensors, and no names of outputs
https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/framework/step_stats.proto

// Output sizes recorded for a single execution of a graph node.
message NodeOutput {
  int32 slot = 1;
  TensorDescription tensor_description = 3;
};

// Time/size stats recorded for a single execution of a graph node.
message NodeExecStats {
  // TODO(tucker): Use some more compact form of node identity than
  // the full string name.  Either all processes should agree on a
  // global id (cost_id?) for each node, or we should use a hash of
  // the name.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
  DT_UINT32 = 22;
  DT_UINT64 = 23;

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101;
  DT_DOUBLE_REF = 102;
  DT_INT32_REF = 103;
  DT_UINT8_REF = 104;
  DT_INT16_REF = 105;
  DT_INT8_REF = 106;
  DT_STRING_REF = 107;
  DT_COMPLEX64_REF = 108;
  DT_INT64_REF = 109;
  DT_BOOL_REF = 110;
  DT_QINT8_REF = 111;
  DT_QUINT8_REF = 112;
  DT_QINT32_REF = 113;
  DT_BFLOAT16_REF = 114;
  DT_QINT16_REF = 115;
  DT_QUINT16_REF = 116;
  DT_UINT16_REF = 117;
  DT_COMPLEX128_REF = 118;
  DT_HALF_REF = 119;
  DT_RESOURCE_REF = 120;
  DT_VARIANT_REF = 121;
  DT_UINT32_REF = 122;
  DT_UINT64_REF = 123;
}

ptb 1 input to recheck:  Train/Model/sequence_loss/Sum
input Tensor("Train/Model/sequence_loss/Reshape_3:0", shape=(?, ?), dtype=float32)
input Tensor("Train/Model/sequence_loss/Sum/reduction_indices:0", shape=(1,), dtype=int32)


"""