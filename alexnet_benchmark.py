# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Timing benchmark for AlexNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.s32

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = None


#Optional imports:
import numpy as np
#from collections import defaultdict


# For creating results and launching browser from bash
import subprocess
import os

import csv
from tensorflow.python.client import timeline
from tensorflow.python.framework import tensor_shape

# Finding op input and output tensor shapes.
# Complications/notes:
# - models may not have info about outputs, only inputs;
# - metadata 'streams' may not have info about inputs. The output info may also be absent (for example, in 'stream:all')
# 'stream:13', 'stream:15' may have only partial timings, we should use 'stream:all' to get op timings
# - op names in model and metadata may be slightly different:
# e.g. MatMul:MatMul vs. MatMul:0;
# - for correct chart generation, op placement device and 
# tensor shapes should be added to the node path.

class SessionCharacterized(tf.Session):
    
    # Set to True to disable all profiling
    disable_all = False
    
    node_input_names = dict()
    node_output_shapes = dict()
    final_io_shapes = dict()
    op_time_total = dict()
        
    current_session = 0
    num_steps_analyzed = 0
    total_time_analyzed = 0
    total_num_sessions = 0
    num_steps_to_analyze = 0        
    tracing_graph_parsed = False

    @staticmethod
    def convert_op_name_from_model(name_from_model):
        # Node name clean-up - removes ':0' at the end 
        # this may need to be adapted for a specific model
        common_name = name_from_model.split(':')
        if len(common_name) > 1:
            common_name = ":".join(common_name[:-1])
        
        return common_name
 
    @staticmethod
    def convert_op_name_from_metadata(name_from_metadata):
 
        current_node_path_parts = name_from_metadata.split('/')
        cnt = 0
        for path_part in current_node_path_parts:
            path_part_split = path_part.split(':')
            #remove repetitions in node name
            should_compact = True
            for asplit in path_part_split:
                if asplit != path_part_split[0]:
                    should_compact = False
                    break
            if should_compact:
                current_node_path_parts[cnt] = path_part_split[0]
            cnt+=1  
         
        common_name='/'.join(current_node_path_parts)
                    
        return common_name
                    
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
        for node in all_ops:
            input_names = []
            for (i, inp) in enumerate(node.inputs):
                input_names.append(self.convert_op_name_from_model(inp.name))

            input_names_from_model[node.name] =  input_names
            
                        
    def find_output_shape_from_metadata(self, step_stats, output_shape_device):
        ''' 
        Find node output parameters (tensor shapes)
        Args:
            (in)  step_stats (tensorflow.RunMetadata): run_metadata from session.run()
            (out) output_shape (dict): dictionary of node name and output tensor shape
        '''
        num_nodes_total = 0
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
                
                if self.convert_op_name_from_metadata(node_stat.node_name) in output_shape_device:
                    [output_shapes, _device_type, _device_number] = output_shape_device[self.convert_op_name_from_metadata(node_stat.node_name)]
                
                if node_stat.output:
                    for (i, node_stat_out) in enumerate(node_stat.output):
                        node_stat_dims = node_stat_out.tensor_description.shape.dim
                        node_stat_shape = tensor_shape.TensorShape([d.size for d in node_stat_dims])
                        if not node_stat_shape.__str__():
                            print('Possibe data corruption: node_stat_shape.__str__() is None in ', node_stat_out)
                            sys.exit(0)
                        output_shapes.append(node_stat_shape.__str__())
                        
                new_device = device_type
                if _device_type != '':
                    if  new_device != _device_type:
                        new_device = new_device + '+' + _device_type
                
                new_device_number = device_number
                if _device_number != '':
                    if  new_device_number != _device_number:
                        new_device_number = new_device_number + '+' + _device_number
                     
                output_shape_device[self.convert_op_name_from_metadata(node_stat.node_name)] = [output_shapes, new_device , new_device_number]
                            
        print("Total number of nodes in a metadata graph: ", num_nodes_total)


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
        
        for k, v in node_output_shape.items():
            result_shapes = []
            if k in input_names_from_model:
                all_input_names = input_names_from_model[k]
                
                for inp in all_input_names:
                    if inp in node_output_shape:
                        _tensor_shape, _device_type, _device_number = node_output_shape[inp]
                        if not _tensor_shape:
                            print('_tensor_shape is None: ', node_output_shape[inp])
                            _tensor_shape = ['()']   
                        result_shapes.extend(_tensor_shape)
                    else:
                        result_shapes.append('(const)')
            else:
                result_shapes.extend(['(no input)'])

            result_shapes.extend(['->'])
            result_shapes.extend(v)
            io_shapes[k] = result_shapes

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
                
                # for GPU we are interested only in aggregated stats 
                # from 'stream:all' and not another streams.
                # Also assuming that CPU doesn't have streams reported
                if (device_type == 'GPU' and device_path[-1] == 'stream:all') or device_type == 'CPU':
                    #if device_type == 'CPU':
                    #    print('%s was run on CPU %s and took %5.5f ms' % (node_stat.node_name, device_number, op_time/1000.0))
                        
                    final_op_path = self.convert_op_name_from_metadata(node_stat.node_name)
                    if final_op_path in total_op_time:
                        total_op_time[final_op_path][0] += op_time
                        total_op_time[final_op_path][1] += 1
                    else:
                        total_op_time[final_op_path] = [op_time, 1]
                        
                        
    def characterize(self, *args, num_steps = 10):

        # skip 2 runs at start and 2 runs at the end
        self.total_num_sessions = num_steps + 4
        self.num_steps_to_analyze = num_steps
         
        self.session_metadata = tf.RunMetadata()
        self.options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
         
        # Paste this snippet to replace the line with session.run(). 
        # Add two more arguments to new session.run()
        
        if self.disable_all:
          # original call
          _ = self.run( *args)
        else:
          self.current_session += 1
          self.session_metadata = tf.RunMetadata()
          self.session_start_time = time.time()
              
          # now with two more arguments:
          _ = self.run( *args,
                          options=self.options,
                          run_metadata=self.session_metadata)
                  
          self.current_session_time = time.time() - self.session_start_time       
          if self.session_metadata is not None:
              self.model_graph = tf.get_default_graph()
              
              # skip two first and two last session runs
              if self.current_session > 1 and self.current_session < (self.total_num_sessions - 2) : 
                  self.total_time_analyzed += self.current_session_time
                  #print('len self.session_metadata.step_stats: ', type(self.session_metadata.step_stats))
                  self.update_op_total_time(self.session_metadata.step_stats, self.op_time_total)
                  self.num_steps_analyzed += 1
                  
                  if not self.tracing_graph_parsed:
                      # If output shapes are intially unknown in a dynamic graph,
                      # we need to parse both session metadata and graph to extract this info
                      parsing_start_time = time.time()
                      self.find_input_names_from_model(self.model_graph, self.node_input_names)
                      self.find_output_shape_from_metadata(self.session_metadata.step_stats, self.node_output_shapes)
                      self.find_final_shapes(self.node_output_shapes, self.node_input_names, self.final_io_shapes)
                      parsing_end_time = time.time()
                      self.tracing_graph_parsed = True
                      print("Session graph and metadata parsed in %4.4f seconds" % (parsing_end_time - parsing_start_time))                    

          else:
              print("Metadata was not collected, check calls to session.run()")
          
          # save timeline at a specific step
          if self.current_session == 2:
              _timeline = timeline.Timeline(self.session_metadata.step_stats)
              _chrome_trace = _timeline.generate_chrome_trace_format(show_memory=True)
              _timeline_file_name = "timeline_step_%d_%s.json" % (self.current_session, str(time.time())[:10])              
              with open(_timeline_file_name,'w') as _timeline_file:
                  _timeline_file.write(_chrome_trace) 
                  print('Timeline saved in', _timeline_file.name)
          
          print("Step %d/%d(%d) completed in %4.4f seconds." % 
          (self.num_steps_analyzed, self.num_steps_to_analyze, self.total_num_sessions, self.current_session_time))

          # create chart
          if self.num_steps_analyzed == self.num_steps_to_analyze:
                  
              sorted_times = ( (key, value) for key, value in sorted(self.op_time_total.items(), key=lambda x: list(x[1])[0], reverse=True))
              sorted_times = list(sorted_times)
              
              total_ops_time = 0
              op_count = 0

              # Distribution of time among top k ops - useful for large models with thousands of ops.
              # This should get a rough idea of what needs to be optimised in the first place.
              # Value of k is equally spaced in log10 to get a clearer picture of what's going on.

              num_unique_ops = len(sorted_times)
              k_values = np.round(np.power(10, np.linspace(0, np.log10(num_unique_ops), num=10, endpoint=True)).astype(np.double))
              top_k_time = np.zeros(len(k_values))
              
              bin_count = 0
              for op_name, op_time in sorted_times:
                op_count = op_count + 1
                total_ops_time = total_ops_time + op_time[0]
                
                if op_count >= k_values[bin_count]:
                  top_k_time[bin_count] = total_ops_time
                  bin_count = bin_count + 1
                  
              # sort ops and create tsv files
              tsv_file_name = "results_"+str(time.time())[:10]+".tsv"
              tsv_file_name_for_chart = "data_for_chart_"+str(time.time())[:10]+".tsv"
              
              tsv_file_obj = open(tsv_file_name,'w')
              tsv_file_obj_for_chart = open(tsv_file_name_for_chart,'w')

              tsv_file_writer = csv.writer(tsv_file_obj, delimiter='\t')
              tsv_file_writer_for_chart = csv.writer(tsv_file_obj_for_chart, delimiter='\t')

              header_tsv = ['Op rank by time', 'Time per 1 call', 'Num of calls per run', 'Total time per run', 'Total time %', 'Total cumulative time', 'Total cumulative time %', 'Op name', 'Device', 'Input/output tensor shape']
              header_tsv_for_chart = ['Node','Time']

              tsv_file_writer.writerow(header_tsv)
              tsv_file_writer_for_chart.writerow(header_tsv_for_chart)

              op_count = 0
              cumul_time = 0.0

              microsec_num = 1000
              mean_time_per_step = float(self.total_time_analyzed)/self.num_steps_analyzed
              mean_all_ops_time_per_step = float(total_ops_time)/(self.num_steps_analyzed*microsec_num)
              denom_const = 0.01*microsec_num*self.num_steps_analyzed*mean_all_ops_time_per_step
 
              print("\nTotal time for %d steps: %.3f sec., mean time:  %.1f ms., mean time from metadata: %.1f ms., unique ops: %d" % 
                  (self.num_steps_analyzed, self.total_time_analyzed, microsec_num*mean_time_per_step, mean_all_ops_time_per_step, num_unique_ops))
              
              for times_key, times_value in sorted_times:

                shape_str = ['N/A']
                if times_key in self.final_io_shapes:
                    shape_str = self.final_io_shapes[times_key]
                    # reformat node path to get device info on a chart
                    if len(shape_str) > 2:     
                        if (not (shape_str[-3])):
                            shape_str[-3] = '(no output)'
                        else:
                            if len(shape_str[-3])<1:
                                shape_str[-3] = '(no output)'
                            else:
                                shape_str[-3] = shape_str[-3][0]
                        shape_str = [shape_str[-2] + '_' + shape_str[-1], ''.join(shape_str[:-2])]
                else:
                    print('This node was not found in self.final_io_shapes, however it existed in metadata graph: ',  times_key)
                    
                num_calls_all_runs =  times_value[1]
                num_calls_per_run =  int(times_value[1]/self.num_steps_analyzed)
                
                op_time_all_runs = times_value[0]
                op_time_per_run_ms = op_time_all_runs/(microsec_num*float(self.num_steps_analyzed))
                op_time_per_call_ms = op_time_per_run_ms/float(num_calls_per_run)
                
                cumul_time += op_time_per_run_ms
                op_count = op_count + 1
                         
                current_row_output = [op_count, 
                                      "%.3f ms" % op_time_per_call_ms,
                                      "%d" % num_calls_per_run,
                                      "%.3f ms" % (op_time_per_run_ms),
                                      "%.3f %%" % (100*op_time_all_runs/float(total_ops_time)),
                                      "%.3f ms" % (cumul_time),
                                      "%.3f %%" % (100*cumul_time/mean_all_ops_time_per_step),
                                      times_key]
                                      
                current_row_output.extend(shape_str)
                                                  
                current_row_output_for_chart_tsv = [op_time_per_run_ms]
                current_row_output_for_chart_tsv.extend(times_key.split('/'))
                current_row_output_for_chart_tsv.extend(shape_str)

                tsv_file_writer.writerow(current_row_output)
                tsv_file_writer_for_chart.writerow(current_row_output_for_chart_tsv)

              tsv_file_obj.close()
              tsv_file_obj_for_chart.close()


              for k_count in range(len(top_k_time)):
                print("Top-%d ops (%.1f%%)\t%.1f ms (%.1f%%)" % (k_values[k_count], 100*k_values[k_count]/float(num_unique_ops),
                      top_k_time[k_count]/(microsec_num*self.num_steps_analyzed),
                      top_k_time[k_count]/denom_const))
              
              # process results and launch browser to draw charts
              pwd_path = os.path.dirname(os.path.realpath(__file__))
              dir = pwd_path + "/TensorScope/scripts/"
              input_file = tsv_file_name_for_chart
              result_file = pwd_path + "/result_"+str(time.time())[:10]+".html"
              cmd_run = 'cd %s && ./ImportText.pl %s/%s -o %s && sudo -H -u $SUDO_USER google-chrome %s && cd -' % (dir, pwd_path, input_file, result_file, result_file)
              subprocess.Popen(cmd_run, shell=True)

              print('\n*** Details saved in %s ***\n' % tsv_file_name)
              sys.exit()
                    


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  """Build the AlexNet model.

  Args:
    images: Images Tensor

  Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # lrn2
  with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  return pool5, parameters


def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 0
  total_duration = 0.0
  total_duration_squared = 0.0
  
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
  
    start_time = time.time()

    #_ = session.run(target) 
    _ = session.characterize(target)
      
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
         
 
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
  
 
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pool5, parameters = inference(images)

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = SessionCharacterized(config=config)
    sess.run(init)

    # Run the forward benchmark.
    # (MDR: commented out to run only for training)
    # time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")
    

def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=200,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
