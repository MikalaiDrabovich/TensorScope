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

Across 100 steps on batch size = 128.

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

class SessionCharacterized(tf.Session):
    
    # Set to True to quickly check for any side effects from pasted snippets
    disable_all = False
    
    timing_map = dict()
    node_input = dict()
    node_output = dict()
    output_dimension = dict()
    final_output_dimension = dict()

    current_session = 0
    num_sessions_analyzed = 0
    total_time = 0
    tracing_graph_parsed = False
    timeline_saved = False

    # we'll skip 2 runs at start and 2 runs at the end 
    total_num_sessions = 6 + 4
    
    def accumulate_time(self, step_stats, map_op_time):
        '''Adds op time from current step to total time for op
        Args:
            step_stats (RunMetadata): timings for current step
            map_op_time (dict): dictionary of op name and its total time for training session
        '''
        for dev_stats in step_stats.dev_stats:
            for node_stat in dev_stats.node_stats:
                op_time = node_stat.op_end_rel_micros - node_stat.op_start_rel_micros
                
                if node_stat.node_name in map_op_time:
                    map_op_time[node_stat.node_name][0] += op_time
                    map_op_time[node_stat.node_name][1] += 1
                else:
                    map_op_time[node_stat.node_name] = [0, 0]
                    

    def map_node_to_ionames(self, tensorscope_model_graph_for_names, tensorscope_node_input, tensorscope_node_output):
        all_ops = tensorscope_model_graph_for_names.get_operations()
        print("Total number of operations in a graph: ", len(all_ops))
        for node in all_ops:
            input_names = []
            output_names = []
            
              
            for (i, inp) in enumerate(node.inputs):
                
                ionodename = inp.name
                
                # Strip device placement number from name.
                # Disable to distinguish ops also by device number.
                ionodename_before = ionodename
                words_out = ionodename.split(":") 
                if len(words_out)>1:
                    ionodename = words_out[-2]
                    
                input_names.append(ionodename)
           
            for (i, outp) in enumerate(node.outputs):
                ionodename = outp.name
                 
                # Strip device placement number from name.
                # Disable to distinguish ops also by device number.
                ionodename_before = ionodename
                words_out = ionodename.split(":") 
                if len(words_out)>1:
                    ionodename = words_out[-2]
                               
                output_names.append(ionodename)
           
            tensorscope_node_input[node.name] =  input_names
            tensorscope_node_output[node.name] =  output_names
            

    def map_node_to_output_shape(self, step_stats, tensorscope_output_dimension):
        acc = 0
        for dev_stats in step_stats.dev_stats:
            for node_stat in dev_stats.node_stats:
                
                if not node_stat.output:
                    continue

                output_shapes = []
                for (i, node_stat_out) in enumerate(node_stat.output):
                
                    node_stat_dims = node_stat_out.tensor_description.shape.dim
                    node_stat_shape = tensor_shape.TensorShape(
                        [d.size for d in node_stat_dims])
                    
                    tensorscope_shape_str = node_stat_shape.__str__()
                    output_shapes.append(tensorscope_shape_str)
                
                ionodename = node_stat.node_name
                
                # Strip device placement number from name.
                # Disable to distinguish ops also by device number.
                ionodename_before = ionodename
                words_out = ionodename.split(":") 
                if len(words_out)>1:
                    ionodename = words_out[-2]
                    
                tensorscope_output_dimension[ionodename] = output_shapes
                
                
    def map_node_to_io_shapes(self, tensorscope_output_dimension, tensorscope_node_input, tensorscope_node_output, tensorscope_final_output_dimension):
         
        for k,v in tensorscope_output_dimension.items():
            final_io_shapes = []
            
            if k in tensorscope_node_input:
                all_input_names = tensorscope_node_input[k]
                
                for inp in all_input_names:
                    if inp in tensorscope_output_dimension:                
                        final_io_shapes.extend(tensorscope_output_dimension[inp])
                    else:
                        final_io_shapes.extend(["()"])
            else:
                final_io_shapes.extend(["()"])

            final_io_shapes.extend(['->'])
            final_io_shapes.extend(v)
             
            tensorscope_final_output_dimension[k] = final_io_shapes
    
        
    def characterize(self, *args, max_iter = 6):
        # we'll skip 2 runs at start and 2 runs at the end 
        self.total_num_sessions = max_iter + 4
        
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
                  self.total_time += self.current_session_time
                  self.accumulate_time(self.session_metadata.step_stats, self.timing_map)
                  self.num_sessions_analyzed += 1
                  
                  if not self.tracing_graph_parsed:
                      # If output shapes are intially unknown in a dynamic graph,
                      # we need to parse both session metadata and graph to extract this info
                      parsing_start_time = time.time()
                      self.map_node_to_ionames(self.model_graph, self.node_input, self.node_output)
                      self.map_node_to_output_shape(self.session_metadata.step_stats, self.output_dimension)
                      self.map_node_to_io_shapes(self.output_dimension, self.node_input, self.node_output, self.final_output_dimension)
                      parsing_end_time = time.time()
                      self.tracing_graph_parsed = True
                      print("Session graph and metadata parsed in %4.4f seconds" % (parsing_end_time - parsing_start_time))                    

          else:
              print("Metadata was not collected, check calls to session.run()")
            
          if not self.timeline_saved:
              _timeline = timeline.Timeline(self.session_metadata.step_stats)
              _chrome_trace = _timeline.generate_chrome_trace_format(show_memory=True)
              with open("tensorflow_timeline_"+str(time.time())[:10]+".json",'w') as _timeline_file:
                  _timeline_file.write(_chrome_trace) 
                  print('Timeline saved in', _timeline_file.name)
              self.timeline_saved = True
          
          if self.current_session % 100 == 0:
              print("Step #%d/%d completed in %4.4f seconds" % (self.current_session, self.total_num_sessions, self.current_session_time))

          # create chart
          if self.num_sessions_analyzed == max_iter:
              if not self.disable_all:
                if self.num_sessions_analyzed < max_iter:
                    print('self.num_sessions_analyzed: ', self.num_sessions_analyzed)
                    print('Training stats were not captured - more training steps required')
                    return -1
                  
                sorted_times = ( (key, value) for key, value in sorted(self.timing_map.items(), key=lambda x: list(x[1])[0], reverse=True))
                sorted_times = list(sorted_times)
                total_ops_time = 0

                op_count = 0
                top_k_ops_time = 0
                
                # Distribution of time among top k ops - useful for large models with thousands of ops.
                # This should get a rough idea of what needs to be optimised in the first place.
                # Value of k is equally spaced in log10 to get a clearer picture of what's going on.
                k_values = np.round(np.power(10, np.linspace(0, np.log10(len(sorted_times)), num=10, endpoint=True)).astype(np.double))
                top_k_time = np.zeros(len(k_values))
                
                bin_count = 0
                for op_name, op_time in sorted_times:
                  op_count = op_count + 1
                  op_time_total = op_time[0]
                  total_ops_time = total_ops_time + op_time_total
                  top_k_ops_time = top_k_ops_time + op_time_total
                  
                  if op_count >= k_values[bin_count]:
                    top_k_time[bin_count] = top_k_ops_time
                    bin_count = bin_count + 1
                    
                # sort ops and create tsv files
                tsv_file_name = "all_info_sorted_"+str(time.time())[:10]+".tsv"
                tsv_file_name_for_chart = "data_for_chart_"+str(time.time())[:10]+".tsv"
                
                tsv_file_obj = open(tsv_file_name,'w')
                tsv_file_obj_for_chart = open(tsv_file_name_for_chart,'w')

                tsv_file_writer = csv.writer(tsv_file_obj, delimiter='\t')
                tsv_file_writer_for_chart = csv.writer(tsv_file_obj_for_chart, delimiter='\t')

                header_tsv = ['Time Rank', '% of total', 'Cumulative %', 'Wall time, ms', '# of calls', 'Op name', 'Input/output tensor shape']
                header_tsv_for_chart = ['Node','Time']

                tsv_file_writer.writerow(header_tsv)
                tsv_file_writer_for_chart.writerow(header_tsv_for_chart)

                op_count = 0
                cumul_time = 0.0
                for times_key, times_value in sorted_times:
                  op_count = op_count + 1
                  op_time_total = times_value[0]
                  
                  shape_str = ""
                  if times_key in self.final_output_dimension:
                      shape_str = self.final_output_dimension[times_key]
                  else:
                      current_node = times_key
                      words_out = times_key.split(":") 
                      if len(words_out)>1:
                          current_node = words_out[-2]
                      
                      if current_node in self.final_output_dimension:
                          shape_str = self.final_output_dimension[current_node]
                      else:
                          shape_str = 'N/A'
                  
                  cumul_time += 100.0*op_time_total/float(total_ops_time)
                  
                  current_row_output =  [op_count, 
                                                    "%3.2f" % (100.0*op_time_total/float(total_ops_time)), 
                                                    "%3.2f" % cumul_time, op_time_total/(1000.0*float(self.num_sessions_analyzed)),
                                                    times_value[1],
                                                    times_key, 
                                                    ' '.join(shape_str)]
                                                    
                  current_row_output_for_chart_tsv = [op_time_total/float(self.num_sessions_analyzed)]
                  nodepath = times_key.split('/')
                  
                  # remove duplicates in node name, e.g. MatMul:MatMul -> MatMul
                  double_name =  nodepath[-1].split(':')
                  if len(double_name)==2 and double_name[0]==double_name[1]:
                     nodepath[-1] = double_name[0]
                     print("Removing redundancy in node name from ", double_name, " to ", [nodepath[-1]])
                  
                  current_row_output_for_chart_tsv.extend(nodepath)
                  current_row_output_for_chart_tsv.append(''.join(shape_str))
                  
                  tsv_file_writer.writerow(current_row_output)
                  tsv_file_writer_for_chart.writerow(current_row_output_for_chart_tsv)

                     
                print("Total time for all ops: %.3f seconds" % self.total_time)
                print("Number of analyzed session runs (skipping the first and the last): ", self.num_sessions_analyzed)
                
                microsec_num = 1000000.0
                mean_time_per_step = float(self.total_time)/self.num_sessions_analyzed
                mean_all_ops_time_per_step = float(total_ops_time)/(self.num_sessions_analyzed*microsec_num)
                denom_const = 0.01*microsec_num*self.num_sessions_analyzed*mean_all_ops_time_per_step

                print("Mean time for one iteration: %.3f seconds" % (mean_time_per_step))
                for k_count in range(len(top_k_time)):
                  print("Cumulative time for top %d ops: %5.5f sec out of %5.5f sec (%5.1f%%)" % (k_values[k_count],
                        top_k_time[k_count]/(microsec_num*self.num_sessions_analyzed),
                        mean_all_ops_time_per_step,
                        top_k_time[k_count]/denom_const))
                
                # launch TensorScope
                pwd_path = os.path.dirname(os.path.realpath(__file__))
                dir = pwd_path + "/TensorScope/scripts/"
                input_file = tsv_file_name_for_chart
                result_file = pwd_path + "/result_"+str(time.time())[:10]+".html"
                cmd_run = 'cd %s && ./ImportText.pl %s/%s -o %s && google-chrome --no-sandbox %s && cd -' % (dir, pwd_path, input_file, result_file, result_file)
                subprocess.Popen(cmd_run, shell=True)
                #sys.exit()
                    


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
    _ = session.characterize(target, max_iter=42)
      
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
      default=50,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
