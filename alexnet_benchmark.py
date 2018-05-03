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

# ========== TensorScope Snippet 1 - paste right after imports =================
#Optional imports:
import numpy as np
#from collections import defaultdict


# For creating results and launching browser from bash
import subprocess
import os

import csv
from tensorflow.python.client import timeline
from tensorflow.python.framework import tensor_shape
   
def tensorscope_accumulate_time(step_stats, map_op_time):
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
                

def tensorscope_map_node_to_ionames(tensorscope_model_graph_for_names, tensorscope_node_input, tensorscope_node_output):
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
        

def tensorscope_map_node_to_output_shape(step_stats, tensorscope_output_dimension):
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
            
            
def tensorscope_map_node_to_io_shapes(tensorscope_output_dimension, tensorscope_node_input, tensorscope_node_output, tensorscope_final_output_dimension):
     
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
       
        # alternative ways to conscisely format found i/o tensor shape 
        #final_io_shapes = [str1.replace('(,', '(1x') for str1 in  final_io_shapes]   
        #final_io_shapes = [str1.replace(',)', '(x1') for str1 in  final_io_shapes]   
        #final_io_shapes = [str1.replace(',', 'x') for str1 in  final_io_shapes]        
        #final_io_shapes = [str1.replace('(', '') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace(')', '') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace(' ', '') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace('()', '(Scalar)') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace('x)', 'x1)') for str1 in  final_io_shapes] 
        #final_io_shapes = [str1.replace('(x', '(1x') for str1 in  final_io_shapes] 
         
        tensorscope_final_output_dimension[k] = final_io_shapes
        
# ========== End of TensorScope Snippet 1 ========== 





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
  
  
  # ====== TensorScope Snippet 2 - paste right before main training loop ===== 
  
  # Set to True to quickly check for any side effects from pasted snippets
  tensorscope_disable_all = False
  
  tensorscope_timing_map = dict()
  tensorscope_node_input = dict()
  tensorscope_node_output = dict()
  tensorscope_output_dimension = dict()
  tensorscope_final_output_dimension = dict()

  tensorscope_current_session = 0
  tensorscope_num_sessions_analysed = 0
  tensorscope_total_time = 0
  tensorscope_tracing_graph_parsed = False
  tensorscope_timeline_saved = False

  # Set to the max loop value of main loop below, typically a total number of batches
  # for example, for alexnet_benchmark.py this will be FLAGS.num_batches + num_steps_burn_in
  tensorscope_max_sessions =  FLAGS.num_batches + num_steps_burn_in #REPLACE_THIS_WITH_MAX_LOOP_ITERATION # change to FLAGS.num_batches + num_steps_burn_in
  
  # ========== End of TensorScope Snippet 2 ==================================    

  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
  
    start_time = time.time()
     
    #==========  TensorScope Snippet 3 ======
    # Paste this snippet to replace the line with session.run(). 
    # Add two more arguments to new session.run()
    
    if tensorscope_disable_all:
      # original call
      _ = session.run(target)
    else:
      tensorscope_current_session = i #REPLACE_THIS_TO_THE_LOOP_ITERATION # i, step etc. for alexnet_benchmark.py should be i
      tensorscope_session_metadata = tf.RunMetadata()
      tensorscope_session_start_time = time.time()
          
      # now with two more arguments:
      _ = session.run(target,
                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                      run_metadata=tensorscope_session_metadata)
              
      tensorscope_current_session_time = time.time() - tensorscope_session_start_time       
      if tensorscope_session_metadata is not None:
          tensorscope_model_graph = tf.get_default_graph()
          
          # skip two first and two last session runs
          if tensorscope_current_session > 1 and tensorscope_current_session < (tensorscope_max_sessions - 1): 
              tensorscope_total_time += tensorscope_current_session_time
              tensorscope_accumulate_time(tensorscope_session_metadata.step_stats, tensorscope_timing_map)
              tensorscope_num_sessions_analysed += 1
              
              if not tensorscope_tracing_graph_parsed:
                  # If output shapes are intially unknown in a dynamic graph,
                  # we need to parse both session metadata and graph to extract this info
                  tensorscope_parsing_start_time = time.time()
                  tensorscope_map_node_to_ionames(tensorscope_model_graph, tensorscope_node_input, tensorscope_node_output)
                  tensorscope_map_node_to_output_shape(tensorscope_session_metadata.step_stats, tensorscope_output_dimension)
                  tensorscope_map_node_to_io_shapes(tensorscope_output_dimension, tensorscope_node_input, tensorscope_node_output, tensorscope_final_output_dimension)
                  tensorscope_parsing_end_time = time.time()
                  tensorscope_tracing_graph_parsed = True
                  print("Session graph and metadata parsed in %4.4f seconds" % (tensorscope_parsing_end_time - tensorscope_parsing_start_time))                    

      else:
          print("Metadata was not collected, check calls to session.run()")
        
      if not tensorscope_timeline_saved:
          tensorscope_timeline = timeline.Timeline(tensorscope_session_metadata.step_stats)
          tensorscope_chrome_trace = tensorscope_timeline.generate_chrome_trace_format(show_memory=True)
          with open("tensorflow_timeline_"+str(time.time())[:10]+".json",'w') as tensorscope_timeline_file:
              tensorscope_timeline_file.write(tensorscope_chrome_trace) 
              print('Timeline saved in', tensorscope_timeline_file.name)
          tensorscope_timeline_saved = True
      
      if tensorscope_current_session % 100 == 0:
          print("Step #%d/%d completed in %4.4f seconds" % (tensorscope_current_session, tensorscope_max_sessions, tensorscope_current_session_time))

      # ========== End of TensorScope snippet 3 ==================================
                        
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
         
  # ====== TensorScope Snippet 4 - paste right after main training loop =====
  if not tensorscope_disable_all:
    if tensorscope_num_sessions_analysed < 5:
        print('Training stats were not captured - more training steps required')
        return -1
      
    tensorscope_sorted_times = ( (key, value) for key, value in sorted(tensorscope_timing_map.items(), key=lambda x: list(x[1])[0], reverse=True))
    tensorscope_sorted_times = list(tensorscope_sorted_times)
    tensorscope_total_ops_time = 0

    tensorscope_op_count = 0
    tensorscope_top_k_ops_time = 0
    
    # Distribution of time among top k ops - useful for large models with thousands of ops.
    # This should get a rough idea of what needs to be optimised in the first place.
    # Value of k is equally spaced in log10 to get a clearer picture of what's going on.
    tensorscope_k_values = np.round(np.power(10, np.linspace(0, np.log10(len(tensorscope_sorted_times)), num=10, endpoint=True)).astype(np.double))
    tensorscope_top_k_time = np.zeros(len(tensorscope_k_values))
    
    tensorscope_bin_count = 0
    for tensorscope_op_name, tensorscope_op_time in tensorscope_sorted_times:
      tensorscope_op_count = tensorscope_op_count + 1
      tensorscope_op_time_total = tensorscope_op_time[0]
      tensorscope_total_ops_time = tensorscope_total_ops_time + tensorscope_op_time_total
      tensorscope_top_k_ops_time = tensorscope_top_k_ops_time + tensorscope_op_time_total
      
      if tensorscope_op_count >= tensorscope_k_values[tensorscope_bin_count]:
        tensorscope_top_k_time[tensorscope_bin_count] = tensorscope_top_k_ops_time
        tensorscope_bin_count = tensorscope_bin_count + 1
        
    # sort ops and create tsv files
    tensorscope_tsv_file_name = "tensorscope_all_info_sorted_"+str(time.time())[:10]+".tsv"
    tensorscope_tsv_file_name_for_chart = "tensorscope_data_for_chart_"+str(time.time())[:10]+".tsv"
    
    tensorscope_tsv_file_obj = open(tensorscope_tsv_file_name,'w')
    tensorscope_tsv_file_obj_for_chart = open(tensorscope_tsv_file_name_for_chart,'w')

    tensorscope_tsv_file_writer = csv.writer(tensorscope_tsv_file_obj, delimiter='\t')
    tensorscope_tsv_file_writer_for_chart = csv.writer(tensorscope_tsv_file_obj_for_chart, delimiter='\t')

    tensorscope_header_tsv = ['Time Rank', '% of total', 'Cumulative %', 'Wall time, ms', '# of calls', 'Op name', 'Input/output tensor shape']
    tensorscope_header_tsv_for_chart = ['Node','Time']

    tensorscope_tsv_file_writer.writerow(tensorscope_header_tsv)
    tensorscope_tsv_file_writer_for_chart.writerow(tensorscope_header_tsv_for_chart)

    tensorscope_op_count = 0
    tensorscope_cumul_time = 0.0
    for tensorscope_times_key, tensorscope_times_value in tensorscope_sorted_times:
      tensorscope_op_count = tensorscope_op_count + 1
      tensorscope_op_time_total = tensorscope_times_value[0]
      
      tensorscope_shape_str = ""
      if tensorscope_times_key in tensorscope_final_output_dimension:
          tensorscope_shape_str = tensorscope_final_output_dimension[tensorscope_times_key]
      else:
          tensorscope_current_node = tensorscope_times_key
          words_out = tensorscope_times_key.split(":") 
          if len(words_out)>1:
              tensorscope_current_node = words_out[-2]
          
          if tensorscope_current_node in tensorscope_final_output_dimension:
              tensorscope_shape_str = tensorscope_final_output_dimension[tensorscope_current_node]
          else:
              tensorscope_shape_str = 'N/A'
      
      tensorscope_cumul_time += 100.0*tensorscope_op_time_total/float(tensorscope_total_ops_time)
      
      tensorscope_current_row_output =  [tensorscope_op_count, 
                                        "%3.2f" % (100.0*tensorscope_op_time_total/float(tensorscope_total_ops_time)), 
                                        "%3.2f" % tensorscope_cumul_time, tensorscope_op_time_total/(1000.0*float(tensorscope_num_sessions_analysed)),
                                        tensorscope_times_value[1],
                                        tensorscope_times_key, 
                                        ' '.join(tensorscope_shape_str)]
                                        
      tensorscope_current_row_output_for_chart_tsv = [tensorscope_op_time_total/float(tensorscope_num_sessions_analysed)]
      tensorscope_nodepath = tensorscope_times_key.split('/')
      
      # remove duplicates in node name, e.g. MatMul:MatMul -> MatMul
      tensorscope_double_name =  tensorscope_nodepath[-1].split(':')
      if len(tensorscope_double_name)==2 and tensorscope_double_name[0]==tensorscope_double_name[1]:
         tensorscope_nodepath[-1] = tensorscope_double_name[0]
         print("Name redundancy removed: ", tensorscope_nodepath)
      
      tensorscope_current_row_output_for_chart_tsv.extend(tensorscope_nodepath)
      tensorscope_current_row_output_for_chart_tsv.append(''.join(tensorscope_shape_str))
      
      tensorscope_tsv_file_writer.writerow(tensorscope_current_row_output)
      tensorscope_tsv_file_writer_for_chart.writerow(tensorscope_current_row_output_for_chart_tsv)

         
    print("Total time for all ops: %.3f seconds" % tensorscope_total_time)
    print("Number of analyzed session runs (skipping the first and the last): ", tensorscope_num_sessions_analysed)
    
    tensorscope_microsec_num = 1000000.0
    tensorscope_mean_time_per_step = float(tensorscope_total_time)/tensorscope_num_sessions_analysed
    tensorscope_mean_all_ops_time_per_step = float(tensorscope_total_ops_time)/(tensorscope_num_sessions_analysed*tensorscope_microsec_num)
    tensorscope_denom_const = 0.01*tensorscope_microsec_num*tensorscope_num_sessions_analysed*tensorscope_mean_all_ops_time_per_step

    print("Mean time for one batch: %.3f seconds" % (tensorscope_mean_time_per_step))
    for tensorscope_k_count in range(len(tensorscope_top_k_time)):
      print("Cumulative time for top %d ops: %5.5f sec out of %5.5f sec (%5.1f%%)" % (tensorscope_k_values[tensorscope_k_count],
            tensorscope_top_k_time[tensorscope_k_count]/(tensorscope_microsec_num*tensorscope_num_sessions_analysed),
            tensorscope_mean_all_ops_time_per_step,
            tensorscope_top_k_time[tensorscope_k_count]/tensorscope_denom_const))
    
    # launch TensorScope
    tensorscope_pwd_path = os.path.dirname(os.path.realpath(__file__))
    tensorscope_dir = tensorscope_pwd_path + "/TensorScope/scripts/"
    tensorscope_input_file = tensorscope_tsv_file_name_for_chart
    tensorscope_result_file = tensorscope_pwd_path + "/tensorscope_result_"+str(time.time())[:10]+".html"
    tensorscope_cmd_run = 'cd %s && ./ImportText.pl %s/%s -o %s && google-chrome --no-sandbox %s && cd -' % (tensorscope_dir, tensorscope_pwd_path, tensorscope_input_file, tensorscope_result_file, tensorscope_result_file)
    subprocess.Popen(tensorscope_cmd_run, shell=True)
        
    # ========== End of TensorScope snippet 4 =================

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
    sess = tf.Session(config=config)
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, pool5, "Forward")

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
      default=100,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
