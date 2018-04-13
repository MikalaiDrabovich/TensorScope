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

# ========== Snippet to enable tracing 0/3 =================
from tensorflow.python.client import timeline
#from collections import defaultdict
from tensorflow.python.framework import tensor_shape
import numpy as np
import csv
import os
import subprocess
  
global_tracing_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

def accumulate_op_time(step_stats, map_op_time):
    '''Adds op time from current step to total time for op
    Args:
        step_stats (RunMetadata): timings for current step
        map_op_time (dict): dictionary of op name and its total time for training session
    '''
    for dev_stats in step_stats.dev_stats:
        for node_stat in dev_stats.node_stats:
            op_time = node_stat.op_end_rel_micros - node_stat.op_start_rel_micros
            #map_op_time[node_stat.node_name] += op_time
            
            if node_stat.node_name in map_op_time:
                set_ref = map_op_time[node_stat.node_name]
                set_ref[0] += op_time
                set_ref[1].add("Input size:")
                #print(map_op_time[node_stat.node_name])
                #exit(0)
            else:
                init_set = set()
                init_set.add("Input size:")
                map_op_time[node_stat.node_name] = [op_time, init_set]
                

def fill_maps_nodename_ionames(train_model_graph, map_nodename_inputnames, map_nodename_outputnames):
    all_ops = train_model_graph.get_operations()
    print("Total number of operations in a graph: ", len(all_ops))
    for node in all_ops:
        input_names = []
        output_names = []
        
          
        for (i, inp) in enumerate(node.inputs):
            # strip device placement num from name 
            ionodename = inp.name
            ionodename_before = ionodename

            words_out = ionodename.split(":") 
            if len(words_out)>1:
                ionodename = words_out[-2]
                
            input_names.append(ionodename)
       
        for (i, outp) in enumerate(node.outputs):
            # strip device placement num from name 
            ionodename = outp.name
             
            ionodename_before = ionodename

            words_out = ionodename.split(":") 
            if len(words_out)>1:
                ionodename = words_out[-2]
                           
            output_names.append(ionodename)
       
        map_nodename_inputnames[node.name] =  input_names
        map_nodename_outputnames[node.name] =  output_names
        

def fill_map_nodename_result_shape(step_stats, map_nodename_resultshape):
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
                
                shape_str = node_stat_shape.__str__()
                output_shapes.append(shape_str)
            
             # strip device placement num from name 
            ionodename = node_stat.node_name
            
            ionodename_before = ionodename

            words_out = ionodename.split(":") 
            if len(words_out)>1:
                ionodename = words_out[-2]
                
            map_nodename_resultshape[ionodename] = output_shapes
            
            
def fill_map_nodename_input_output_shapes(map_nodename_resultshape, map_nodename_inputnames, map_nodename_outputnames, final_map_shapes):
     
    for k,v in map_nodename_resultshape.items():
        final_io_shapes = []
        
        #final_io_shapes.append("In:")
        
        if k in map_nodename_inputnames:
            all_input_names = map_nodename_inputnames[k]
            
            for inp in all_input_names:
                if inp in map_nodename_resultshape:                
                    final_io_shapes.extend(map_nodename_resultshape[inp])
                else:
                    final_io_shapes.extend(["(Scalar)"])
        else:
            final_io_shapes.extend(["(Scalar)"])

        final_io_shapes.extend(['->'])
        final_io_shapes.extend(v)
       
        #final_io_shapes = [str1.replace('(,', '(1x') for str1 in  final_io_shapes]   
        #final_io_shapes = [str1.replace(',)', '(x1') for str1 in  final_io_shapes]   
        
        final_io_shapes = [str1.replace(',', 'x') for str1 in  final_io_shapes]        
        final_io_shapes = [str1.replace('(', '') for str1 in  final_io_shapes]
        final_io_shapes = [str1.replace(')', '') for str1 in  final_io_shapes]
        final_io_shapes = [str1.replace(' ', '') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace('()', '(Scalar)') for str1 in  final_io_shapes]
        #final_io_shapes = [str1.replace('x)', 'x1)') for str1 in  final_io_shapes] 
        #final_io_shapes = [str1.replace('(x', '(1x') for str1 in  final_io_shapes] 
         
        final_map_shapes[k] = final_io_shapes
        
# ========== End of snippet to enable tracing 0/3 ========== 





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
  
  
  # ========== Snippet to enable tracing 1/3 =================
  timing_map = dict()
  map_nodename_inputnames = dict()
  map_nodename_outputnames = dict()
  map_nodename_resultshape = dict()
  final_map_shapes = dict()

   
  total_time_all_sessions = 0
  total_session_runs = 0
  num_accumulated_sessions = 0
  op_info_found = False
  
  n_hidden = 12345
  global_train_batch_size = FLAGS.batch_size
  global_num_batches_total = FLAGS.num_batches
  # ========== End of snippet to enable tracing 1/3 ==========     

  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
  

    start_time = time.time()
    
    
    #_ = session.run(target)
    
    # ========== Snippet to enable tracing 2/3 =================
   
    total_session_runs = i
    current_step = i
    max_session_runs = FLAGS.num_batches + num_steps_burn_in
    
    session_start_time = time.time()
    disable_tracing = False
    
    if disable_tracing:
        _ = session.run(target)
        current_session_time = time.time() - session_start_time
        total_time_all_sessions = total_time_all_sessions + current_session_time
        num_accumulated_sessions += 1
               
    else:
        tracing_run_metadata = tf.RunMetadata()
        _ = session.run(target, options=global_tracing_options, run_metadata=tracing_run_metadata)
        current_session_time = time.time() - session_start_time
         
        if tracing_run_metadata is not None:
            train_model_graph = tf.get_default_graph()
            
            if total_session_runs > 0 and current_step < (max_session_runs - 1): # skip first and last session runs for stats

                total_time_all_sessions = total_time_all_sessions + current_session_time
  
                accumulate_op_time(tracing_run_metadata.step_stats, timing_map)
                num_accumulated_sessions += 1
                if not op_info_found:
                    start_time_fillmap = time.time()
                    fill_maps_nodename_ionames(train_model_graph, map_nodename_inputnames, map_nodename_outputnames)
                    fill_map_nodename_result_shape(tracing_run_metadata.step_stats, map_nodename_resultshape)
                    fill_map_nodename_input_output_shapes(map_nodename_resultshape, map_nodename_inputnames, map_nodename_outputnames, final_map_shapes)
                    
                    end_time_fillmap = time.time()
                    print("Graph parsed in %4.4f seconds" % (end_time_fillmap - start_time_fillmap))                    
                    op_info_found = True
        else:
            print("Metadata is None")
        
        
        #if current_step - last_stats_step >= steps_per_stats:    
        #fetched_timeline = timeline.Timeline(tracing_run_metadata.step_stats)
        #chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
        #with open("timeline_deepspeech_hidden_%d_batch_%d_timesteps_%d_step_%d.json" % (n_hidden, global_train_batch_size, global_num_batches_total, current_step),'w') as traces_file:
        #    traces_file.write(chrome_trace) 
        #    print('Traces saved in', traces_file.name)
        


    print("Batch #%d completed in %4.4f seconds" % (current_step, current_session_time))
    
    total_session_runs +=1
    # ========== End of snippet to enable tracing 2/3 ========== 
                        
    
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  print("Total experiment duration: %f sec." % total_duration)
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))
         
  # ========== Snippet to enable tracing 3/3 =================
  
  sorted_times = ( (key, value) for key, value in sorted(timing_map.items(), key=lambda x: list(x[1])[0], reverse=True))
  sorted_times = list(sorted_times)
  total_ops_time = 0 

  op_count = 0
  tot_time_in_top_k_ops = 0
  
  # top k ops, k is equally spaced in log10
  import numpy as np
  k_values = np.round(np.power(10, np.linspace(0, np.log10(len(sorted_times)), num=10, endpoint=True)).astype(np.double))
  top_k_time = np.zeros(len(k_values))
  
  bin_count = 0
  for k1,v1 in sorted_times:
      op_count = op_count + 1
      op_time_total = v1[0]
      total_ops_time = total_ops_time + op_time_total
      
      tot_time_in_top_k_ops = tot_time_in_top_k_ops + op_time_total
      if op_count >= k_values[bin_count]:
        top_k_time[bin_count] = tot_time_in_top_k_ops
        bin_count = bin_count + 1
 
  num_train_steps = num_accumulated_sessions
  if num_train_steps < 3:
      print('Training stats were not captured - more training steps required (increase number of --epoch)')
      return

  file_name_tsv_all_info = "tensorscope_all_data.tsv"
  file_name_tensorscope_input = "tensorscope_input_file.tsv"
   
  file_tsv_all_info = open(file_name_tsv_all_info,'w')
  file_tensorscope_input = open(file_name_tensorscope_input,'w')

  writer_file_tsv_all_info = csv.writer(file_tsv_all_info, delimiter='\t')#,quoting=csv.QUOTE_MINIMAL) quotechar='"', 
  writer_file_tensorscope_input = csv.writer(file_tensorscope_input, delimiter='\t')#,quoting=csv.QUOTE_MINIMAL) quotechar='"', 

  header_str_tsv = ['Time Rank', 'Time % total', 'Cumulative time %', 'Wall Time, ms', 'Op name', 'Input/output tensor shape']
  header_str_tensorscope_input = ['Node','Time']

  writer_file_tsv_all_info.writerow(header_str_tsv)
  writer_file_tensorscope_input.writerow(header_str_tensorscope_input)

  op_count = 0
  cumul_time = 0.0
  for k,v in sorted_times:
    op_count = op_count + 1
    op_time_total = v[0]
    
    shape_str = ""
    if k in final_map_shapes:
        #print('IO tensor dimensions: ', final_map_shapes[k])
        shape_str = final_map_shapes[k]
    else:
        orig_k = k
        ionodename = k
        words_out = k.split(":") 
        if len(words_out)>1:
            ionodename = words_out[-2]
        
        if ionodename in final_map_shapes:
            shape_str = final_map_shapes[ionodename]
            #print('IO tensor dimensions: ', final_map_shapes[ionodename])
        else:
            shape_str = 'N/A'
            #print('IO tensor dimensions not found for ', ionodename , " and ", orig_k)
    
    cumul_time += 100.0*op_time_total/float(total_ops_time)
    
    current_row_output =  [op_count, "%3.2f" % (100.0*op_time_total/float(total_ops_time)), "%3.2f" % cumul_time, op_time_total/(1000.0*float(num_train_steps)), k, ' '.join(shape_str)]
    current_row_output_for_chart_tsv = [op_time_total/float(num_train_steps)]
    nodepaths = k.split('/')
    
    # remove duplicates in node name, e.g. MatMul:MatMul -> MatMul
    doubled_names = nodepaths[-1].split(':')
    if len(doubled_names)==2 and doubled_names[0]==doubled_names[1]:
      nodepaths[-1] = doubled_names[0]
    
    current_row_output_for_chart_tsv.extend(nodepaths)
    current_row_output_for_chart_tsv.append(''.join(shape_str))
    
    writer_file_tsv_all_info.writerow(current_row_output)
    writer_file_tensorscope_input.writerow(current_row_output_for_chart_tsv)

       
  print("Total ops time: ", total_time_all_sessions)
  print("Number of analysed session runs (skipping the first and the last): ", num_accumulated_sessions)

  microsec_num = 1000000.0
  mean_time_per_step = float(total_time_all_sessions)/num_train_steps
  mean_allops_time_per_step = float(total_ops_time)/(num_train_steps*microsec_num)
  denom_const = 0.01*microsec_num*num_train_steps*mean_allops_time_per_step

  print("Mean time for one batch, sec: %5.5f" % (mean_time_per_step))
  for k_count in range(len(top_k_time)):
    print("Top %d ops time: %5.5f sec out of %5.5f sec (%5.1f%%)" % (k_values[k_count], top_k_time[k_count]/(microsec_num*num_train_steps), mean_allops_time_per_step, top_k_time[k_count]/denom_const))
  
  # launch TensorScope
  pwd_path = os.path.dirname(os.path.realpath(__file__))
  tensorscope_dir = pwd_path + "/TensorScope/scripts/"
  tensorscope_input_file = file_name_tensorscope_input
  tensorscope_result_file = pwd_path + "/tensorscope_result.html"
  cmd_run = 'cd %s && ./ImportText.pl %s/%s -o %s && google-chrome --no-sandbox %s l && cd -' % (tensorscope_dir, pwd_path, tensorscope_input_file, tensorscope_result_file, tensorscope_result_file)
  subprocess.Popen(cmd_run,shell=True)

  # ========== End of snippet to enable tracing 3/3 =================



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

    """
    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")
    """

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
