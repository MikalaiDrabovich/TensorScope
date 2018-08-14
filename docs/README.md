# How to

## Quick benchmark mode:

cd reproduce_results
./run_me.sh


## Add to your own model setup:

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
     (ts may also need to be defined as global object, see this case in nmt/nmt/model.py)
      
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
cumulative time (under 'Top-K ops vs total time ' and duplicated 
in /results/model_name/model_name_log.txt). If you see that, for example, 
top-10 ops (out of total 1000 ops) take 90% of time, it is quite likely that 
focusing optimization/tuning efforts on that 1% of ops could be more 
advantageous. The assumption of course is that full utilization of 
compute device is still far from being reached. 
It is possible to verify this assumption for a small number of ops 
(which have RegisterStatistics('flops') implemented) - see data 
in column 'G(FL)OP/S achieved'.

6) To compare to some other system, copy its 'data.tsv' to 
results/model_name directory, rename 'data.tsv' to 'data_from_another_system.tsv'.
See results in 'data_compared.tsv', unmatched ops will be saved
to 'data_unmatched_ops.tsv'

## 

If you have questions/comments please open an issue at 
https://github.com/MikalaiDrabovich/TensorScope or send e-mail to nick.drabovich@amd.com 


