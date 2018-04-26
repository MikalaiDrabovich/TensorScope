# TensorScope

![Alt text](screenshot_tensorscope.png?raw=true "Screenshot")

Find bottlenecks in any complex model within minutes, with exact parameters, even for dynamically instantiated graphs.
- Produces interactive pie charts and allows easy navigation through complex hierarchy of ops.
- Compared to Tensorflow timeline, aggregates across op type and tensor dimensions to allow targeted tuning of top-k kernels.
- Chart generation is based on Krona tools: https://github.com/marbl/Krona/wiki/KronaTools

## Usage
1. To start, copy 4 snippets from alexnet_benchmark.py and paste them into similar locations in your main training file (the one with iterations over batches and session.run() call). 
2. Set correct values for `REPLACE_THIS_WITH_MAX_LOOP_ITERATION` and `REPLACE_THIS_TO_THE_LOOP_ITERATION`
3. When Chrome is launched, decrease/increase "Max depth" to quickly navigate through hierarchy.

## Contact
Send PR or email to nick.drabovich@amd.com

