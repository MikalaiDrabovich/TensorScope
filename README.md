# TensorScope

![Alt text](screenshot_tensorscope.png?raw=true "Screenshot")

Find bottlenecks in any complex model within minutes, with exact parameters, even for dynamically instantiated graphs.
- Produces interactive pie charts and allows easy navigation through complex hierarchy of ops.
- Compared to Tensorflow timeline, aggregates across op type and tensor dimensions to allow targeted tuning of top-k kernels.
- Chart generation is based on Krona tools: https://github.com/marbl/Krona/wiki/KronaTools

## Usage
1. See tensorscope.py
2. When Chrome is launched, decrease/increase "Max depth" on  a chart to quickly navigate through ops hierarchy. 
3. Additionally, see results in generated plain text tsv file in output_dir.

## Contact
nick.drabovich@amd.com
