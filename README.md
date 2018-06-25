# TensorScope

![Alt text](img/seq2seq_depth_2.png?raw=true "depth_2")
![Alt text](img/seq2seq_depth_3.png?raw=true "depth_3")
![Alt text](img/seq2seq_depth_max.png?raw=true "depth_max")

Find bottlenecks in any complex model within minutes, with exact parameters, even for dynamically instantiated graphs.
- Produces interactive pie charts and allows easy navigation through complex hierarchy of ops.
- Compared to Tensorflow timeline, aggregates across op type and tensor dimensions to allow targeted tuning of top-k kernels.
- Chart generation is based on Krona tools: https://github.com/marbl/Krona/wiki/KronaTools
- For examples of generated interactive pie charts (html5) see /results

## Usage
1. See tensorscope.py
2. When Chrome is launched, decrease/increase "Max depth" on  a chart to quickly navigate through ops hierarchy. 
3. Additionally, see results in generated plain text tsv file in output_dir.

## Contact
nick.drabovich@amd.com
