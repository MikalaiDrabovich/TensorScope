# TensorScope

![Alt text](docs/seq2seq_depth_2.png?raw=true "depth_2")
![Alt text](docs/seq2seq_depth_3.png?raw=true "depth_3")
![Alt text](docs/seq2seq_depth_max.png?raw=true "depth_max")

Find bottlenecks in any complex model within minutes, with exact parameters, even for dynamically instantiated graphs.
- Produces interactive pie charts and allows easy navigation through complex hierarchy of ops.
- Compared to Tensorflow timeline, aggregates across op type and tensor dimensions to allow targeted tuning of top-k kernels.
- Chart generation is based on Krona tools: https://github.com/marbl/Krona/wiki/KronaTools
- For examples of generated interactive pie charts (html5) see /results

## Usage
1. Simply ./run_me.sh in reproduce_results/
2. See results in pie_chart.html and data.tsv ./results_summary/
3. To add to your existing training setup - see module comment section in tensorscope.py

## Contact
nick.drabovich@amd.com


Update 09/2018
The project is in the process of moving to currently private https://github.com/ROCmSoftwarePlatform/TensorScope
Coming soon - bug fixes and changes are to be published here as well
