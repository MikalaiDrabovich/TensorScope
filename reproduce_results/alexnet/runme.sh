#!/bin/bash

MODEL=alexnet

# new experiment will overwrite all files in ../../results/model_name
# make sure to make its copy to keep results for previous hardware setup 

LOG_FILE=$MODEL'_log.txt'
OUTPUT_DIR="../../results/$MODEL"
if [ ! -d $OUTPUT_DIR ]; then
  mkdir $OUTPUT_DIR
fi
sudo python3 $MODEL.py 2>&1 | tee $OUTPUT_DIR/$LOG_FILE

# generate pie chart
cd '../../chart_tools/scripts'
./ImportText.pl "../../results/$MODEL/data_for_pie_chart.tsv" -o "../../results/$MODEL/pie_chart.html"
wd=`pwd`
echo Results saved to "$wd/../../results/$MODEL/pie_chart.html"
#google-chrome --no-sandbox "../../results/$MODEL/pie_chart.html"

