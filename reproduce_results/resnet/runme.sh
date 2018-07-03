#!/bin/bash

MODEL=resnet

# new experiment will overwrite all files in ../../results/model_name
# make sure to make its copy to keep results for previous hardware setup 

LOG_FILE=${MODEL}'_log.txt'
OUTPUT_DIR="../../results/${MODEL}"
if [ ! -d $OUTPUT_DIR ]; then
  mkdir $OUTPUT_DIR
fi

resnet_model_path=resnet_model
if [ -d "$resnet_model_path" ]; then
  rm -rf "$resnet_model_path"
fi
mkdir "$resnet_model_path"

cd official
cd resnet
sudo python3 cifar10_main.py 2>&1 | tee ../../$OUTPUT_DIR/$LOG_FILE
cd ../..

# generate pie chart
cd '../../chart_tools/scripts'
./ImportText.pl "../../results/$MODEL/data_for_pie_chart.tsv" -o "../../results/$MODEL/pie_chart.html"
wd=`pwd`
echo Results saved to "$wd/../../results/$MODEL/pie_chart.html"
#google-chrome --no-sandbox "../../results/$MODEL/pie_chart.html"

