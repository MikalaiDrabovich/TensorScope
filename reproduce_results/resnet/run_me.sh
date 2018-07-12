#!/bin/bash

MODEL=resnet

# new experiment will overwrite all files in ../../results/model_name
# make sure to make its copy to keep results for previous hardware setup 

LOG_FILE=${MODEL}'_log.txt'
OUTPUT_DIR="../../results/${MODEL}"
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

RESNET_MODEL_PATH=resnet_model
if [ -d "${RESNET_MODEL_PATH}" ]; then
  rm -rf "${RESNET_MODEL_PATH}"
fi
mkdir "${RESNET_MODEL_PATH}"

cd official
cd resnet
python3 cifar10_main.py 2>&1 | tee ../../${OUTPUT_DIR}/${LOG_FILE}
cd ../..

# generate pie chart
cd '../../chart_tools/scripts'
./ImportText.pl "../../results/${MODEL}/data_for_pie_chart.tsv" -o "../../results/${MODEL}/pie_chart.html"


CURRENT_DIR=`pwd`
SUMMARY_DIR="${CURRENT_DIR}/../../results_summary/${MODEL}"

if [ ! -d ${SUMMARY_DIR} ]; then
  mkdir ${SUMMARY_DIR}
fi

cp  "${CURRENT_DIR}/../../results/${MODEL}/pie_chart.html" "${CURRENT_DIR}/../../results_summary/${MODEL}/"
cp  "${CURRENT_DIR}/../../results/${MODEL}/data.tsv" "${CURRENT_DIR}/../../results_summary/${MODEL}/"

echo Detailed results saved to "${CURRENT_DIR}/../../results/${MODEL}/"
echo Summary of results saved to "${CURRENT_DIR}/../../results_summary/${MODEL}/"

#chromium-browser "../../results/${MODEL}/pie_chart.html"

