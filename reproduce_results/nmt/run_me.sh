#!/bin/bash

MODEL=nmt

# new experiment will overwrite all files in ../../results/model_name
# make sure to make its copy to keep results for previous hardware setup 

LOG_FILE=${MODEL}'_log.txt'
OUTPUT_DIR="../../results/${MODEL}"
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir -p ${OUTPUT_DIR}
fi

DATA_PATH=../common_datasets/nmt
NMT_MODEL_PATH=${MODEL}/nmt_model
if [ -d "${NMT_MODEL_PATH}" ]; then
  rm -rf "${NMT_MODEL_PATH}"
fi
mkdir "${NMT_MODEL_PATH}"


# to run only on CPU, edit get_config_proto() in misc_util.py ,
# add to config there: device_count = {'GPU': 0}

CMD_TRAIN="python3 -m nmt.nmt \
    --src=vi \
    --tgt=en \
    --attention=scaled_luong \
    --vocab_prefix=${DATA_PATH}/vocab  \
    --train_prefix=${DATA_PATH}/train \
    --dev_prefix=${DATA_PATH}/tst2012  \
    --test_prefix=${DATA_PATH}/tst2013 \
    --out_dir=${NMT_MODEL_PATH} \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu \
    --random_seed=1"
#   --num_intra_threads=1 --num_inter_thread=1"

echo ${CMD_TRAIN} 2>&1 | tee ${OUTPUT_DIR}/${LOG_FILE}
${CMD_TRAIN} 2>&1 | tee ${OUTPUT_DIR}/${LOG_FILE}

# remove temp training data
if [ -d "${NMT_MODEL_PATH}" ]; then
  rm -rf "${NMT_MODEL_PATH}"
fi

# generate pie chart
cd '../../chart_tools/scripts'
./ImportText.pl "../../results/${MODEL}/data_for_pie_chart.tsv" -o "../../results/${MODEL}/pie_chart.html"


CURRENT_DIR=`pwd`
SUMMARY_DIR="${CURRENT_DIR}/../../results_summary/${MODEL}"

if [ ! -d ${SUMMARY_DIR} ]; then
  mkdir -p ${SUMMARY_DIR}
fi

cp  "${CURRENT_DIR}/../../results/${MODEL}/pie_chart.html" "${CURRENT_DIR}/../../results_summary/${MODEL}/"
cp  "${CURRENT_DIR}/../../results/${MODEL}/data.tsv" "${CURRENT_DIR}/../../results_summary/${MODEL}/"

echo
echo "Results saved to " ${CURRENT_DIR}/../../results/${MODEL}/
echo "Summary of results (pie_chart.html, data.tsv) copied to" "${CURRENT_DIR}/../../results_summary/${MODEL}/"



#chromium-browser "../../results/${MODEL}/pie_chart.html"

