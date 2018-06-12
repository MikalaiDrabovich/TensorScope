#!/bin/bash
#sudo rm -rf /home/ndr/work/tf/data/seq2seq/nmt_model
sudo python3 -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/home/ndr/work/tf/data/seq2seq/nmt_data/vocab  \
    --train_prefix=/home/ndr/work/tf/data/seq2seq/nmt_data/train \
    --dev_prefix=/home/ndr/work/tf/data/seq2seq/nmt_data/tst2012  \
    --test_prefix=/home/ndr/work/tf/data/seq2seq/nmt_data/tst2013 \
    --out_dir=/home/ndr/work/tf/data/seq2seq/nmt_model \
    --num_train_steps=24000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu \
    --random_seed=1
#    --num_intra_threads=1 \
#    --num_inter_thread=1

# to disable gpu, edit get_config_proto() in misc_util.py , add to config there
#      device_count = {'GPU': 0}

