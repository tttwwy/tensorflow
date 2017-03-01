#!/usr/bin/env bash
data_dir="/home/wangzhe/work/poem/dl4mt/dl4mt.gz/dl4mt/mt_data"
CUDA_VISIBLE_DEVICES=2 python translate.py --from_train_data $data_dir/train/merge.en.txt \
--to_train_data $data_dir/train/merge.zh.txt \
--from_dev_data $data_dir/dev/en.txt \
--to_dev_data $data_dir/dev/zh.txt \
--data_dir ./data \
--train_dir ./train \
--from_vocab_size 30000 \
--to_vocab_size 30000 \
--batch_size 128 \
--num_layers 1 \
--size 650 \
--learning_rate 0.01 \
--steps_per_checkpoint 1000