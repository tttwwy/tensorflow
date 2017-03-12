#!/usr/bin/env bash
data_dir="/home/wangzhe/work/tensorflow/chat/data"
CUDA_VISIBLE_DEVICES=2 python translate.py --from_train_data $data_dir/train.from \
--to_train_data $data_dir/train.to \
--from_dev_data $data_dir/dev.from \
--to_dev_data $data_dir/dev.to \
--data_dir ./vocab \
--train_dir ./train \
--from_vocab_size 60000 \
--to_vocab_size 60000 \
--batch_size 128 \
--num_layers 1 \
--size 512 \
--learning_rate 0.01 \
--steps_per_checkpoint 1000