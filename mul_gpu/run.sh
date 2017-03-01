#!/usr/bin/env bash
ps -ef | grep "python distribute2.py" | awk '{print $2}' | xargs kill -9
rm -rf checkpoint
sleep 2
issync="0"
# ps server
CUDA_VISIBLE_DEVICES='' python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --issync=$issync --job_name=ps --task_index=0 &>ps.log &

# worker server:
CUDA_VISIBLE_DEVICES=2 python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --issync=$issync --job_name=worker --task_index=1 &>worker1.log &

CUDA_VISIBLE_DEVICES=1 python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --issync=$issync --job_name=worker --task_index=0 &>worker0.log

