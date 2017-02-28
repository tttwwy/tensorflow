#!/usr/bin/env bash
# ps server
CUDA_VISIBLE_DEVICES='' python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --job_name=ps --task_index=0

# worker server:
CUDA_VISIBLE_DEVICES=1 python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=2 python distribute2.py --ps_hosts=localhost:9000 --worker_hosts=localhost:9001,localhost:9002 --job_name=worker --task_index=1