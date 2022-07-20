#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=2 --node_rank=0 --master_addr="10.113.182.3" --master_port=29500 train.py -opt /home/jtzhou/song/Restormer-main/UDC_restor/Options/UDC_restormer.yml --launcher pytorch > running.log 2>&1 &
python -m test.py -opt /home/doublebin/Restormer-main/Deraining/Options/UDC_test_synthetic_data.yml