#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py cifar100_dualprompt \
    --epochs 40 \
    --no_auto 1 \
    --output_dir ./typical_setting/cifar_B_0_INC_10/dualprompt/baseline