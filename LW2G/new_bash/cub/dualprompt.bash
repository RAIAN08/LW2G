#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py cub_dualprompt \
    --epochs 50 \
    --no_auto 1 \
    --output_dir ./typical_setting/cub_B_0_INC_20/dualprompt/baseline