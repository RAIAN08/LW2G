#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py imr_dualprompt \
    --epochs 50 \
    --no_auto 1 \
    --output_dir ./typical_setting/imr_B_0_INC_20/dualprompt/baseline