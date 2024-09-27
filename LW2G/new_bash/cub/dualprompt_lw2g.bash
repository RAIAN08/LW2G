#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py cub_dualprompt \
    --epochs 50 \
    --no_auto 0 \
    --threshold 0.90 \
    --threshold_pretrained 0.90 \
    --use_pre_gradient_constraint 1 \
    --threshold2 0.3 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
    --output_dir ./typical_setting/cub_B_0_INC_20/dualprompt/pgc_03







































