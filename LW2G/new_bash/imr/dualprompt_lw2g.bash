#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py imr_dualprompt \
    --epochs 50 \
    --no_auto 0 \
    --threshold 0.99 \
    --threshold_pretrained 0.99 \
    --use_pre_gradient_constraint 1 \
    --threshold2 0.6 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
    --output_dir ./typical_setting/imr_B_0_INC_20/dualprompt/pgc_06





































