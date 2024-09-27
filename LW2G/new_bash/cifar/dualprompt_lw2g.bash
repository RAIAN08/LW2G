#!/bin/bash


CUDA_VISIBLE_DEVICES=6 python main.py cifar100_dualprompt \
    --epochs 40 \
    --no_auto 0 \
    --use_pre_gradient_constraint 1 \
    --threshold2 0.5 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
    --threshold 0.95 \
    --threshold_pretrained 0.95 \
    --output_dir ./typical_setting/cifar_B_0_INC_10/dualprompt/all_05






































