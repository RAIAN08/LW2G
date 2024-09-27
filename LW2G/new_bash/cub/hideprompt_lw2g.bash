#!/bin/bash

HIDEP_CKPT_cub='./ckpt_for_hidep/cub/cub_b_0_inc_20/cub_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=1 python main.py cub_hideprompt_5e \
    --epochs 50 \
    --no_auto 0 \
    --threshold 0.50 \
    --threshold_pretrained 0.50 \
    --use_pre_gradient_constraint 1 \
    --threshold2 1.0 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--prompt_momentum 0.01 \
	--reg 0.01 \
	--length 20 \
	--trained_original_model $HIDEP_CKPT_cub \
    --output_dir ./typical_setting/cub_B_0_INC_20/hideprompt/pgc_07







































