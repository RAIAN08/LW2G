#!/bin/bash

HIDEP_CKPT_imr='./ckpt_for_hidep/imr/imr_b_0_inc_20/imr_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=2 python main.py imr_hideprompt_5e \
    --epochs 150 \
    --no_auto 0 \
    --threshold 0.90 \
    --threshold_pretrained 0.90 \
    --use_pre_gradient_constraint 1 \
    --threshold2 0.2 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
    --ca_lr 0.005 \
    --crct_epochs 30 \
	--sched cosine \
	--prompt_momentum 0.01 \
	--reg 0.5 \
	--length 20 \
    --larger_prompt_lr \
	--trained_original_model $HIDEP_CKPT_imr \
    --output_dir ./typical_setting/imr_B_0_INC_20/hideprompt/pgc_02







































