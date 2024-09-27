#!/bin/bash

HIDEP_CKPT_cub='./ckpt_for_hidep/cub/cub_b_0_inc_20/cub_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=1 python main.py cub_hideprompt_5e \
    --epochs 50 \
    --no_auto 1 \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--prompt_momentum 0.01 \
	--reg 0.01 \
	--length 20 \
	--trained_original_model $HIDEP_CKPT_cub \
    --output_dir ./typical_setting/cub_B_0_INC_20/hideprompt/baseline