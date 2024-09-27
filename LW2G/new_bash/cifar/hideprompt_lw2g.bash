#!/bin/bash

HIDEP_CKPT_cifar='./ckpt_for_hidep/cifar/cifar_b_0_inc_10/cifar100_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=0 python main.py cifar100_hideprompt_5e \
    --epochs 50 \
    --no_auto 0 \
    --use_pre_gradient_constraint 1 \
    --threshold2 0.5 \
    --use_old_subspace_forward 1 \
    --topk_old_subspace 3 \
	--ca_lr 0.005 \
	--crct_epochs 30 \
    --threshold 0.99 \
    --threshold_pretrained 0.99 \
	--prompt_momentum 0.01 \
	--reg 0.1 \
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model $HIDEP_CKPT_cifar \
    --output_dir ./typical_setting/cifar_B_0_INC_10/hideprompt/all_05








































