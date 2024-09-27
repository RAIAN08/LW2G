#!/bin/bash

HIDEP_CKPT_cifar='./ckpt_for_hidep/cifar/cifar_b_0_inc_10/cifar100_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=3 python main.py cifar100_hideprompt_5e \
    --epochs 50 \
    --no_auto 1 \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--prompt_momentum 0.01 \
	--reg 0.1 \
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model $HIDEP_CKPT_cifar \
    --output_dir ./typical_setting/cifar_B_0_INC_10/hideprompt/baseline