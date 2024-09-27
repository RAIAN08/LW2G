HIDEP_CKPT_cub='./ckpt_for_hidep/cub/cub_b_0_inc_20/cub_hideprompt_5e/stage_one_ckpt'


CUDA_VISIBLE_DEVICES=0 python main.py \
        cub_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.01 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --train_inference_task_only \
        --output_dir $HIDEP_CKPT_cub 