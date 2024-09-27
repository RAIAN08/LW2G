HIDEP_CKPT_imr='./ckpt_for_hidep/imr/imr_b_0_inc_20/imr_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=1 python main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --train_inference_task_only \
        --output_dir $HIDEP_CKPT_imr