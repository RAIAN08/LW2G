HIDEP_CKPT_cifar='./ckpt_for_hidep/cifar/cifar_b_0_inc_10/cifar100_hideprompt_5e/stage_one_ckpt'

CUDA_VISIBLE_DEVICES=2 python main.py \
        cifar100_hideprompt_5e \
        --original_model vit_base_patch16_224 \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path ./datasets/ \
        --output_dir $HIDEP_CKPT_cifar \
        --epochs 20 \
        --sched constant \
        --seed 42 \
        --train_inference_task_only \
        --lr 0.0005 