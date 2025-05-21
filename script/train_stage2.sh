output_dir=out/stage2_pretraining

ckpt_path=out/stage2_pretraining/best_checkpoint.pth
export MALLOC_CHECK_=1

deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 2 \
   --gradient-accumulation-steps 16 \
   --epochs 2 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset CSL_News \
   --wandb_online \
   --rgb_support
