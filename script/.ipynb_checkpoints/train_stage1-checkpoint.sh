output_dir=out/stage1_pretraining
ckpt_path=out/stage1_pretraining/checkpoint_0.pth
export MALLOC_CHECK_=0

   # --wandb_online \

deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 16 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset CSL_News \
   --wandb_online
