output_dir=out/stage2_pretraining

ckpt_path=out/stage1_pretraining/csl_stage1_weight.pth

deepspeed --include localhost:0,1 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 8 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset WLASL \

#--rgb_support
