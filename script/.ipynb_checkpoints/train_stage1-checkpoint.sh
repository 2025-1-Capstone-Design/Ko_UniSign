output_dir=out/stage1_pretraining

deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 16 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset CSL_News \
   --enable_tensorboard \
   --warmup-epochs 3
