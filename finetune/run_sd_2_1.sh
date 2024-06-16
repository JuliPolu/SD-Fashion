accelerate config default

huggingface-cli login

# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base
export MODEL_NAME="/home/ubuntu/diffusers/0_fashion_sd_2_1"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

accelerate launch  /home/ubuntu/diffusers/0fashion/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=768 --random_flip \
  --train_batch_size=4 \
  --checkpointing_steps=1000 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=28000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant"  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="0_fashion_sd_2_1_2" \
  --report_to="wandb"
