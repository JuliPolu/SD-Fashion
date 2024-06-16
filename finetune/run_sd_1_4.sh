accelerate config default

huggingface-cli login

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

accelerate launch  /home/ubuntu/diffusers/0fashion/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --random_flip \
  --train_batch_size=8 \
  --checkpointing_steps=1000 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant"  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="0_fashion_sd_1_4_2"
