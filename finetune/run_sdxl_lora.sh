huggingface-cli login

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="hahminlew/kream-product-blip-captions"
export ACCELERATE_CONFIG_FILE="/home/ubuntu/diffusers/0fashion/accelerate_config.yaml"

accelerate launch  --config_file $ACCELERATE_CONFIG_FILE  /home/ubuntu/diffusers/0fashion/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sdxl-kream-model-lora-SDXL" \
  --validation_prompt="outer, The Nike x Balenciaga down jacket black, a photography of a black down jacket with a logo on the chest" \
  --report_to="wandb" \
  --resume_from_checkpoint="latest"