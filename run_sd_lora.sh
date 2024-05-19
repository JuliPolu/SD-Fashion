accelerate config default

huggingface-cli login

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

accelerate launch /home/ubuntu/diffusers/0fashion/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME --caption_column="text" \
    --resolution=512 --random_flip \
    --train_batch_size=16 \
    --num_train_epochs=100 --checkpointing_steps=1000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --seed=42 \
    --output_dir="sd-kream-model-lora" \
    --validation_prompt="outer, The Nike x Balenciaga down jacket black, a photography of a black down jacket with a logo on the chest" \
    --report_to="wandb"
