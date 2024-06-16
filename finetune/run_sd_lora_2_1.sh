accelerate config default

huggingface-cli login

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

accelerate launch /home/ubuntu/diffusers/0fashion/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME --caption_column="text" \
    --resolution=768 --random_flip \
    --train_batch_size=8 \
    --num_train_epochs=10 --checkpointing_steps=1000 \
    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --seed=42 \
    --output_dir="sd-kream-model-lora-2-1" \
    --validation_prompt="outer, The Nike x Balenciaga down jacket black, a photography of a black down jacket with a logo on the chest" \
    --report_to="wandb"
