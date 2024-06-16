import gradio as gr
import numpy as np
import random
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_id =  "CompVis/stable-diffusion-v1-4"
finetuned_model_id = "JuliPolus/sd-1-4-fashion-model"

if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)

    unet = UNet2DConditionModel.from_pretrained(finetuned_model_id, subfolder="unet_ema", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_id,
                                               unet=unet,
                                               torch_dtype=torch.float16,
                                               use_safetensors=True,
                                               variant="fp16",)
    pipe.to(device)
else:

    unet = UNet2DConditionModel.from_pretrained(finetuned_model_id, subfolder="unet_ema").to(device)
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_id,
                                               unet=unet,
                                               use_safetensors=True,)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 512

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        width = width,
        height = height,
        generator = generator
    ).images[0]

    return image

examples = [
    "bottom, Supreme Animal Print Baggy Jean Washed Indigo, a photography of a dark blue jean with an animal printing on",
    "top, Lacoste Big Elephant Print Sweatshirt Green, a photography of a green sweatshirt with elephant on it",
    "bottom, IAB Studio x Stussy Green palms print Pink Shorts, a photography of Pink pants with a green palms print",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Fashion Text-to-Image Generator
        Currently running on {power_device}.
        """)

        with gr.Row():

            prompt = gr.Text(
                value="bottom, Supreme Сhamomile Flower Print Baggy Jean Light Blue, a photography of a Light blue jean with an Сhamomile flower printing on",
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):

            negative_prompt = gr.Text(
                value="Oversaturated, blurry, low quality",
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=500,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=9.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=45,
                    step=1,
                    value=40,
                )

        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )

    run_button.click(
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result]
    )

demo.queue().launch(share=True)
