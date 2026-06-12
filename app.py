from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import torch
import gradio as gr

# Install dependencies
# pip install transformers diffusers torch pillow gradio

# Login to Hugging Face
login()

# Load model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter Image Description"),
    outputs=gr.Image(label="Generated Image"),
    title="Gen AI Text-to-Image App",
    description="Enter a descriptive text to generate an image."
)

interface.launch(share=True, debug=True)
