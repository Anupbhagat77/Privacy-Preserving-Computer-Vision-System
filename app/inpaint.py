# app/inpaint.py
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import streamlit as st

MODEL = "stabilityai/stable-diffusion-2-inpainting"

@st.cache_resource(show_spinner=False)
def get_inpaint_pipe():
    """Load the Stable Diffusion inpainting pipeline once and reuse it across Streamlit reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

PROMPT = (
    "photo, realistic replacement for anonymization; "
    "neutral-looking person face with average features and natural skin texture; "
    "realistic vehicle plate look; natural lighting"
)
NEG = "text, watermark, deformed, extra eyes, blurry, extra limbs"

def inpaint(image_pil: Image.Image, mask_pil: Image.Image, steps=30, guidance=7.0):
    pipe = get_inpaint_pipe()
    out = pipe(
        prompt=PROMPT,
        image=image_pil.convert("RGB"),
        mask_image=mask_pil.convert("L"),
        negative_prompt=NEG,
        num_inference_steps=steps,
        guidance_scale=guidance
    )
    return out.images[0]
