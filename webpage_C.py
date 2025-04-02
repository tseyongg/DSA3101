import streamlit as st
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image, make_image_grid

st.title("AI-Driven Merchandise Customisation")

@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

pipe = load_pipeline()

def process_image(url, prompt):
    init_image = load_image(url)
    image = np.array(init_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    result = pipe(prompt, canny_image, num_inference_steps=20, strength=1, guidance_scale=8.0).images[0]
    return init_image, canny_image, result

# Initialize session state for feedback
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
    st.session_state.satisfaction = 5
    st.session_state.comments = ""

# Input for image URL
image_url = st.text_input("Enter image URL:", "https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg")

# Display the original image immediately below the URL input
if image_url:
    try:
        original_image = load_image(image_url)
        st.image(original_image, caption="Original Image", width=300)
    except Exception as e:
        st.error("Error loading image. Please check the URL.")

# Input for prompt
prompt = st.text_input("Enter prompt:", "neck pillow, jungle pattern")

# Generate button triggers processing
if st.button("Generate"):
    with st.spinner("Generating..."):
        init_image, canny_image, generated_image = process_image(image_url, prompt)
        st.image(generated_image, caption="Generated Image", width=300)
        st.session_state.feedback_submitted = False  # Reset feedback on new generation

# Feedback Section
st.write("### How satisfied were you?")
st.session_state.satisfaction = st.slider("Satisfaction", 1, 10, st.session_state.satisfaction)
st.session_state.comments = st.text_area("Comments", st.session_state.comments)

if st.button("Submit Feedback"):
    st.session_state.feedback_submitted = True

# Show feedback confirmation without resetting the page
if st.session_state.feedback_submitted:
    st.success("Thank you for your feedback!")
    st.write(f"**Satisfaction:** {st.session_state.satisfaction}")
    st.write(f"**Comments:** {st.session_state.comments}")
