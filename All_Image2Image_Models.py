# -*- coding: utf-8 -*-
"""
# All Image2Image Models

The following models are all the models we tested to generate our AI customised products.  

All of the models are image-to-image models from Hugging Face. The image of the original product is to be inserted and customisation is done via a written prompt.  

The models include the base model and 4 other models which are base models guided with ControlNet techniques. Among the 5 models, the Canny Edge model performs the best, provding high fidelity customised images in a relatively short time.  

All models are credited to Lvmin Zhang: https://huggingface.co/lllyasviel

## Base Model without ControlNet (SLOW)

### Average runtime: 4 mins

https://huggingface.co/docs/diffusers/en/using-diffusers/img2img

The base model performs the worst in both time and accuracy when compared to models guided by ControlNet techniques. ControlNets enhance creativity during image generation, allowing the model to produce images that do not strictly adhere to the original.

Two important hyperparameters influence the model’s output:

Strength (0–1): Controls the model's creativity. Higher values increase creativity and deviation from the original image.

Guidance Scale (0–10): Determines how closely the model follows the prompt. Higher values result in outputs that better match the prompt.

To maintain high fidelity to the original image, the hyperparameters are set to 1 and 8.0, respectively.

Without ControlNets, achieving close adherence to the original image becomes challenging. Although adjusting the negative prompts hyperparameter can improve fidelity, it is often tedious and requires careful tuning.
Example: negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
"""

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

# This loads the pre-trained model, the model is "stabilityai/stable-diffusion-xl-refiner-1.0"
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()

# insert your image to be customised here
url = "https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg"
init_image = load_image(url)

# Enter your prompt here
prompt = "neck pillow, jungle pattern"

# pass prompt and image to pipeline
# stength (0-1): Adjust the creativity of the model, higher the value, the more creative the model is
# guidance_scale (0-10): Dictates how closely should the model follow the prompt, higher the value, closer to the prompt
image = pipeline(prompt, image=init_image, strength=1, guidance_scale=8.0).images[0]
image.show()
make_image_grid([init_image, image], rows=1, cols=2)

"""## Canny Edge Detection (BEST)

### Average runtime: 25s

https://huggingface.co/lllyasviel/sd-controlnet-canny

This model uses Canny Edge Detection as ControlNet to guide the generated image. The model detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. This model runs the fastest among all models.
"""

import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from diffusers.utils import make_image_grid, load_image

init_image = load_image("https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg")
image = np.array(init_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# Enter your prompt here
prompt = "neck pillow, jungle pattern"

image = pipe(prompt, canny_image, num_inference_steps=20, strength=1, guidance_scale=8.0).images[0]

make_image_grid([init_image, canny_image, image], rows=1, cols=3)

"""## Midas Depth Estimation

### Average runtime: 1 min

https://huggingface.co/lllyasviel/sd-controlnet-depth

This model uses Midas Depth Estimation as ControlNet to guide the generated image. The model detects the depth of the original image and generates the image with the depth as guide. The result is a high fidelity customised image. This model runs the slower than the Canny Edge model.
"""

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
from diffusers.utils import make_image_grid, load_image

depth_estimator = pipeline('depth-estimation')

init_image = load_image("https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg")

image = depth_estimator(init_image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
depth_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# Enter your prompt here
prompt = "neck pillow, jungle pattern"

image = pipe(prompt, depth_image, num_inference_steps=20, strength=1, guidance_scale=8.0).images[0]

make_image_grid([init_image, depth_image, image], rows=1, cols=3)

"""## HED Edge Detection

### Average runtime: 1 min

https://huggingface.co/lllyasviel/sd-controlnet-hed

This model uses HED Edge Detection as ControlNet to guide the generated image. Similar to the Canny Edge model, this model also detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. However, this model runs slower than the Canny Edge model.
"""

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from diffusers.utils import make_image_grid, load_image

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

init_image = load_image("https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg")


hed_image = hed(init_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# Enter your prompt here
prompt = "neck pillow, jungle pattern"

image = pipe(prompt, hed_image, num_inference_steps=20, strength=1, guidance_scale=8.0).images[0]

make_image_grid([init_image, hed_image, image], rows=1, cols=3)

"""## M-LSD Straight Line Detection

https://huggingface.co/lllyasviel/sd-controlnet-mlsd

The M-LSD Straight Line ControlNet is built to detect straight lines. It performs poorly on images with non-straight edges. As such it is not recommended to use this model.
"""

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

init_image = load_image("https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg")

mlsd_image = mlsd(init_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# Enter your prompt here
prompt = "neck pillow, jungle pattern"

image = pipe(prompt, mlsd_image, num_inference_steps=20, strength=1, guidance_scale=8.0).images[0]

make_image_grid([init_image, mlsd_image, image], rows=1, cols=3)

