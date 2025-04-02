# DSA3101 Group 10: AI-Driven Merchandise Customization Platform for E-commerce

## Set-Up

### **Clone the repository to your computer**  

Run this in your terminal: 

```shell
cd Desktop
git clone https://github.com/tseyongg/DSA3101.git
```

Then head inside:

```shell
cd DSA3101
```
Or you can just open it manually (whichever you prefer).

### Checking for changes:

To see if there are any updates:

```shell
git status
```

If there are changes, pull the latest version:

```
git pull 
```

## Data Dictionary

### 1. Amazon Fashion (5-core)
| Column Name    | Data Type | Description                           
|:--------------|:---------:|--------------------------------------------------------------|
| overall        | Float     | Rating of the product                                       |
| verified       | Boolean   | Date of purchase                                            |
| reviewTime     | Object    | Time of the review (raw)                                    |
| reviewerID     | Object    | ID of the reviewer                                          |
| asin           | Object    | ID of the product                                           |
| style          | Object    | Order status (Pending, Shipped, etc.)                       |
| reviewerName   | Object    | Name of the reviewer                                        |
| reviewText     | Object    | Text of the review                                          |
| summary        | Object    | Summary of the review                                       |
| unixReviewTime | Integer   | Time of the review (unix time)                              |
| vote           | Object    | Helpful votes of the review                                 |
| image          | Object    | Images that users post after they have received the product |

## All Image2Image Models

The following models are all the models we tested to generate our AI customised products.  
All of the models are image-to-image models from Hugging Face. The image of the original product is to be inserted and customisation is done via a written prompt.  
The models include the base model and 4 other models which are base models guided with ControlNet techniques. Among the 5 models, the Canny Edge model performs the best, provding high fidelity customised images in a relatively short time.  
All models are credited to Lvmin Zhang: https://huggingface.co/lllyasviel  

### Base Model without ControlNet (SLOW)
Average runtime: 4 mins  
https://huggingface.co/docs/diffusers/en/using-diffusers/img2img

The base model performs the worst in both time and accuracy when compared to models guided by ControlNet techniques. ControlNets enhance creativity during image generation, allowing the model to produce images that do not strictly adhere to the original.

Two important hyperparameters influence the model’s output:  
Strength (0–1): Controls the model's creativity. Higher values increase creativity and deviation from the original image.  
Guidance Scale (0–10): Determines how closely the model follows the prompt. Higher values result in outputs that better match the prompt.

To maintain high fidelity to the original image, the hyperparameters are set to 1 and 8.0, respectively.

Without ControlNets, achieving close adherence to the original image becomes challenging. Although adjusting the negative prompts hyperparameter can improve fidelity, it is often tedious and requires careful tuning. Example: negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

### Canny Edge Detection (BEST)
Average runtime: 25s  
https://huggingface.co/lllyasviel/sd-controlnet-canny

This model uses Canny Edge Detection as ControlNet to guide the generated image. The model detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. This model runs the fastest among all models.

### Midas Depth Estimation
Average runtime: 1 min  
https://huggingface.co/lllyasviel/sd-controlnet-depth

This model uses Midas Depth Estimation as ControlNet to guide the generated image. The model detects the depth of the original image and generates the image with the depth as guide. The result is a high fidelity customised image. This model runs the slower than the Canny Edge model.

### HED Edge Detection
Average runtime: 1 min  
https://huggingface.co/lllyasviel/sd-controlnet-hed

This model uses HED Edge Detection as ControlNet to guide the generated image. Similar to the Canny Edge model, this model also detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. However, this model runs slower than the Canny Edge model.

### M-LSD Straight Line Detection
https://huggingface.co/lllyasviel/sd-controlnet-mlsd  

The M-LSD Straight Line ControlNet is built to detect straight lines. It performs poorly on images with non-straight edges. As such it is not recommended to use this model.
