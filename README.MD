# DSA3101 Group 10: AI-Driven Merchandise Customization Platform for E-commerce

## Description
We have developed an AI-powered platform which enables real-time product customization, allowing realistic visuals to be generated on demand. By ensuring high-fidelity product designs, it enhances the customer experience, improves inventory efficiency, and boosts the competitiveness of our e-commerce platform.

## Set-Up

### Clone the repository to your computer  

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

## Obtaining API access key to execute docker run later on

**To obtain the API key, follow these instructions:**

    a. First create a Hugging Face account.  
    b. After creation, click on your profile picture on the top right hand side corner of the page, and go to *"Access Tokens"*.  
    c. Click on *"+ Create new token"*, and navigate to *"Write"* for token type.  
    d. Choose token name of choice and press *"Create token"*.  
    e. Copy the generated token ("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx") onto your clipboard.  


## Building and running Dockerfile

In the same terminal from setup previously:

```shell
docker build -t grp10-streamlit-app . 
```

After the image is built (ignore the warnings), replace hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx (below) with your own generated API Key, then:

```shell
docker run -p 8501:8501 -e HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx grp10-streamlit-app
```


## About Our Data
### Data Source
Our platform uses the **Amazon Fashion Metadata** and its **5-core** data subset from [**Amazon Review Data (2018)**](https://nijianmo.github.io/amazon/index.html), released by Jianmo Ni at UCSD. These datasets provide key product information and user reviews relevant to fashion items. 
The curated data is used to enhance the real-time product customization and visualization experience on our platform, as well as inventory management and pricing optimization.

### Data Dictionary
The data dictionary provides a structured overview of the original dataset and the transformed tables we created, detailing column names, data types, and descriptions.

#### Original Data Tables
##### 1. Amazon Fashion Metadata
This table contains information on product metadata.
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `title`          | Object    | Name of the product                                         |
| `brand`          | Object    | Brand name                                                  |
| `feature`        | Object    | Bullet-point format features of the product                 |
| `rank`           | Object    | Sales rank information                                      |
| `asin`           | Object    | ID of the product                                           |
| `imageURL`       | Object    | URL of product image                                        |
| `imageURLHighRes`| Object    | URL of high resolution product image                        |

##### 2. Amazon Fashion 5-core Metadata
This table is a subset of the data in which all users and items have at least 5 reviews. 
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `overall`        | Float     | Rating of the product                                       |
| `verified`       | Boolean   | Date of purchase                                            |
| `reviewTime`     | Object    | Time of the review                                          |
| `reviewerID`     | Object    | ID of the reviewer                                          |
| `asin`           | Object    | ID of the product                                           |
| `style`          | Object    | Dict containing size, color of product                      |
| `reviewerName`   | Object    | Name of the reviewer                                        |
| `reviewText`     | Object    | Text of the review                                          |
| `summary`        | Object    | Summary of the review                                       |
| `unixReviewTime` | Integer   | Time of the review (unix time)                              |
| `vote`           | Object    | Helpful votes of the review                                 |
| `image`          | Object    | Images that users post after they have received the product |

#### Cleaned Data Tables
##### 1. `df_review`
This table is the cleaned version of the original Amazon Fashion 5-core metadata, displaying the processed reviews data.
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `asin`           | Object    | ID of the product                                           |
| `reviewer_id`    | Object    | ID of the reviewer                                          |
| `rating`         | Float     | Rating of the product                                       |
| `unix_timestamp` | Integer   | Time of the review (unix time)                              |
| `reviewText`     | Object    | Text of the review                                          |
| `reviewerName`   | Object    | Name of the reviewer                                        |
| `image`          | Object    | Images that users post after they have received the product |
| `Size`           | Object    | Size of the product                                         |
| `Color`          | Object    | Color of the product                                        |

##### 2. `catalogue`
This table is the cleaned version of the original Amazon Fashion metadata, displaying the processed product catalogue.
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `title`          | Object    | Name of the product                                         |
| `brand`          | Object    | Brand name                                                  |
| `feature`        | Object    | Bullet-point format features of the product                 |
| `asin`           | Object    | ID of the product                                           |
| `imageURLHighRes`| Object    | URL of high resolution product image                        |

#### Transformed Data Tables
##### 1. `catalogue_images`
This table is a subset of `catalogue`. It contains only `asin` and `imageURLHighRes`.

##### 2. `base`
This table is a subset of `df_review`, which only includes data on products in the `catalogue`.

#### Synthetic Data Tables
##### 1. `inventory_df`
This table contains product inventory data, derived from `base`.
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `asin`           | Object    | ID of the product                                           |
| `base_price`     | Float     | Base price of the product                                   |
| `stock_level`    | Integer   | Stock quantity of the product                               |
| `reorder_point`  | Integer   | Stock level at which the product should be reordered        |
| `lead_time_days` | Integer   | Number of days required to replenish stock of the product   |
| `storage_cost`   | Float     | Cost of storing the product                                 |
| `material_cost`  | Float     | The cost of the raw materials used in the product           |
| `avg_rating`     | Float     | Average rating of the product                               |

This dataset simulates realistic inventory characteristics for a list of products (asin). Each feature is generated using a distribution that mirrors typical business scenarios:

- `base_price`: Estimated from product ratings — higher-rated products tend to be priced higher, with some added randomness.

- `stock_level`: Simulated using a Poisson distribution, suitable for modeling inventory counts.

- `reorder_point`: Random integers between 18 and 25 to reflect when restocking should be triggered.

- `lead_time_days`: Follows a log-normal distribution, modeling the fact that most shipments arrive quickly, but a few take much longer.

- `storage_cost`: Generated with a Gamma distribution, rightskewed to represent varying storage expenses: that most products will have low to moderate storage costs, and only a few products will have very high storage costs.

- `material_cost`: Also Gamma-distributed but with higher expected values, as production materials generally cost more than storage.

##### 2. `sales_df`
This table contains product sales information.
| Column Name          | Data Type | Description                           
|:--------------------|:---------:|--------------------------------------------------------------|
| `asin`               | Object    | ID of the product                                           |
| `sale_date`          | DateTime  | Date when the sale occurred                                 |
| `quantity`           | Integer   | Quantity of product sold                                    |
| `customization_level`| Integer   | Level of customization applied to the product (0-5)         |
| `unit_price`         | Float     | Per unit price of product                                   |
| `unit_cost`          | Float     | Per unit cost of product                                    |
| `unit_profit`        | Float     | Per unit profit of product                                  |
| `Size`               | Object    | Size of the product                                         |
| `Color`              | Object    | Color of the product                                        |
| `month`              | Integer   | Month in which the sale occurred (1-12)                     |
| `sale_price`         | Float     | Final sale price of the product                             |
| `cost`               | Float     | Total cost of the product                                   |
| `profit`             | Float     | Total profit made from the product                          |

Sales data is generated for each product (asin) based on user reviews and inventory details, using a mix of business logic and realistic distributions:

`sale_date`: Generated by slightly offsetting the review dates (e.g. sales likely occurred just before a review).

`quantity`: Simulated using an exponential distribution (right-skewed) — most orders are small, but a few are large. Seasonal trends and pricing effects are applied later.

`customization_level`: A random integer from 0 to 5, representing product personalization (e.g. engraving, packaging).

`unit_price`: Depends on the product's base price and customization level, with slight random variation to mimic discounts or promos.

`unit_cost`: Based on material cost and customization level.

`unit_profit`: Simply `unit_price - unit_cost`.

`Size` & `Color`: Inherited from the original product review, maintaining contextual relevance.

`sale_price` : Simply `unit_price` * `quantity`.

`cost` : Simply `cost_price` * `quantity`.

`profit`: Simply `unit_price - unit_cost`, multipled by `quantity`.

📆 Seasonal Effects
Our quantities are adjusted to reflect seasonality, using a cosine-based function. Here, sales increase in Spring (Apr-May) and Winter (Nov-Dec).

💵 Price–Quantity Relationship
To simulate price elasticity, our quantities are further adjusted after seasonal changes so that:

Higher prices → Lower quantities sold

Lower prices → Higher quantities

This makes our data more realistic and suitable for our downstream tasks like demand forecasting, price optimization, and inventory management.

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

![Image](https://github.com/user-attachments/assets/9db02c1f-56ba-413d-827f-403454d75cf2)

### Canny Edge Detection (BEST)
Average runtime: 25s  
https://huggingface.co/lllyasviel/sd-controlnet-canny

This model uses Canny Edge Detection as ControlNet to guide the generated image. The model detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. This model runs the fastest among all models.

![Image](https://github.com/user-attachments/assets/e835341b-85b1-4cfe-a778-23a295847ea6)

### Midas Depth Estimation
Average runtime: 1 min  
https://huggingface.co/lllyasviel/sd-controlnet-depth

This model uses Midas Depth Estimation as ControlNet to guide the generated image. The model detects the depth of the original image and generates the image with the depth as guide. The result is a high fidelity customised image. This model runs the slower than the Canny Edge model.

![Image](https://github.com/user-attachments/assets/16ad3e1f-b01c-43b8-b273-49c6abeb8f25)

### HED Edge Detection
Average runtime: 1 min  
https://huggingface.co/lllyasviel/sd-controlnet-hed

This model uses HED Edge Detection as ControlNet to guide the generated image. Similar to the Canny Edge model, this model also detects the edges of the original image and generates the image with the edges as guide. The result is a high fidelity customised image. However, this model runs slower than the Canny Edge model.

![Image](https://github.com/user-attachments/assets/4d1de954-3947-43ab-a9e8-a38609f41e04)

### M-LSD Straight Line Detection
https://huggingface.co/lllyasviel/sd-controlnet-mlsd

The M-LSD Straight Line ControlNet is built to detect straight lines. It performs poorly on images with non-straight edges. As such it is not recommended to use this model.

![Image](https://github.com/user-attachments/assets/d1b5260c-a770-4731-9c22-79f297e237f7)

## Webpages A and B

Considering A/B testing, we decided to build 2 webpages utilising Streamlit to aid in the customisation of products for users. Both webpages call onto the Canny Edge Detection model, which was the best model from our testing, to generate AI e-commerce images.

We decided to go with Webpage_B.py for production after multiple tests and feedback from within our group, as we felt that it had a more intuitive and engaging UI. 

## Inventory Management Strategies

Our inventory management strategies are within the file `inventory_management.py`. It consist of using **ABC analysis** based on the sales and inventory data we have created. Furthermore, we applied **Just-in-Case** and **Just-in-Time** management strategies to find an optimal reorder point for the different groups generated by **ABC analysis**. This helps create targeted groups where the company can both focus additional efforts on Group A products and minimise time wasted on Group C products, which can boost sales and profit.
Afterwhich, we also included **Economic Order Quantity** which is based on our demand forecast results in order to predict a good amount of product to order which maximises cost.
