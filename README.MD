# DSA3101 Group 10: AI-Driven Merchandise Customization Platform for E-commerce

## Description
We have developed an AI-powered platform which enables real-time product customization, allowing realistic visuals to be generated on demand. By ensuring high-fidelity product designs, it enhances the customer experience, improves inventory efficiency, and boosts the competitiveness of our e-commerce platform.

## Set-Up

### Install the required packages from `requirements.txt`
```shell
pip install -r requirements.txt
```

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
##### 1. `inventory`
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

- base_price: Estimated from product ratings — higher-rated products tend to be priced higher, with some added randomness.

- stock_level: Simulated using a Poisson distribution, suitable for modeling inventory counts.

- reorder_point: Random integers between 18 and 25 to reflect when restocking should be triggered.

- lead_time_days: Follows a log-normal distribution, modeling the fact that most shipments arrive quickly, but a few take much longer.

- storage_cost: Generated with a Gamma distribution, rightskewed to represent varying storage expenses: that most products will have low to moderate storage costs, and only a few products will have very high storage costs.

- material_cost: Also Gamma-distributed but with higher expected values, as production materials generally cost more than storage.

##### 2. `sales`
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