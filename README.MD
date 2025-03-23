# DSA3101 Group 10: AI-Driven Merchandise Customization Platform for E-commerce

## Description
We have developed an AI-powered platform which enables real-time product customization, allowing realistic visuals to be generated on demand. By ensuring high-fidelity product designs, it enhances the customer experience, improves inventory efficiency, and boosts the competitiveness of our e-commerce platform.

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

### Download the dataset
The **Amazon Fashion Metadata** is hosted on Google Drive due to its large size. 
You may download the dataset from this [link](https://drive.google.com/file/d/1hMd7FoqSBxce7rQxRTcXZ8G4mDXJB6Ry/view?usp=sharing).

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
The data dictionary provides a structured overview of the cleaned dataset, detailing column names, data types, and descriptions.

#### 1. Amazon Fashion (5-core)
The data here is a subset of the data in which all users and items have at least 5 reviews. We use the data here as a reviews table as it contains information on user reviews.
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `overall`        | Float     | Rating of the product                                       |
| `verified`       | Boolean   | Date of purchase                                            |
| `reviewTime`     | Datetime  | Time of the review                                          |
| `reviewerID`     | Object    | ID of the reviewer                                          |
| `asin`           | Object    | ID of the product                                           |
| `style`          | Object    | Order status (Pending, Shipped, etc.)                       |
| `reviewerName`   | Object    | Name of the reviewer                                        |
| `reviewText`     | Object    | Text of the review                                          |
| `summary`        | Object    | Summary of the review                                       |
| `image`          | Object    | Images that users post after they have received the product |

#### 2. Amazon Fashion (Metadata)
The data here contains product metadata, which includes the following:
| Column Name      | Data Type | Description                           
|:----------------|:---------:|--------------------------------------------------------------|
| `title`          | Object    | Name of the product                                         |
| `brand`          | Object    | Brand name                                                  |
| `feature`        | Object    | Bullet-point format features of the product                 |
| `rank`           | Object    | Sales rank information                                      |
| `asin`           | Object    | ID of the product                                           |
| `imageURL`       | Object    | URL of product image                                        |
| `imageURLHighRes`| Object    | URL of high resolution product image                        |

