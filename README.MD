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
The data dictionary provides a structured overview of the cleaned dataset, detailing column names, data types, and descriptions.

### 1. Amazon Fashion (5-core)
The data here is a subset of the data in which all users and items have at least 5 reviews. We use the data here as a reviews table as it contains information on user reviews.
| Column Name    | Data Type | Description                           
|:--------------|:---------:|--------------------------------------------------------------|
| overall        | Float     | Rating of the product                                       |
| verified       | Boolean   | Date of purchase                                            |
| reviewTime     | Datetime  | Time of the review                                          |
| reviewerID     | Object    | ID of the reviewer                                          |
| asin           | Object    | ID of the product                                           |
| style          | Object    | Order status (Pending, Shipped, etc.)                       |
| reviewerName   | Object    | Name of the reviewer                                        |
| reviewText     | Object    | Text of the review                                          |
| summary        | Object    | Summary of the review                                       |
| image          | Object    | Images that users post after they have received the product |

### 2. Amazon Fashion (metadata)
The data here contains product metadata, which includes the following:
| Column Name    | Data Type | Description                           
|:--------------|:---------:|--------------------------------------------------------------|
| title          | Object    | Name of the product                                         |
| brand          | Object    | Brand name                                                  |
| feature        | Object    | Bullet-point format features of the product                 |
| rank           | Object    | Sales rank information                                      |
| asin           | Object    | ID of the product                                           |
| imageURL       | Object    | URL of product image                                        |
| imageURLHighRes| Object    | URL of high resolution product image                        |

