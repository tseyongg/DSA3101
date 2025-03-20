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
