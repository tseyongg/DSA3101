# Get required packages
import pandas as pd
import numpy as np
import json
import gzip
from IPython.display import display
from PIL import Image
import requests
import gdown
from io import BytesIO

# ========================================================
#                   LOAD DATA
# ========================================================


# original metadata
# meta_url = 'https://drive.google.com/file/d/1hMd7FoqSBxce7rQxRTcXZ8G4mDXJB6Ry/view?usp=sharing'
# download_url = 'https://drive.google.com/uc?export=download&id='+ meta_url.split('/')[-2]
output = 'file.jsonl'   # random name to feed the output argument
# gdown.download(download_url, output, quiet=False)
df_meta = pd.read_json(output, lines=True)

#reviews
fp_review = r"reviews_AMAZON_FASHION.csv.gz"
df_review = pd.read_csv(fp_review, header=None, names=['asin', 'reviewer_id', 'rating', 'unix_timestamp'])



# ========================================================
#                   DATA CLEANING
# ========================================================


# `asin` refers to the ID of the product and is unique for each product
# Hence, we will remove duplicate products based on `asin`, keeping only the first occurence
df_meta = df_meta.drop_duplicates(subset='asin', keep='first')


# Optional: Remove duplicates
df_review_unique = df_review.drop_duplicates()


# Drop irrelevant 'date' column (contains no date information)
df_meta = df_meta.drop(columns=['date'])
# Calculate the percentage of NaN values in each column
na_percentage = df_meta.isna().mean() * 100
columns_to_drop = na_percentage[na_percentage > 80].index
# Drop the columns with more than 80% NaN values as they are likely to be useless in analysis
df_meta = df_meta.drop(columns=columns_to_drop)


# ========================================================
#                   DATA TRANSFORMATIONS
# ========================================================

# Drop unnecessary columns (not needed for analysis)
catalogue = df_meta.drop(columns=['rank', 'imageURL'])
# Drop rows where 'imageURLHighRes' is null
catalogue = catalogue.dropna(subset=['imageURLHighRes'])

# Create a separate table for image links, with original 'images' column pivoted out into different image types for each row of data
links = df_meta["imageURLHighRes"].explode()
catalogue_images = pd.DataFrame(df_meta['asin']).join(links)


base = df_review_unique[df_review_unique['asin'].isin(catalogue['asin'])]
pd.set_option('display.max_colwidth', None)
# print(catalogue.columns.tolist())
# print(catalogue_images.columns.tolist())
# print(base.columns.tolist())

# Extract unique products from base table
products = base['asin'].unique()

# Pre-calculate all ratings at once
ratings_by_asin = base.groupby('asin')['rating'].mean()

np.random.seed(70)

# Generate inventory data
# 0.8 is minimum price multiplier (80% of base price), 0.4 is ratings influence range (can add up to 40% more)
inventory_data = {
    'asin': products,
    'base_price': np.random.uniform(10, 200, size=len(products)) * (0.8 + 0.4 * (ratings_by_asin.loc[products].values/5)),
    'stock_level': np.random.randint(0, 100, size=len(products)),
    'reorder_point': np.random.randint(5, 30, size=len(products)),
    'lead_time_days': np.random.randint(1, 14, size=len(products)),
    'storage_cost': np.random.uniform(0.5, 5, size=len(products)),
    'material_cost': np.random.uniform(5, 100, size=len(products))
}

inventory_df = pd.DataFrame(inventory_data)
print(inventory_df.head(10))