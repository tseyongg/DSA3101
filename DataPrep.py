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

def parse_gz_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line.strip())

def getDF_jsonl_gz(path):
    data_list = list(parse_gz_jsonl(path))
    return pd.DataFrame(data_list)


# ========================================================
#                   LOAD DATA
# ========================================================


# original metadata
meta_url = 'https://drive.google.com/file/d/1hMd7FoqSBxce7rQxRTcXZ8G4mDXJB6Ry/view?usp=sharing'
download_url = 'https://drive.google.com/uc?export=download&id='+ meta_url.split('/')[-2]
output = 'file.jsonl'   # random name to feed the output argument
gdown.download(download_url, output, quiet=False)
df_meta = pd.read_json(output, lines=True)

#reviews
fp_review = "AMAZON_FASHION_5core_meta.json.gz"
df_5core = getDF_jsonl_gz(fp_review)

# Extract the relevant columns and rename them to match the old format
df_review = df_5core[['asin', 'reviewerID', 'overall', 'unixReviewTime', 'reviewText', 'reviewerName', 'image']].rename(
    columns={
        'reviewerID': 'reviewer_id',
        'overall': 'rating',
        'unixReviewTime': 'unix_timestamp'
    }
)


# ========================================================
#                   DATA CLEANING
# ========================================================


# `asin` refers to the ID of the product and is unique for each product
# Hence, we will remove duplicate products based on `asin`, keeping only the first occurence
df_meta = df_meta.drop_duplicates(subset='asin', keep='first')

# Remove duplicates from df_review
df_review_unique = df_review.drop_duplicates(subset=['asin', 'reviewer_id', 'rating', 'unix_timestamp'])

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


# ========================================================
#                FUNCTION FOR CLEANED DATA
# ========================================================

def cleaned_5core():
    return base


# ========================================================
#                   INVENTORY
# ========================================================

# Extract unique products from base table
products = base['asin'].unique()

# Pre-calculate all ratings at once
ratings_by_asin = base.groupby('asin')['rating'].mean()

np.random.seed(70)
# Generate inventory data with more realistic distributions
inventory_data = {
    'asin': products,
    'base_price': np.random.uniform(10, 200, size=len(products)) * (0.8 + 0.4 * (ratings_by_asin.loc[products].values/5)),
    'stock_level': np.random.poisson(lam=30, size=len(products)),  # Poisson for stock (counts)
    'reorder_point': np.random.randint(5, 30, size=len(products)),
    'lead_time_days': np.random.lognormal(mean=1.5, sigma=0.6, size=len(products)).astype(int) + 1,  # Lognormal for lead times: right-skewed (many short deliveries, fewer long ones)
    'storage_cost': np.random.gamma(shape=2, scale=1, size=len(products)),  # Gamma for costs (always positive, right-skewed)
    'material_cost': np.random.gamma(shape=10, scale=5, size=len(products))  # Gamma with different parameters
}

# Storage cost (shape=2, scale=1): More skewed, most values between 0-5
# Material cost (shape=10, scale=5): More symmetric, centered around 50

inventory_df = pd.DataFrame(inventory_data)
inventory_df['base_price'] = inventory_df['base_price'].round(1)
inventory_df['storage_cost'] = inventory_df['storage_cost'].round(1)
inventory_df['material_cost'] = inventory_df['material_cost'].round(1)


# ========================================================
#                   SALES
# ========================================================

sales_data = {
    'asin': [],
    'sale_date': [],
    'quantity': [],
    'customization_level': [],
    'sale_price': [],
    'cost': [],
    'profit': []
}

# Pre-calculate customization levels for each asin
product_customization = {asin: np.random.randint(0, 6) for asin in products}

# Group reviews by ASIN to avoid iterating through all reviews for each product
reviews_by_asin = {asin: base[base['asin'] == asin] for asin in products}

# Pre-fetch all inventory data to avoid lookups inside the loop
inventory_lookup = inventory_df.set_index('asin')

# Process each product
for asin in products:
    # Get product data once
    base_price = inventory_lookup.loc[asin, 'base_price']
    material_cost = inventory_lookup.loc[asin, 'material_cost']
    customization = product_customization[asin]
    customization_factor = 1 + (customization * 0.1)
    
    # Get all reviews for this product
    product_reviews = reviews_by_asin[asin]
    
    # Batch process all sales for this product
    num_reviews = len(product_reviews)
    if num_reviews == 0:
        continue
        
    # Normally distributed sales counts (minimum 1, no upper bound)
    sales_count_per_review = np.random.normal(loc=5, scale=3, size=num_reviews)
    sales_count_per_review = np.maximum(np.round(sales_count_per_review).astype(int), 1)
    total_sales = sum(sales_count_per_review)
    
    # Uniform distribution for price variation
    price_variations = np.random.uniform(0.9, 1.1, size=total_sales)
    # Left-skewed distribution for quantities (minimum 1, no upper bound)
    quantities = np.random.exponential(scale=3, size=total_sales)
    quantities = np.maximum(np.round(quantities).astype(int), 1)
    
    # Convert all timestamps at once
    sale_dates = pd.to_datetime(product_reviews['unix_timestamp'].values, unit='s')
    
    # Process each review and generate multiple sales records per review
    sale_index = 0
    for i, review in enumerate(sale_dates):
        for _ in range(sales_count_per_review[i]):
            # Calculate price with variation
            price = base_price * customization_factor * price_variations[sale_index]
            cost = material_cost * customization_factor
            profit = price - cost
            
            # Add to sales data
            sales_data['asin'].append(asin)
            sales_data['sale_date'].append(review)
            sales_data['quantity'].append(quantities[sale_index])
            sales_data['customization_level'].append(customization)
            sales_data['sale_price'].append(round(price, 1))
            sales_data['cost'].append(round(cost, 1))
            sales_data['profit'].append(round(profit, 1))
            
            sale_index += 1

sales_df = pd.DataFrame(sales_data)


if __name__ == "__main__":
    # print(catalogue.columns.tolist())
    # print(catalogue_images.columns.tolist())
    print(len(df_review_unique))
    print(len(base))
    print(df_review_unique.head(5))
    print(base.head(5))
    print(inventory_df.head(7))
    print(sales_df.head(7))
