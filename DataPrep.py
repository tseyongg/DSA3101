import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# reviews
fp_review = "AMAZON_FASHION_5core_meta.json.gz"
df_5core = getDF_jsonl_gz(fp_review)

# Extract the relevant columns including 'style' and rename them to match the old format
df_review = df_5core[['asin', 'reviewerID', 'overall', 'unixReviewTime', 'reviewText', 'reviewerName', 'image', 'style']].rename(
    columns={
        'reviewerID': 'reviewer_id',
        'overall': 'rating',
        'unixReviewTime': 'unix_timestamp'
    }
)

# Flatten out the 'style' column
# Ensure each entry in 'style' is a dictionary
df_review['style'] = df_review['style'].apply(lambda x: x if isinstance(x, dict) else {})

# Normalize the JSON in 'style' into separate columns
style_df = pd.json_normalize(df_review['style'])

# Clean up the column names if needed (remove extra spaces or colons)
style_df.columns = [col.strip().replace(":", "") for col in style_df.columns]

# Drop the original 'style' column and join the flattened columns back to df_review
df_review = df_review.drop(columns=['style']).join(style_df)

df_review = df_review.drop(columns=['Size Name', 'Style'], errors='ignore')


# ========================================================
#                   DATA CLEANING
# ========================================================


# Remove duplicate products based on `asin`
df_meta = df_meta.drop_duplicates(subset='asin', keep='first')

# Remove duplicates from reviews
df_review_unique = df_review.drop_duplicates(subset=['asin', 'reviewer_id', 'rating', 'unix_timestamp'])

# Drop irrelevant 'date' column and columns with >80% missing values
df_meta = df_meta.drop(columns=['date'])
na_percentage = df_meta.isna().mean() * 100
columns_to_drop = na_percentage[na_percentage > 80].index
df_meta = df_meta.drop(columns=columns_to_drop)


# ========================================================
#                   DATA TRANSFORMATIONS
# ========================================================


# Drop unnecessary columns
catalogue = df_meta.drop(columns=['rank', 'imageURL'])
catalogue = catalogue.dropna(subset=['imageURLHighRes'])

# Create a separate table for image links
links = df_meta["imageURLHighRes"].explode()
catalogue_images = pd.DataFrame(df_meta['asin']).join(links)

# Filter reviews to only include products in the catalogue
base = df_review_unique[df_review_unique['asin'].isin(catalogue['asin'])]
pd.set_option('display.max_colwidth', None)


# ========================================================
#                ADD AVERAGE RATING FEATURE
# ========================================================


avg_rating = base.groupby('asin')['rating'].mean().reset_index().rename(columns={'rating': 'avg_rating'})


# ========================================================
#                   INVENTORY
# ========================================================


# Extract unique products from base table
products = base['asin'].unique()
ratings_by_asin = base.groupby('asin')['rating'].mean()

np.random.seed(70)

inventory_data = {
    'asin': products,
    'base_price': np.round(50 + 20 * (ratings_by_asin.loc[products].values - 3) 
                             + np.random.uniform(-5, 5, size=len(products)), 1),  # Based on ratings and a normal dist.
    'stock_level': np.random.poisson(lam=30, size=len(products)),  # Poisson for stock (counts)
    'reorder_point': np.random.randint(18, 26, size=len(products)),
    'lead_time_days': np.random.lognormal(mean=1.5, sigma=0.6, size=len(products)).astype(int) + 1,  # Lognormal for lead times: right-skewed (many short deliveries, fewer long ones)
    'storage_cost': np.random.gamma(shape=2, scale=1, size=len(products)),  # Gamma for costs (always positive, right-skewed)
    'material_cost': np.random.gamma(shape=5, scale=3, size=len(products))  # Gamma but slightly higher than storage costs, cause production
}

inventory_df = pd.DataFrame(inventory_data)
inventory_df['base_price'] = inventory_df['base_price'].round(1)
inventory_df['storage_cost'] = inventory_df['storage_cost'].round(1)
inventory_df['material_cost'] = inventory_df['material_cost'].round(1)

# Merge average rating into inventory**
inventory_df = inventory_df.merge(avg_rating, on='asin', how='left')


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
    'profit': [],
    'Size': [],
    'Color': []
}

# Pre-calculate customization levels for each product
product_customization = {asin: np.random.randint(0, 6) for asin in products}

# Group reviews by product
reviews_by_asin = {asin: base[base['asin'] == asin] for asin in products}

# Pre-fetch inventory data for quick look-up
inventory_lookup = inventory_df.set_index('asin')

# Process each product for sales simulation
for asin in products:
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

    # Convert timestamps to datetime
    sale_dates = pd.to_datetime(product_reviews['unix_timestamp'].values, unit='s')
    
    # Adjust baseline sales count by average rating, with a normal dist. as foundation
    product_avg_rating = product_reviews['rating'].mean()
    baseline_sales_count = np.maximum(np.round(5 * (product_avg_rating / 3) 
                                 + np.random.normal(loc=0, scale=0.2, size=num_reviews)), 1).astype(int)
    
     # Apply seasonality factor to the baseline count and ensure a minimum of 1
    seasonality_factor = 0.2 * np.abs(np.cos( 2 * np.pi * (sale_dates.month - 12) / 12)) + 1.0
    sales_count_per_review = np.maximum(np.round(baseline_sales_count * seasonality_factor), 1).astype(int)

    total_sales = int(sum(sales_count_per_review))
    
    # Uniform distribution for price variation, accounting for occasional discounts from vouvhers, sales etc.
    price_variations = np.random.uniform(0.99, 1.01, size=total_sales)

    # Left-skewed distribution for quantities (minimum 1, no upper bound)
    np.random.seed(70)
    quantities = np.random.exponential(scale=1.0, size=total_sales)
    quantities = np.maximum(np.round(quantities).astype(int), 1)
    
    # Process each review and generate multiple sales records per review
    sale_index = 0
    for i, review_date in enumerate(sale_dates):

        size = product_reviews.iloc[i].get('Size')
        color = product_reviews.iloc[i].get('Color')

        # Generate variations of the review date for each sale (to the left/earlier only)
        days_before = np.abs(np.random.normal(loc=3, scale=3, size=sales_count_per_review[i]))
        # Set min thus not too far back (cap at 14 days before review)
        days_before = np.minimum(days_before, 14)
        # Gen sale dates that are slightly before the review date
        varied_sale_dates = [review_date - pd.Timedelta(days=float(days)) for days in days_before]

        for j in range(sales_count_per_review[i]):
            # Calculate price with variation
            price = base_price * customization_factor * price_variations[sale_index]
            cost = material_cost * customization_factor
            profit = price - cost
            
            # Add to sales data with varied sale date
            sales_data['asin'].append(asin)
            sales_data['sale_date'].append(varied_sale_dates[j])  # Using varied date instead of review date
            sales_data['quantity'].append(quantities[sale_index])
            sales_data['customization_level'].append(customization)
            sales_data['sale_price'].append(round(price, 1))
            sales_data['cost'].append(round(cost, 1))
            sales_data['profit'].append(round(profit, 1))
            sales_data['Size'].append(size)
            sales_data['Color'].append(color)

            sale_index += 1

sales_df = pd.DataFrame(sales_data)


# ========================================================
#              ADD SEASONAL EFFECTS TO QUANTITY
# ========================================================

def add_seasonal_effects(sales_df):
    sales_df['month'] = sales_df['sale_date'].dt.month
    sales_df['seasonality'] = 0.2 * np.abs(np.cos( 2 * np.pi * (sales_df['month'] - 12) / 12)) + 1.0
    sales_df['quantity'] = (sales_df['quantity'] * sales_df['seasonality']).apply(np.round).astype(int)
    sales_df['quantity'] = sales_df['quantity'].apply(lambda x: max(1, x))
    return sales_df.drop(columns=['seasonality'])

sales_df = add_seasonal_effects(sales_df)


# ========================================================
#              ADD PRICE-QUANTITY RELATIONSHIP
# ========================================================

def add_price_quantity_relationship(sales_df):
    """
    Scale quantity based on sale_price - simple direct relationship
    Higher price = lower quantity
    """
    # Simple price factor - inverse relationship between price and quantity
    # Higher prices mean lower quantities
    sales_df['price_factor'] = 100 / (sales_df['sale_price'] + 10)
    
    # Apply the price factor to quantity 
    sales_df['quantity'] = (sales_df['quantity'] * sales_df['price_factor']).apply(np.round).astype(int)
    
    # Ensure minimum quantity is 1
    sales_df['quantity'] = sales_df['quantity'].apply(lambda x: max(1, x))
    
    # Drop intermediate column
    sales_df = sales_df.drop(columns=['price_factor'])
    
    # Rename original columns for clarity, they are per-unit values
    sales_df = sales_df.rename(columns={
        'sale_price': 'unit_price',
        'cost': 'unit_cost',
        'profit': 'unit_profit'
    })
    
    # Ensure sale_price, cost, and profit now is per order
    # Current columns represent per-unit values, thus we multiply by quantity
    sales_df['sale_price'] = sales_df['unit_price'] * sales_df['quantity']
    sales_df['cost'] = sales_df['unit_cost'] * sales_df['quantity']
    sales_df['profit'] = sales_df['sale_price'] - sales_df['cost']
    
    # Round values
    sales_df['sale_price'] = sales_df['sale_price'].round(1)
    sales_df['cost'] = sales_df['cost'].round(1)
    sales_df['profit'] = sales_df['profit'].round(1)
    
    return sales_df

# Apply price-quantity relationship
sales_df = add_price_quantity_relationship(sales_df)

def plot_graph():

    new_sales_quantity_by_mth = sales_df.groupby('month')['quantity'].sum()
    quant_against_mth = new_sales_quantity_by_mth.plot() # Visualisation of Quantity for each Month
    quant_against_mth.set_ylabel("Quantity")
    quant_against_mth.set_xlabel("Month")
    plt.xticks(np.arange(1,13))
    plt.title("Quantity against Month (After Seasonality Emplification)")
    plt.show()
    sales_df.drop(columns=['month'], inplace=True)

# ========================================================
#                FUNCTION TO RETURN CLEANED DATA
# ========================================================

def cleaned_5core():
    return base

# ========================================================
#                   MAIN EXECUTION
# ========================================================

if __name__ == "__main__":
    print("Base sample:")
    print(base.head(5))
    print(base.columns.tolist())
    print("Inventory sample:")
    print(inventory_df.head(7))
    print("Sales sample:")
    print(sales_df.head(30))
    plot_graph()