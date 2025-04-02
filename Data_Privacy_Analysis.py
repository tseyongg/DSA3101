# Import necessary libraries
import pandas as pd
import re
import requests
from PIL import Image
from PIL.ExifTags import TAGS
import io
import ast

# Load the cleaned dataset
from DataPrep import cleaned_5core
df_5core = cleaned_5core()

###########################################
#                                         #
# Check for potential PII in reviewerName #
#                                         #
###########################################

# Function to check if a name is likely a full name
def is_full_name(name):
    name = str(name).strip()

    # Ensure the name doesn't end with "Consumer" or "Customer"
    if name.split()[-1] in ["Consumer", "Customer"]:
        return False
    
    # Ensure at least two words as it is likely a full name (excluding initials like "B.")
    parts = name.split()
    
    # If only one word, it's not a full name
    if len(parts) < 2:
        return False
    
    # Check if the last word is an initial (e.g., "B.")
    if re.match(r"^[A-Z]\.?$", parts[-1]):  
        return False

    # Allow names with at least two alphabetic words
    return all(re.match(r"^[A-Za-z-]+$", part) for part in parts)

# Drop duplicate names
df_5core_names = df_5core.drop_duplicates(subset='reviewerName').copy()

# Apply PII detection function for reviewer names
df_5core_names['potential_PII_name'] = df_5core_names['reviewerName'].apply(is_full_name)

# Show sample of PII names
pii_names = df_5core_names[df_5core_names['potential_PII_name']]
print("Potential PII in reviewerName:")
print(pii_names[['reviewerName']].head())

###########################################
#                                         #
#  Check for potential PII in reviewText  #
#                                         #
###########################################

# Function to detect emails
def contains_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return bool(re.search(email_pattern, str(text)))

# Function to detect phone numbers (US format example, modify for other regions)
def contains_phone(text):
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return bool(re.search(phone_pattern, str(text)))

# Apply PII detection functions for reviewer text
df_5core['contains_email'] = df_5core['reviewText'].apply(contains_email)
df_5core['contains_phone'] = df_5core['reviewText'].apply(contains_phone)

# Show PII-containing rows
pii_reviews = df_5core[(df_5core['contains_email']) | (df_5core['contains_phone'])]
if pii_reviews.empty:
    print("No potential PII found in reviewText.")
else:
    print("Potential PII in reviewText:")
    print(pii_reviews[['reviewText']].head())

###########################################
#                                         #
#    Check for potential PII in images    #
#                                         #
###########################################

# Function to extract EXIF data from an image in memory
def extract_exif_from_url(image_url):
    try:
        # Ensure image_url is a clean string
        if isinstance(image_url, list):
            if len(image_url) > 0:
                image_url = image_url[0]
            else:
                return None

        # Remove any extra brackets or quotes
        image_url = image_url.strip("[]").strip("'\"")
        
        response = requests.get(image_url)  # Ensure it's a clean URL
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            exif_data = image._getexif() or {}  # Prevent NoneType errors
            return {TAGS.get(tag, tag): value for tag, value in exif_data.items()} if exif_data else None
        else:
            print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

def process_image_urls(df_5core):
    exif_results = []

    for image_url in df_5core['image'].dropna():
        # Handle cases where image_url might be a string representation of a list
        try:
            # Try to safely convert string to list if it looks like a list representation
            if isinstance(image_url, str) and image_url.startswith('[') and image_url.endswith(']'):
                image_url = ast.literal_eval(image_url)
        except (ValueError, SyntaxError):
            pass  # If conversion fails, keep as is

        # Process single URL or first URL from list
        exif_data = extract_exif_from_url(image_url)
        if exif_data:
            exif_results.append((image_url, exif_data))

    return exif_results

# Detection function for potential PII in images
def main(df_5core):
    exif_results = process_image_urls(df_5core)

    if exif_results:
        print("EXIF Metadata from the analysed images:")
        for url, exif in exif_results:
            print(f"URL: {url}")
            print(f"EXIF: {exif}\n")
    else:
        print("No EXIF metadata found in the images.")

# Apply function on 5core data
main(df_5core)
