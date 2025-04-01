import pandas as pd
import numpy as np
from datetime import datetime
from demand_forecast import load_data, prepare_features, train_demand_model

# Load data and train model once
print("Loading data and training demand forecast model...")
catalogue, sales_df, inventory_df = load_data()
agg_features = prepare_features(sales_df, inventory_df)
model = train_demand_model(agg_features)
latest_data = agg_features.sort_values(['asin', 'year', 'month']).groupby('asin').last().reset_index() # Here, we sort to get the latest asin later: we want the most current state of our product
print("Model ready")

def get_dynamic_price(asin):
    """Get dynamic price using demand forecasting model"""
    if asin not in catalogue['asin'].values:
        return {"error": f"ASIN {asin} not found in catalogue"}
    
    product_title = catalogue.loc[catalogue['asin'] == asin, 'title'].values[0]

    # Sanitize title if it's suspicious or malformed
    if "<script" in product_title.lower() or len(product_title) > 500:
        product_title = "Unknown Product"

    # Get product info
    inventory_data = inventory_df[inventory_df['asin'] == asin]
    if inventory_data.empty:
        return {"error": f"No inventory data for ASIN {asin}"}
    
    inventory_row = inventory_data.iloc[0]
    base_price = inventory_row['base_price']
    material_cost = inventory_row['material_cost']
    
    # Get current date for input
    now = datetime.now()
    current_year, current_month = now.year, now.month
    
    # Get latest data for this ASIN
    asin_data = latest_data[latest_data['asin'] == asin]
    if asin_data.empty:
        return {"error": f"No sales data for ASIN {asin}"}
    
    # Test different price points
    test_prices = [base_price * factor for factor in [0.8, 0.9, 1.0, 1.1, 1.2]]
    best_price, best_profit, predicted_demand = base_price, 0, 0
    
    # Stock info (used to scale demand)
    stock_level = inventory_row['stock_level']
    reorder_point = inventory_row['reorder_point']
    stock_ratio = stock_level / max(reorder_point, 1)

    for price in test_prices:
        # Adjust price based on stock level to influence demand strategically:
        # - If stock is **low** (below the reorder point), we slightly increase the price 
        #   (by multiplying with 1.05) to slow down sales, giving us time to restock.
        # - If stock is excessively high (more than twice the reorder point), 
        #   we slightly reduce the price (by multiplying with 0.95) to boost demand 
        #   and avoid overstocking or holding costs.
        # - If stock is within normal range, we just leave the price unchanged.
        stock_factor = 1.05 if stock_ratio < 1 else 0.95 if stock_ratio > 2 else 1.0
        adjusted_price = price * stock_factor

        # Clone data and update with adjusted test price
        test_data = asin_data.copy()
        test_data['year'] = current_year
        test_data['month'] = current_month
        test_data['sale_price'] = adjusted_price * test_data['quantity']
        test_data['cost'] = material_cost * test_data['quantity']
        test_data['profit'] = test_data['sale_price'] - test_data['cost']
        test_data['profit_margin'] = test_data['profit'] / test_data['sale_price']
        test_data['price_cost_ratio'] = test_data['sale_price'] / test_data['cost']

        # Prepare prediction features
        X_pred = test_data[['asin', 'year', 'month', 'customization_level', 'sale_price',
                            'cost', 'profit', 'profit_margin', 'price_cost_ratio',
                            'prev_quantity', 'prev_price', 'prev_profit', 'avg_rating', 'Color']]
        X_pred = pd.get_dummies(X_pred, columns=['asin', 'Color'], drop_first=True)

        # Match model features (else will throw error because of one-hot encoding, since here we feed only one asin)
        missing_cols = set(model.feature_names_in_) - set(X_pred.columns)
        for col in missing_cols:
            X_pred[col] = 0
        X_pred = X_pred[model.feature_names_in_]

        # Predict demand
        demand = model.predict(X_pred)[0]

        # Cap demand at available stock ( we can't sell more than what is in stock)
        final_demand = min(demand, stock_level)

        # Final profit based on adjusted demand
        profit = (adjusted_price - material_cost) * final_demand

        if profit > best_profit:
            best_profit = profit
            best_price = adjusted_price
            predicted_demand = final_demand

    return {
        "asin": asin,
        "title": product_title,
        "price": float(round(best_price, 2)),
        "demand": int(predicted_demand),
        "profit": float(round(best_profit, 2))
    }

# Main loop
if __name__ == "__main__":
    # Find an ASIN that exists in both catalogue and inventory (on every run, this gives a random ASIN due to set(), just simply run it to try on a new product every time)
    valid_asins = set(catalogue['asin']) & set(inventory_df['asin'])
    if valid_asins:
        example_asin = list(valid_asins)[0]
        print(f"Using sample ASIN: {example_asin}")
        result = get_dynamic_price(example_asin)
        print(f"Result: {result}")
    else:
        print("No valid ASINs found in both catalogue and inventory")