import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
from data_preparation import catalogue, sales_df, inventory_df

# Load data
def load_data():
    return catalogue, sales_df, inventory_df

# Feature engineering for time series with additional features
def prepare_features(sales_df, inventory_df):
    """Prepare features for XGBoost time series model with enriched data"""
    
    # Convert to datetime and extract components
    sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
    
    # Create time-based features
    features = sales_df.copy()
    features['year'] = features['sale_date'].dt.year
    features['month'] = features['sale_date'].dt.month
    features['day'] = features['sale_date'].dt.day
    features['dayofweek'] = features['sale_date'].dt.dayofweek
    features['quarter'] = features['sale_date'].dt.quarter
    
    # Join inventory data for additional features
    features = features.merge(inventory_df, on='asin', how='left')
    
    # Create profit margin feature
    features['profit_margin'] = features['profit'] / features['sale_price']
    
    # Create inventory turnover ratio
    features['inventory_turnover'] = features['quantity'] / features['stock_level']
    
    # Create price to cost ratio
    features['price_cost_ratio'] = features['sale_price'] / features['cost']
    

    agg_features = features.groupby(['asin', 'year', 'month']).agg({
        'quantity': 'sum',
        'customization_level': 'mean',
        'sale_price': 'sum',
        'cost': 'sum',
        'profit': 'sum',
        'profit_margin': 'mean',
        'inventory_turnover': 'mean',
        'price_cost_ratio': 'mean',
        'stock_level': 'mean',
        'reorder_point': 'mean',
        'lead_time_days': 'mean',
        'storage_cost': 'sum',
        'material_cost': 'sum',
        'avg_rating': 'mean',
        # 'Size': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'Color': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()

    # After aggregation, recalculate the ratio metrics using the summed values
    agg_features['profit_margin'] = agg_features['profit'] / agg_features['sale_price']
    agg_features['price_cost_ratio'] = agg_features['sale_price'] / agg_features['cost']
    # -----------------------------------------------------------
    
    # Sort by asin, year, month to enable proper lagging
    agg_features = agg_features.sort_values(['asin', 'year', 'month'])
    
    # Create lag features
    agg_features['prev_quantity'] = agg_features.groupby('asin')['quantity'].shift(1)
    agg_features['prev_price'] = agg_features.groupby('asin')['sale_price'].shift(1)
    agg_features['prev_profit'] = agg_features.groupby('asin')['profit'].shift(1)
    

    # Fill NaN values with 0
    agg_features = agg_features.fillna(0)
    
    return agg_features

# Custom evaluation metric: acceptable accuracy within 100-165% of actual
def acceptable_accuracy_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    lower_bound = y_true  # 100% of actual
    upper_bound = y_true * 1.65  # 165% of actual
    accuracy = np.mean((y_pred >= lower_bound) & (y_pred <= upper_bound))
    return 'acceptable_accuracy', accuracy

def train_demand_model(agg_features):
    """Train XGBoost model for demand forecasting with enhanced features"""

    X = agg_features[['asin', 'year',
                      'month', 'customization_level', 'sale_price',
                      'cost', 'profit', 
                      'profit_margin', 
                      'price_cost_ratio', 
                      'prev_quantity', 'prev_price', 'prev_profit',
                      'avg_rating', 'Color'
                      ]]
    # -----------------------------------------------------------
    
    # One-hot encode asin
    X = pd.get_dummies(X, columns=['asin', 'Color'], drop_first=True)
    y = agg_features['quantity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    # Define sample weights
    sample_weights = 1.1 + (y_train / y_train.mean())
    
    # Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=41,
        early_stopping_rounds=20,  # Add early stopping
        eval_metric='mae' 
    )

    # Modify fit to include validation data
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)

    bias_factor = 1.05  # Increase predictions by 5%, we want our model to overestimate a little
    y_pred = y_pred * bias_factor

    mae = mean_absolute_error(y_test, y_pred)
    print(f"Standard MAE: {mae:.2f}")
    
    # Check if prediction is between actual and 165% of actual
    lower_bound = y_test  # 100% of actual
    upper_bound = y_test * 1.65  # 165% of actual
    acceptable_pred = ((y_pred >= lower_bound) & (y_pred <= upper_bound))
    acceptable_accuracy = acceptable_pred.mean() * 100
    print(f"Predictions within 100-165% of actual: {acceptable_accuracy:.2f}%")
        
    return model, X


if __name__ == "__main__":
    catalogue, sales_df, inventory_df = load_data()
    agg_features = prepare_features(sales_df, inventory_df)
    print(agg_features)
    model, X = train_demand_model(agg_features)
