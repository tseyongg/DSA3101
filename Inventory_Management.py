import math
from DataPrep import products, sales_df, inventory_df
from demand_forecast import prepare_features, train_demand_model

# ========================================================
#             Creation of ABC Analysis Dataframe
# ========================================================

quant_by_product = sales_df.groupby('asin')['quantity'].sum().reset_index()

for i in products:
    quant = quant_by_product.loc[quant_by_product['asin'] == i , 'quantity'].values[0]
    storage = inventory_df.loc[inventory_df['asin'] == i , 'storage_cost'].values[0]
    mat = inventory_df.loc[inventory_df['asin'] == i , 'material_cost'].values[0]
    con_val = quant * storage * mat
    quant_by_product.loc[quant_by_product['asin'] == i , 'consumption_value'] = round(con_val,1)

quant_by_product = quant_by_product.sort_values(by=['consumption_value'], ascending = False).reset_index()
total_con_val = round(quant_by_product['consumption_value'].sum(),1)
quant_by_product["percent_consumption_value"] = ((quant_by_product["consumption_value"].values / total_con_val) * 100).round(2)

top_product = quant_by_product['asin'].iloc[0]
quant_by_product.loc[quant_by_product['asin'] == top_product , 'ABC_group'] = 'A'
running_percent_consum = quant_by_product.loc[quant_by_product['asin'] == top_product , 'percent_consumption_value'].values[0]

for j in range(1,products.size) :
    curr_product = quant_by_product['asin'].iloc[j]
    if running_percent_consum <= 80:
        quant_by_product.loc[quant_by_product['asin'] == curr_product , 'ABC_group'] = 'A'
        running_percent_consum += quant_by_product.loc[quant_by_product['asin'] == curr_product , 'percent_consumption_value'].values[0]
    elif 80 <= running_percent_consum <= 98:
        quant_by_product.loc[quant_by_product['asin'] == curr_product , 'ABC_group'] = 'B'
        running_percent_consum += quant_by_product.loc[quant_by_product['asin'] == curr_product , 'percent_consumption_value'].values[0]
    else:
        latest = j
        break

quant_by_product.loc[latest:,'ABC_group'] = 'C'

ABC_analysis_df = quant_by_product

# Since the top 5 items account for 80% of the company's consumption value (Group A), 
# the company can focus on improving lead times and optimise costs for these items.

# ==============================================================================
#       Applying JIT/JIC to reorder points based on ABC analysis results
# ==============================================================================

new_inventory_df = inventory_df.copy()

for prod in products:
    curr_product = prod
    group = ABC_analysis_df.loc[ABC_analysis_df['asin'] == curr_product,'ABC_group'].values[0]
    original_reorder = new_inventory_df.loc[new_inventory_df['asin'] == curr_product,'reorder_point'].values[0]
    if group == 'A':
        new_inventory_df.loc[new_inventory_df['asin'] == curr_product,'reorder_point'] = round(original_reorder * 1.5)
    elif group == 'B':
        continue
    else:
        new_inventory_df.loc[new_inventory_df['asin'] == curr_product,'reorder_point'] = round(max(original_reorder * 0.75,1))



# We increase the reorder_point for group A items to fit the Just-in-Case framework since they have high sales quantity
# On the opposite hand, we lower the reorder_point for group C items since their sales quantity is low. Just-in-Time framework.

# ==============================================================================
#          Economic Order Quantity Inventory Management Strategy
# ==============================================================================

# Load data and train model once
print("Loading data and training demand forecast model...")
agg_features = prepare_features(sales_df, inventory_df)
model, X = train_demand_model(agg_features)
print("Model ready")

temp = X.copy()

asin_cols = temp.columns[12:42]
temp.columns[12:42][1][5:]

num_asin = asin_cols.size
n = temp.shape[0]
temp['asin'] = 'n' # place holder value

for j in range(num_asin):
    curr_col = asin_cols[j]
    curr = asin_cols[j][5:]
    temp.loc[temp[curr_col] == 1 , 'asin'] = curr

temp.loc[temp['asin'] == 'n', 'asin'] = 'B000K2PJ4K' 
temp['predicted_demand'] = model.predict(X)
pred_demand_by_asin = temp.groupby('asin')['predicted_demand'].sum().reset_index() # Get predicted demand based on asin

for i in products:
    demand = pred_demand_by_asin.loc[pred_demand_by_asin['asin'] == i , 'predicted_demand'].values[0]
    storage = inventory_df.loc[inventory_df['asin'] == i , 'storage_cost'].values[0]
    mat = inventory_df.loc[inventory_df['asin'] == i , 'material_cost'].values[0]
    EOQ = math.sqrt((2 * demand * mat) / storage)
    pred_demand_by_asin.loc[pred_demand_by_asin['asin'] == i , 'Economic_Order_Quantity'] = round(EOQ)

Economic_Order_Quantity_df = pred_demand_by_asin.sort_values(by=['Economic_Order_Quantity'], ascending = False).reset_index()

# With the EOQ, we are able to determine the optimal amount of product to order in order to maximise cost. Improving inventory management.

# ========================================================
#                   MAIN EXECUTION
# ========================================================

if __name__ == "__main__":
    print('\n')
    print("ABC Analysis Data Frame:")
    print(ABC_analysis_df)
    print('\n')
    print("Inventory Data Frame after JIC/JIT correction for reorder_point:")
    print(new_inventory_df)
    print('\n')
    print("Economic Order Quantity Data Frame:")
    print(Economic_Order_Quantity_df)
