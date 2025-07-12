import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import json

# Load CSVs
df = pd.read_csv("sales_data.csv")
stock_df = pd.read_csv("stock_data.csv")

products = df['product'].unique()
result = []

for product in products:
    product_df = df[df['product'] == product]
    X = np.array(product_df['day']).reshape(-1, 1)
    y = np.array(product_df['sales'])

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array([8, 9, 10]).reshape(-1, 1)
    predictions = model.predict(future_days)

    avg_pred = int(np.mean(predictions))  # average of next 3 days
    current_stock = int(stock_df[stock_df['product'] == product]['current_stock'].values[0])
    reorder_qty = max(0, avg_pred - current_stock)

    result.append({
        "product": product,
        "predicted_demand_next_3_days": avg_pred,
        "current_stock": current_stock,
        "reorder_suggestion": reorder_qty
    })

# Save to JSON
with open("predicted_demand.json", "w") as f:
    json.dump(result, f, indent=4)

print("✅ Updated predicted_demand.json with reorder suggestions")
