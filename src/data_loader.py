import pandas as pd
import ast
import os
from datetime import datetime


class RetailDataManager:
    def __init__(self):
        inv_path = 'data/product_inventory.csv' if os.path.exists('data/product_inventory.csv') else 'product_inventory.csv'
        ord_path = 'data/orders.csv' if os.path.exists('data/orders.csv') else 'orders.csv'

        self.inventory = pd.read_csv(inv_path)
        self.orders = pd.read_csv(ord_path)

        # Parse stock_per_size from string to dict
        self.inventory['stock_per_size'] = self.inventory['stock_per_size'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Simulation date: May 1, 2026
        self.current_date = datetime(2026, 5, 1)

    def get_inventory(self):
        return self.inventory

    def get_order_by_id(self, order_id: str):
        clean_id = str(order_id).strip().upper()
        order = self.orders[self.orders['order_id'].astype(str).str.upper() == clean_id]
        return order.iloc[0] if not order.empty else None

    def get_product_by_id(self, product_id: str):
        clean_id = str(product_id).strip().upper()
        product = self.inventory[self.inventory['product_id'].astype(str).str.upper() == clean_id]
        return product.iloc[0] if not product.empty else None