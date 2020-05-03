#/usr/bin/python3
import pandas as pd
import argparse

product_data = pd.read_json('resources/products.jsonl', lines=True)
session_data = pd.read_json('resources/sessions.jsonl', lines=True)
users_data = pd.read_json('resources/users.jsonl', lines=True)

# change strings with categories into list of categories
product_data["category_path"] = product_data["category_path"].apply(lambda x: x.split(';'))

print(product_data[["product_id", "product_name", "category_path"]].head(10))
print(product_data["category_path"].dtypes)

