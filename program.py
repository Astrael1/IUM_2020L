#/usr/bin/python3
import pandas as pd
import argparse

product_data = pd.read_json('resources/products.jsonl', lines=True)
session_data = pd.read_json('resources/sessions.jsonl', lines=True)
users_data = pd.read_json('resources/users.jsonl', lines=True)

# find products with price lower or equal 0 and greater than 1000000 and those which have '#','&' or ';' in name
incorrect_products = product_data[ 
    (product_data["price"] <= 0) | 
    (product_data["price"] > 1000000) |
    (product_data['product_name'].str.contains('#|;|&', regex=True)) ].index

# find values of their 'product_id' column
incorrect_products_ids = product_data["product_id"].iloc[incorrect_products].unique()


# drop incorrect products
product_data.drop(incorrect_products, inplace=True)
# change strings with categories into list of categories
product_data["category_path"] = product_data["category_path"].apply(lambda x: x.split(';'))

# drop rows with null value of columns 'user_id' or 'product_id'
session_data.dropna(subset=['user_id', 'product_id'], inplace=True)
session_data.drop(columns=['purchase_id'], inplace=True)

# replace floats with ints
session_data.user_id = session_data.user_id.astype(int)
session_data.product_id = session_data.product_id.astype(int)
# drop rows with products that are incorrect
incorrect_sessions = session_data[ session_data["product_id"].isin(incorrect_products_ids)].index
print(len(incorrect_sessions))
session_data.drop(incorrect_sessions, inplace=True)

users_data.drop(columns=['name', 'street'], inplace=True)

# set amount of rows print
pd.set_option('display.max_rows', 100)







