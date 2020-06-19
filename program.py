#/usr/bin/python3
import pandas as pd
import numpy as np
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
all_session_ids = session_data.session_id.unique()

def getSessionOfId(session_data, id):
    return session_data[session_data.session_id == id]

def getSuccessfullSessions(session_data, session_ids):
    return [id for id in session_ids if (getSessionOfId(session_data, id).event_type == 'BUY_PRODUCT').any() ]

# print(session_data[session_data.session_id.isin(getSuccessfullSessions(session_data, all_session_ids))])

def getListOfCategories(product_data):
    a = set()
    for categories in product_data.category_path:
        for item in categories:
            a.add(item)
    return sorted(list(a))



groupped_sessions = session_data.groupby('session_id')

def getSessionAttributes(session, products):

    s_id = session['session_id'].unique()[0]
  
    timestamps = session.iloc[[0,-1]].timestamp.values
    duration = (timestamps[1] - timestamps[0])/ np.timedelta64(1,'s')

    discount = session[ 'offered_discount'].values

    product_ids = group['product_id']
    viewed_products_data = products[products.product_id.isin(product_ids)][['price']]
    try:
        viewed_products_data.insert(0, 'offered_discount', discount, True)
    except ValueError:
        print("ERROR", viewed_products_data)
        import pdb; pdb.set_trace()
    viewed_products_data.insert(1, 'discount_value', 
        viewed_products_data.price * viewed_products_data.offered_discount, True)

    mean_values = viewed_products_data.mean()

    avg_price = mean_values.price
    avg_discount_percent = mean_values.offered_discount
    avg_discount_value = mean_values.discount_value

    sum_values = viewed_products_data.sum()
    total_discount_percent = sum_values.offered_discount
    total_discount_value = sum_values.discount_value
    
    result = {
            'id': s_id,
            'duration': duration,
            'avg_price': avg_price,
            'avg_discount_percent': avg_discount_percent,
            'total_discount_percent': total_discount_percent,
            'avg_dicount_value': avg_discount_value,
            'total_discount_value': total_discount_value
        }
    return result

data = []

for name, group in groupped_sessions:
    data.append(getSessionAttributes(group, product_data))

sessiondf = pd.DataFrame(data)
    

import pdb; pdb.set_trace()







