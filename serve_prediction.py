from data_processing.preprocess_basic import make_training_set as make_basic_data
from data_processing.preprocess_mature import make_training_set as make_mature_data
from data_processing.preprocess_mature import getListOfCategories, category_weights
from model.model import load_model, BASE_MODEL_ATTRIBUTES, MATURE_MODEL_ATTRIBUTES
import argparse
import pandas as pd
import numpy as np


def get_categories():
    products = pd.read_json('resources/products.jsonl', lines=True)
    incorrect_products = products[
        (products["price"] <= 0) |
        (products["price"] > 1000000) |
        (products['product_name'].str.contains('#|;|&', regex=True))].index
    products.drop(incorrect_products, inplace=True)
    products["category_path"] = products["category_path"].apply(lambda x: x.split(';'))

    return getListOfCategories(products)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Session purchase probability predictor")
    parser.add_argument('-m', type=str, help="Saved model path")
    parser.add_argument('-b', action='store_true', help='Use basic model?')
    parser.add_argument('-p', type=str, help='Products preprocessed file')
    parser.add_argument('-s', type=str, help='Sessions json file')
    parser.add_argument('-u', type=str, help='Users preprocessed file')
    args = parser.parse_args()

    if args.p is None or args.s is None or args.u is None:
        print("Give all args!")
    else:
        products_preprocessed = pd.read_csv(args.p)
        products_preprocessed.set_index('product_id', inplace=True)
        users = pd.read_csv(args.u)
        users.set_index('user_id', inplace=True)
        sessions = pd.read_json(args.s, lines=True)
        data = None
        features = []

        if args.b:
            data = make_basic_data(sessions, products_preprocessed)
            features = BASE_MODEL_ATTRIBUTES
        else:
            data = make_mature_data(products_preprocessed, sessions, users, get_categories(),
                                    category_weights)
            features = MATURE_MODEL_ATTRIBUTES

        model = load_model(args.m)
        print(data.index.tolist())
        print(model.predict(data[features]))
