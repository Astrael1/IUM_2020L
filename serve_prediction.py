from data_processing.preprocess_basic import make_training_set as make_basic_data
from data_processing.preprocess_mature import make_training_set as make_mature_data
from data_processing.preprocess_mature import getListOfCategories, category_weights
from model.model import load_model, BASE_MODEL_ATTRIBUTES, MATURE_MODEL_ATTRIBUTES
import argparse
import pandas as pd
import csv


def get_categories():
    products = pd.read_json('resources/products.jsonl', lines=True)
    incorrect_products = products[
        (products["price"] <= 0) |
        (products["price"] > 1000000) |
        (products['product_name'].str.contains('#|;|&', regex=True))].index
    products.drop(incorrect_products, inplace=True)
    products["category_path"] = products["category_path"].apply(lambda x: x.split(';'))

    return getListOfCategories(products)


products_mature_preprocessed_path = 'data_processing/products_mature_preprocessed.csv'
products_basic_preprocessed_path = 'data_processing/products_basic_preprocessed.csv'
user_preprocessed_path = 'data_processing/users_preprocessed.csv'
model_mature_path = 'trained_models/mature_model.sav'
model_basic_path = 'trained_models/basic_model.sav'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Session purchase probability predictor")
    parser.add_argument('-b', action='store_true', help='Use basic model?')
    parser.add_argument('-s', type=str, help='Sessions data (jsonl)')
    parser.add_argument('-o', type=str, help='Output file name?')
    args = parser.parse_args()

    users = pd.read_csv(user_preprocessed_path)
    users.set_index('user_id', inplace=True)
    sessions = pd.read_json(args.s, lines=True)
    data = None
    features = []
    model = None

    if args.b:
        products_preprocessed = pd.read_csv(products_basic_preprocessed_path)
        products_preprocessed.set_index('product_id', inplace=True)
        model = load_model(model_basic_path)
        data = make_basic_data(sessions, products_preprocessed)
        features = BASE_MODEL_ATTRIBUTES
    else:
        products_preprocessed = pd.read_csv(products_mature_preprocessed_path)
        products_preprocessed.set_index('product_id', inplace=True)
        model = load_model(model_mature_path)
        data = make_mature_data(products_preprocessed, sessions, users, get_categories(),
                                category_weights)
        features = MATURE_MODEL_ATTRIBUTES

    indexes, predictions = data.index.tolist(), model.predict(data[features])
    with open(args.o if args.o is not None else 'predictions.csv', mode='w', newline='') as prediction_file:
        writer = csv.writer(prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(indexes)):
            writer.writerow([indexes[i], predictions[i]])
            print(f'session_id: {indexes[i]} purchase_probability: {predictions[i]}')

