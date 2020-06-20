from data_processing.preprocess_basic import make_training_set as make_basic_data
from data_processing.preprocess_mature import make_training_set as make_mature_data
from data_processing.preprocess_mature import getListOfCategories
from model.model import load_model, BASE_MODEL_ATTRIBUTES, MATURE_MODEL_ATTRIBUTES
import argparse
import pandas as pd
import numpy as np

category_weights = {
        'Gry i konsole': 1/3,
        'Gry komputerowe': 2/3,
        'Gry na konsole': 1/3,
        'Gry PlayStation3': 1/3,
        'Gry Xbox 360': 1/3,
        'Komputery': 1/3,
        'Drukarki i skanery': 1/3,
        'Biurowe urządzenia wielofunkcyjne': 1/3,
        'Monitory': 1/3,
        'Monitory LCD': 1/3,
        'Tablety i akcesoria': 1/3,
        'Tablety': 1/3,
        'Sprzęt RTV': 1/3,
        'Audio': 1/3,
        'Słuchawki': 1/3,
        'Przenośne audio i video': 1/3,
        'Odtwarzacze mp3 i mp4': 1/3,
        'Video': 1/3,
        'Odtwarzacze DVD': 1/3,
        'Telewizory i akcesoria': 1/6,
        'Anteny RTV': 1/6,
        'Okulary 3D': 1/6,
        'Telefony i akcesoria': 1/3,
        'Akcesoria telefoniczne': 1/3,
        'Zestawy głośnomówiące': 1/3,
        'Zestawy słuchawkowe': 1/3,
        'Telefony komórkowe': 2/3,
        'Telefony stacjonarne': 2/3
    }


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
        products_preprocessed[['product_id']] = products_preprocessed[['product_id']].applymap(np.int64)
        products = pd.read_json('resources/products.jsonl', lines=True)
        users = pd.read_csv(args.u)
        sessions = pd.read_json(args.s, lines=True)
        sessions.user_id = sessions.user_id.astype(int)
        sessions.product_id = sessions.product_id.astype(int)
        data = None
        features = []

        if args.b:
            data = make_basic_data(sessions, products_preprocessed)
            features = BASE_MODEL_ATTRIBUTES
        else:
            data = make_mature_data(products_preprocessed, sessions, users, getListOfCategories(products), category_weights)
            features = MATURE_MODEL_ATTRIBUTES

        model = load_model(args.m)
        print(model.predict(data[features]))




