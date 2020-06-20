from model.model import Model
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('preprocessed_data/session_preprocessed.csv')
    model = Model(data, 0)
    model.save_model('trained_models/basic_model.sav')
    print(f'Mean absolute error: {model.get_mean_error()}')
    print("Parameters: ")
    print(model.get_parameters())
