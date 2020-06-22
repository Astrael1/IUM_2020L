from model.model import BaseModel
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data_processing/session_preprocessed.csv')
    model = BaseModel(data)
    model.save_model('trained_models/basic_model.sav')
    print(f'Mean absolute error: {model.get_mean_error()}')
    print("Parameters: ")
    print(model.get_parameters())
