from model.model import MatureModel
import pandas as pd

if __name__ == '__main__':
    data_train = pd.read_csv('data_processing/session_train.csv')
    data_test = pd.read_csv('data_processing/session_test.csv')
    model = MatureModel(data_train, data_test)
    model.save_model('trained_models/mature_model.sav')
    print(f'Mean absolute error: {model.get_mean_error()}')
    print("Parameters: ")
    print(model.get_parameters())
