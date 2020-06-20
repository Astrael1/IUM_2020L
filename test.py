from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from model.model import BaseModel
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('/preprocessed_data/session_preprocessed.csv')
    model = BaseModel(data, 0)
    print(model.get_mean_error())


