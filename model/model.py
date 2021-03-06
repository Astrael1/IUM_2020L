from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from model.tuning import tune_random_forest_regressor

import pickle

BASE_MODEL_ATTRIBUTES = ['duration', 'avg_price', 'avg_discount_percent', 'total_discount_percent',
                         'avg_discount_value',
                         'total_discount_value']

MATURE_MODEL_ATTRIBUTES = ['duration', 'avg_price', 'avg_discount_percent', 'total_discount_percent',
                           'avg_discount_value',
                           'total_discount_value', 'avg_hotness',
                           'total_hotness',
                           'is_top',
                           'liking_score']


class BaseModel:

    def __init__(self, data):
        x = data[BASE_MODEL_ATTRIBUTES]
        y = data.probability
        train_x, self.val_x, train_y, self.val_y = train_test_split(x, y, random_state=0)
        self.model = tune_random_forest_regressor(train_x, train_y).best_estimator_

    def get_mean_error(self):
        predictions = self.model.predict(self.val_x)
        return mean_absolute_error(self.val_y, predictions)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, destination):
        pickle.dump(self.model, open(destination, 'wb'))

    def get_parameters(self):
        return self.model.get_params()


class MatureModel:

    def __init__(self, data_train, data_test):
        x = data_train[MATURE_MODEL_ATTRIBUTES]
        y = data_train.probability
        self.val_x = data_test[MATURE_MODEL_ATTRIBUTES]
        self.val_y = data_test.probability
        self.model = tune_random_forest_regressor(x, y).best_estimator_

    def get_mean_error(self):
        predictions = self.model.predict(self.val_x)
        return mean_absolute_error(self.val_y, predictions)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, destination):
        pickle.dump(self.model, open(destination, 'wb'))

    def get_parameters(self):
        return self.model.get_params()


def load_model(filename):
    return pickle.load(open(filename, 'rb'))
