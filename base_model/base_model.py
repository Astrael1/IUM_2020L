from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from base_model.tuning import tune_random_forest_regressor


class BaseModel:

    def __init__(self, data):
        x = data[['duration', 'avg_price', 'avg_discount_percent', 'total_discount_percent', 'avg_discount_value',
                  'total_discount_value']]
        y = data.probability
        train_x, self.val_x, train_y, self.val_y = train_test_split(x, y, random_state=0)
        self.model = tune_random_forest_regressor(train_x, train_y).best_estimator_

    def get_mean_error(self):
        predictions = self.model.predict(self.val_x)
        return mean_absolute_error(self.val_y, predictions)

    def predict(self, x):
        return self.model.predict(x)
