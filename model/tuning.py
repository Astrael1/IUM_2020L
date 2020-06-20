from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def tune_random_forest_regressor(train_x, train_y):
    params_grid = {'bootstrap': [True, False],
                   'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    model = RandomForestRegressor()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(train_x, train_y)

    return random_search
