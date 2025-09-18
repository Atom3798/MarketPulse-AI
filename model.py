import xgboost as xgb
from sklearn.metrics import mean_squared_error

class MarketModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {"rmse": mean_squared_error(y_true, y_pred, squared=False)}
