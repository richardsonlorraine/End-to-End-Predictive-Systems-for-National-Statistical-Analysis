import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
class MacroForecaster:
    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42,
verbose=-1)
    def train_evaluate(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            fold_rmse = root_mean_squared_error(y_test, predictions)
            rmse_scores.append(fold_rmse)
            print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
        print(f"Mean Cross-Validation RMSE: {sum(rmse_scores)/len(rmse_scores):.4f}") # Fit final model on complete dataset
        self.model.fit(X, y)
        return self.model