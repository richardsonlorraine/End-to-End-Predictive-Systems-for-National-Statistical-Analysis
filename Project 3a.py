from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
class MigrationModeler:
    def __init__(self):
        self.model = ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    def evaluate_and_train(self, X: pd.DataFrame, y: pd.Series):
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        print(f"Cross-Validation RMSE Scores: {-scores}")
        print(f"Mean RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
        # Fit final model on full dataset
        self.model.fit(X, y)
        return self.model