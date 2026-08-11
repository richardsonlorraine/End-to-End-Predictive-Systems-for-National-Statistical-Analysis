from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
class BudgetAnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    def fit_predict(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]: # Fit model and predict (-1 for outliers, 1 for inliers)
        preds = self.model.fit_predict(X) # Calculate anomaly score (lower/more negative scores indicate stronger anomalies)
        raw_scores = self.model.decision_function(X) # Convert to an intuitive risk score (0 to 1, where 1 is highest risk)
        risk_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        labels = pd.Series(preds, index=X.index, name='anomaly_label')
        scores = pd.Series(risk_scores, index=X.index, name='anomaly_risk_score')
        print(f"Flagged {(preds == -1).sum()} anomalies out of {len(preds)} total records.")
        return labels, scores