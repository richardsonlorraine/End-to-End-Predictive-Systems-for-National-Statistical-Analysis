import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
class SpendingPreprocessor:
    def __init__(self, numeric_features: list, high_card_features: list):
        self.numeric_features = numeric_features
        self.high_card_features = high_card_features
        self.pipeline = None
    def fit_transform(self, df: pd.DataFrame, target_anomaly_proxy: pd.Series) -> pd.DataFrame: # Log-transform skewed financial variables to normalize variance
        df_processed = df.copy()
        for col in self.numeric_features:
            df_processed[f'log_{col}'] = np.log1p(df_processed[col])
        log_numeric_cols = [f'log_{col}' for col in self.numeric_features] # Pipeline for log-numeric features: standardization
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        # Pipeline for high-cardinality variables (e.g., supplier IDs): Target Encoding
        categorical_transformer = Pipeline(steps=[('encoder', TargetEncoder(smooth="auto", random_state=42))])
        self.pipeline = ColumnTransformer(transformers=[('num', numeric_transformer, log_numeric_cols), ('cat', categorical_transformer, self.high_card_features)])
        transformed_array = self.pipeline.fit_transform(df_processed, target_anomaly_proxy)
        all_cols = log_numeric_cols + self.high_card_features
        return pd.DataFrame(transformed_array, columns=all_cols, index=df.index)