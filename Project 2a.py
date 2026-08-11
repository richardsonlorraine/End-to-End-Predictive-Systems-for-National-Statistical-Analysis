import pandas as pd
from sklearn.preprocessing import StandardScaler
class DataPreprocessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.scaler = StandardScaler()
    def load_and_scale(self, region_col: str) -> tuple[pd.DataFrame, pd.Index]:
        df = pd.read_csv(self.filepath) # Isolate region identifiers and numeric features
        regions = df[region_col]
        features = df.drop(columns=[region_col]) # Standardize features
        scaled_array = self.scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_array, columns=features.columns, index=regions)
        return scaled_df, features.columns