import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
class DataPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, parse_dates=['date'], index_col='date')
        return df
    @staticmethod
    def check_stationarity(series: pd.Series) -> bool:
        result = adfuller(series.dropna())
        p_value = result[1]
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {p_value:.4f}")
        return p_value < 0.05
    def make_stationary(self, series: pd.Series) -> pd.DataFrame:
        if not self.check_stationarity(series):
            print("Series is non-stationary. Applying first-order differencing...")
            stationary_series = series.diff().dropna()
        else:
            print("Series is already stationary.")
            stationary_series = series
        return pd.DataFrame({'value': stationary_series})