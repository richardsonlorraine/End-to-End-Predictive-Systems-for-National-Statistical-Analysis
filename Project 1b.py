import pandas as pd
def create_features(df: pd.DataFrame, target_col: str, lags: list, windows: list) -> pd.DataFrame:
    """Generates rolling lag and window statistics to capture temporal dynamics. """
    df_feat = df.copy() # Generate lag features
    for lag in lags:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag) # Generate rolling window statistics
    for window in windows:
        roll = df_feat[target_col].shift(1).rolling(window=window)
        df_feat[f'{target_col}_roll_mean_{window}'] = roll.mean()
        df_feat[f'{target_col}_roll_std_{window}'] = roll.std()
    return df_feat.dropna()