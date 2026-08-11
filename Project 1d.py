import pandas as pd
from src.data_pipeline import DataPipeline
from src.feature_engineering import create_features
from src.models import MacroForecaster
def run_pipeline(): # 1. Synthesize sample macroeconomic data (e.g., quarterly GDP index)
    dates = pd.date_range(start="1995-03-31", end="2025-12-31", freq="QE")
    np.random.seed(42)
    trend = np.linspace(100, 250, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    gdp_values = trend + noise
    df_raw = pd.DataFrame({'date': dates, 'gdp': gdp_values})
    df_raw.to_csv('data/raw/gdp_sample.csv', index=False) # 2. Pipeline Execution
    pipeline = DataPipeline('data/raw/gdp_sample.csv')
    df = pipeline.load_data() # Make stationary (differencing)
    df_stationary = pipeline.make_stationary(df['gdp']) # 3. Feature Engineering
    lags = [1, 2, 4, 8]  # 1 to 2 years of quarterly lags
    windows = [4, 8]     # 1-year and 2-year rolling windows
    df_features = create_features(df_stationary, target_col='value', lags=lags, windows=windows)
    X = df_features.drop(columns=['value'])
    y = df_features['value'] # 4. Model Training & Evaluation
    forecaster = MacroForecaster()
    forecaster.train_evaluate(X, y)
if __name__ == "__main__":
    run_pipeline()