Project 1. Macroeconomic Indicator Forecasting (Time Series & Regression)

* Objective: Predict key national economic indicators—such as quarterly Gross Domestic Product (GDP) growth, inflation rates, or unemployment levels—using historical economic time series.
* Algorithms to Use: ARIMA, Seasonal decomposition of time series (STL), Long Short-Term Memory (LSTM) networks, or Gradient Boosting Regressors (XGBoost/LightGBM) with lag features.
* Data Sources: Office for National Statistics (ONS), Federal Reserve Economic Data (FRED), or World Bank Open Data.
* Key Focus Areas: Handling non-stationary data, capturing seasonality, engineering rolling lag features, and evaluating models using Root Mean Squared Error (RMSE) against baseline economic forecasts.
* Macroeconomic Forecasting Project Architecture: An end-to-end  implementation for forecasting a national economic indicator (e.g., quarterly GDP growth or unemployment) using a Gradient Boosting Regressor (LightGBM) engineered with rolling temporal lag features.

1. Project Structure: To maintain a modular, production-ready design, structure the project directory as follows:

		macro_forecasting/
		│
		├── data/
		│   ├── raw/                 # Original downloaded time-series files
		│   └── processed/           # Cleaned and feature-engineered datasets
		│
		├── src/
		│   ├── __init__.py
		│   ├── data_pipeline.py     # Data ingestion and stationarity transformations
		│   ├── feature_engineering.py # Lag generation and rolling windows
		│   ├── models.py            # Model training and hyperparameter structures
		│   └── evaluation.py        # Metrics computation and visualization
		│
		├── main.py                  # Orchestration script
		└── requirements.txt         # Project dependencies

2. Dependencies (requirements.txt)
		pandas>=2.0.0
		numpy>=1.24.0
		statsmodels>=0.14.0
		lightgbm>=4.0.0
		scikit-learn>=1.3.0
		matplotlib>=3.7.0

3. Core Implementation Modules

* Data Ingestion & Stationarity (src/data_pipeline.py): Macroeconomic time series are frequently non-stationary (exhibiting trends or random walks). We apply differencing to achieve stationarity, validated via the Augmented Dickey-Fuller (ADF) test.

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

* Feature Engineering (src/feature_engineering.py): Tree-based models like LightGBM cannot inherently capture sequential order. We transform the time series into a supervised learning problem by generating lag features and rolling statistics.

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

* Model Training (src/models.py): Using Time-Series Split cross-validation prevents data leakage from the future into the past.

		import lightgbm as lgb
		import pandas as ps 
		from sklearn.model_selection import Timeseriessplit 
		from sklearn.metrics import root_mean_squared_error
		class MacroForecaster:
		    def __init__(self):
		        self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
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

4. Orchestration Script (main.py)

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

5. Key Extension Steps for Production

* Exogenous Variables: Incorporate broader macroeconomic drivers (e.g., interest rates, exchange rates, energy prices) into X alongside lag features.
* Baseline Comparison: Always benchmark your LightGBM model against a naive persistence model (yt = yt-1) and a statistical baseline like an ARIMA(p,d,q) model using the same evaluation folds.

Project Summary: This project demonstrates an end-to-end time series pipeline designed to forecast key national economic indicators. By addressing non-stationarity through Augmented Dickey-Fuller validation and first-order differencing, the pipeline transforms temporal sequences into supervised learning features using rolling lags and window statistics. Utilizing a tuned LightGBM regressor evaluated via Time-Series Split cross-validation, the model achieves robust predictive performance while preventing future data leakage.