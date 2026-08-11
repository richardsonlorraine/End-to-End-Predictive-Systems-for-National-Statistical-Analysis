import pandas as pd
import numpy as np
from src.preprocessing import CensusPreprocessor
from src.modeling import MigrationModeler
from src.interpretation import plot_feature_importance
def run_pipeline(): # 1. Synthesize sample census and housing record data for administrative zones
    np.random.seed(42)
    n_zones = 500
    data = {'zone_id': [f"Zone_{i:04d}" for i in range(n_zones)], 'housing_completions': np.random.poisson(45, n_zones), 'wage_differential': np.random.normal(2500, 800, n_zones), 'median_rent': np.random.normal(850, 150, n_zones), 'unemployment_rate': np.random.uniform(0.02, 0.12, n_zones), 'urban_rural_class': np.random.choice(['Urban', 'Suburban', 'Rural'], n_zones, p=[0.4, 0.4, 0.2]), 'net_migration_rate': np.random.normal(1.2, 2.5, n_zones) # Target Variable} # Introduce random missing values to simulate small-area census suppression
    df_raw = pd.DataFrame(data)
    mask = np.random.rand(*df_raw.shape) < 0.05
    df_raw[mask] = np.nan # Restore identifiers and target
    df_raw['zone_id'] = data['zone_id']
    df_raw['net_migration_rate'] = data['net_migration_rate']
    df_raw.to_csv('data/raw/census_migration_sample.csv', index=False) # 2. Define Feature Lists
    numeric_features = ['housing_completions', 'wage_differential', 'median_rent', 'unemployment_rate']
    categorical_features = ['urban_rural_class'] # 3. Preprocessing Pipeline Execution
    preprocessor = CensusPreprocessor(numeric_features, categorical_features)
    X_processed, feature_names = preprocessor.fit_transform(df_raw)
    y = df_raw['net_migration_rate'] # Drop rows where target became NaN due to synthetic missing mask
    valid_idx = y.dropna().index
    X_processed = X_processed.loc[valid_idx]
    y = y.loc[valid_idx] # 4. Model Training & Evaluation
    modeler = MigrationModeler()
    trained_model = modeler.evaluate_and_train(X_processed, y) # 5. Interpretation Output
    plot_feature_importance(trained_model, feature_names)
if __name__ == "__main__":
    run_pipeline()