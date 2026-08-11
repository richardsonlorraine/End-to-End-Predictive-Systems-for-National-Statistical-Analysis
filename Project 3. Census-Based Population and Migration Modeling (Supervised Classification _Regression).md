Project 3. Census-Based Population and Migration Modeling (Supervised Classification /Regression)

* Objective: Predict demographic shifts, local population growth, or net migration rates for specific administrative zones based on historical census variables, housing completions, and regional wage differentials.
* Algorithms to Use: Random Forests, Extra Trees, or Elastic Net Regression.
* Data Sources: National decennial census datasets, annual population estimates, and local government housing development records.
* Key Focus Areas: Managing mixed-type data (continuous and categorical features), handling missing or suppressed small-area census cells, and interpreting feature importance to see which variables drive localized growth.

Census-Based Population and Migration Modeling Architecture: An end-to-end  implementation for modeling net migration rates or local population growth across administrative zones using an Extra Trees Regressor, complete with preprocessing pipelines for mixed-type tabular data.

1. Project Structure

		census_migration_model/
		│
		├── data/
		│   ├── raw/                 # Original census and administrative records
		│   └── processed/           # Imputed and encoded feature matrices
		│
		├── src/
		│   ├── __init__.py
		│   ├── preprocessing.py     # Missing value imputation and mixed-type encoding
		│   ├── modeling.py          # Extra Trees regressor and K-Fold validation
		│   └── interpretation.py    # Feature importance extraction and visualization
		│
		├── main.py                  # Orchestration script
		└── requirements.txt         # Project dependencies

2. Dependencies (requirements.txt

		pandas>=2.0.0
		numpy>=1.24.0
		scikit-learn>=1.3.0
		matplotlib>=3.7.0
		seaborn>=0.12.0

3. Core Implementation Modules

* Preprocessing & Mixed-Type Encoding (src/preprocessing.py): Census small-area data often contains missing values or suppressed cells alongside a mix of continuous financial metrics and categorical geographic tiers. We use ColumnTransformer to handle these distinct data types cleanly.

		import pandas as pd
		from sklearn.compose import ColumnTransformer
		from sklearn.pipeline import Pipeline
		from sklearn.impute import SimpleImputer
		from sklearn.preprocessing import StandardScaler, OneHotEncoder
		class CensusPreprocessor:
		    def __init__(self, numeric_features: list, categorical_features: list):
		        self.numeric_features = numeric_features
		        self.categorical_features = categorical_features
		        self.pipeline = self._build_pipeline()
		    def _build_pipeline(self) -> ColumnTransformer: # Pipeline for continuous variables: median imputation + scaling
		        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]) # Pipeline for categorical variables: most frequent imputation + one-hot encoding
		        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
		        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_features),
('cat', categorical_transformer, self.categorical_features)])
		        return preprocessor
		    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
		        X = df[self.numeric_features + self.categorical_features]
		        transformed_array = self.pipeline.fit_transform(X) # Extract generated column names for interpretability
		        cat_encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
        encoded_cat_features = list(cat_encoder.get_feature_names_out(self.categorical_features))
		        all_feature_names = self.numeric_features + encoded_cat_features
		        return pd.DataFrame(transformed_array, columns=all_feature_names, index=df.index), all_feature_names

* Model Training & Validation (src/modeling.py): We employ an Extra Trees Regressor (Extremely Randomized Trees), which reduces variance further than standard random forests by randomizing split thresholds, making it robust against noisy small-area census data.

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
		        print(f"Mean RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})") # Fit final model on full dataset
		        self.model.fit(X, y)
		        return self.model

* Feature Importance Interpretation (src/interpretation.py): Extracting feature importances lets policymakers see whether housing completions, wage differentials, or baseline demographic structures drive net migration.

		import pandas as pd
		import matplotlib.pyplot as plt
		import seaborn as sns
		def plot_feature_importance(model, feature_names: list, top_n: int = 10):
		    importances = model.feature_importances_
		    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
		    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(top_n)
		    plt.figure(figsize=(10, 6))
		    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis', hue='feature', legend=False)
		    plt.title(f"Top {top_n} Features Driving Net Migration/Population Growth")
		    plt.xlabel("Feature Importance Score")
		    plt.ylabel("Features")
		    plt.tight_layout()
		    plt.savefig("data/processed/feature_importance.png")
		    plt.show()

4. Orchestration Script (main.py)

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
		    y = df_raw['net_migration_rate'] # Drop rows where target became NaN due to synthetic missing mask.
		    valid_idx = y.dropna().index
		    X_processed = X_processed.loc[valid_idx]
		    y = y.loc[valid_idx] # 4. Model Training & Evaluation
		    modeler = MigrationModeler()
		    trained_model = modeler.evaluate_and_train(X_processed, y) # 5. Interpretation Output
		    plot_feature_importance(trained_model, feature_names)
		if __name__ == "__main__":
		    run_pipeline()

5. Key Extension Steps for Production

* Spatial Lag Inclusion: Add spatial coordinate features (latitude/longitude centroids) or spatially lagged target variables (average net migration of contiguous neighboring zones) to account for spatial autocorrelation.
* Elastic Net Comparison: Run parallel evaluations using Elastic Net Regression to examine linear coefficient tradeoffs against non-linear Extra Trees feature importances.

6. Project Summary: This project delivers a robust supervised regression framework for predicting net migration rates and local population growth across administrative zones. It features a comprehensive preprocessing pipeline that handles mixed-type tabular features, small-area census data suppression, and missing values using median imputation and one-hot encoding. Modeled via an Extra Trees Regressor with K-Fold cross-validation, the system extracts critical feature importances to guide evidence-based policy decisions.