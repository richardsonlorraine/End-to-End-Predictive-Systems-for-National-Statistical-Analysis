Project 4. Public Expenditure and Budget Efficiency Modeling (Anomaly Detection)

* Objective: Analyze regional public spending or local government procurement datasets to identify administrative outliers, cost anomalies, or unusual budget allocations that deviate significantly from national norms.
* Algorithms to Use: Isolation Forests, One-Class SVMs, or Local Outlier Factor (LOF).
* Data Sources: Open government spending transparency data, local authority financial returns, and national public sector expenditure tables.
* Key Focus Areas: Dealing with heavily skewed financial distributions, high-cardinality categorical variables (e.g., spending categories or supplier IDs), and building an automated scoring pipeline to flag anomalous records.

Public Expenditure & Budget Efficiency Modeling Architecture: An end-to-end implementation for detecting administrative outliers and budget allocation anomalies in public sector spending data using an Isolation Forest, complete with pipelines for handling heavily skewed financial distributions and high-cardinality categorical variables.

1. Project Structure

		budget_anomaly_detection/
		│
		├── data/
		│   ├── raw/                 # Original government spending and procurement files
		│   └── processed/           # Transformed features and scored anomaly outputs
		│
		├── src/
		│   ├── __init__.py
		│   ├── preprocessing.py     # Log transformation, scaling, and categorical encoding
		│   ├── modeling.py          # Isolation Forest anomaly scoring pipeline
		│   └── evaluation.py        # Outlier profiling and risk reporting
		│
		├── main.py                  # Orchestration script
		└── requirements.txt         # Project dependencies

2. Dependencies (requirements.txt)

		pandas>=2.0.0
		numpy>=1.24.0
		scikit-learn>=1.3.0
		scipy>=1.10.0
		matplotlib>=3.7.0
		seaborn>=0.12.0

3. Core Implementation Modules

* Preprocessing & Skew Correction (src/preprocessing.py): Public financial distributions (e.g., transaction costs, procurement line items) are heavily right-skewed. We apply a log1p transformation to normalize spending distributions before scaling and encoding high-cardinality attributes like supplier IDs.

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

* Anomaly Detection Modeling (src/modeling.py): We use an Isolation Forest, which isolates observations by randomly selecting features and split values. Anomalies require fewer splits to isolate than normal data points, resulting in shorter path lengths and higher anomaly scores.

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

* Profiling & Risk Reporting (src/evaluation.py): Summarizing the highest-scoring financial records helps audit teams prioritize investigations into unusual budget allocations.

		import pandas as pd
		def generate_audit_report(df_raw: pd.DataFrame, labels: pd.Series, scores: pd.Series, top_n: int = 5) -> pd.DataFrame:
		    report_df = df_raw.copy()
		    report_df['anomaly_label'] = labels
		    report_df['anomaly_risk_score'] = scores # Filter for flagged outliers and sort by risk score
		    outliers = report_df[report_df['anomaly_label'] == -1].sort_values(by='anomaly_risk_score', ascending=False)
		    print(f"\n--- Top {top_n} Highest Risk Spending Anomalies ---")
		    print(outliers[['supplier_id', 'spending_category', 'transaction_amount', 'anomaly_risk_score']].head(top_n))
		    return outliers

4. Orchestration Script (main.py)

		import pandas as pd
		import numpy as np
		from src.preprocessing import SpendingPreprocessor
		from src.modeling import BudgetAnomalyDetector
		from src.evaluation import generate_audit_report
		def run_pipeline(): # 1. Synthesize sample public procurement / spending dataset
		    np.random.seed(42)
		    n_records = 1000
		    suppliers = [f"SUPP_{i:04d}" for i in range(50)]
		    categories = ['IT Hardware', 'Consultancy', 'Facility Management', 'Legal Services', 'Travel']
		    data = {'transaction_id': [f"TXN_{i:06d}" for i in range(n_records)], 'supplier_id': np.random.choice(suppliers, n_records), 'spending_category': 				np.random.choice(categories, n_records), 'transaction_amount': np.random.exponential(scale=5000, size=n_records) + 100, # Heavily skewed 'budget_variance_pct': np.random.normal(0.0, 0.05, n_records), 'approval_delay_days': np.random.poisson(3, n_records)}
		    df_raw = pd.DataFrame(data) # Inject intentional synthetic outliers (extreme spending amounts)
		    df_raw.loc[10, 'transaction_amount'] = 250000.0
		    df_raw.loc[45, 'budget_variance_pct'] = 0.85
		    df_raw.loc[100, 'approval_delay_days'] = 45
		    df_raw.to_csv('data/raw/public_spending_sample.csv', index=False) # 2. Define Feature Lists
		    numeric_features = ['transaction_amount', 'budget_variance_pct', 'approval_delay_days']
		    high_card_features = ['supplier_id', 'spending_category'] # Create a dummy target proxy for target encoding (e.g., standard baseline variance check)
		    target_proxy = pd.Series(np.where(df_raw['transaction_amount'] > 50000, 1, 0), index=df_raw.index) # 3. Preprocessing Execution
		    preprocessor = SpendingPreprocessor(numeric_features, high_card_features)
		    X_processed = preprocessor.fit_transform(df_raw, target_proxy) # 4. Anomaly Detection Execution
		    detector = BudgetAnomalyDetector(contamination=0.03)
		    labels, scores = detector.fit_predict(X_processed) # 5. Audit Report Generation
    generate_audit_report(df_raw, labels, scores, top_n=5)
		if __name__ == "__main__":
		    run_pipeline()

5. Key Extension Steps for Production

* Ensemble Outlier Detection: Combine Isolation Forest scores with Local Outlier Factor (LOF) and One-Class SVM outputs using a voting threshold to reduce false positive rates on complex public expenditure portfolios.
* Temporal Windowing: Group transactions into rolling monthly or quarterly spending windows to capture seasonal budget-dumping behavior near fiscal year-ends.

6. Project Summary: This project establishes an automated anomaly detection system designed to flag administrative outliers and high-risk procurement irregularities in public spending data. It addresses heavily skewed financial distributions using log transformations (\log1p) and handles high-cardinality attributes like supplier IDs via target encoding. Powered by an Isolation Forest algorithm, the pipeline generates intuitive risk scores and automated audit reports to prioritize financial investigations.