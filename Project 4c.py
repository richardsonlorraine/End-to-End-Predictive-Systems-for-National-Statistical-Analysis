import pandas as pd
def generate_audit_report(df_raw: pd.DataFrame, labels: pd.Series, scores: pd.Series, top_n: int = 5) -> pd.DataFrame:
    report_df = df_raw.copy()
    report_df['anomaly_label'] = labels
    report_df['anomaly_risk_score'] = scores # Filter for flagged outliers and sort by risk score
    outliers = report_df[report_df['anomaly_label'] == -1].sort_values(by='anomaly_risk_score', ascending=False)
    print(f"\n--- Top {top_n} Highest Risk Spending Anomalies ---")
    print(outliers[['supplier_id', 'spending_category', 'transaction_amount', 'anomaly_risk_score']].head(top_n))
    return outliers