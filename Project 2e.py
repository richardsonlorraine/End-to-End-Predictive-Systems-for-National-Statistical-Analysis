import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.dimensionality import apply_pca
from src.clustering import optimize_and_cluster
from src.profiling import generate_profile_heatmap
def run_pipeline(): # 1. Synthesize sample regional dataset (e.g., local authorities)
    np.random.seed(42)
    n_regions = 100
    regions = [f"District_{i:03d}" for i in range(n_regions)]
    data = {'region': regions, 'avg_income': np.random.normal(30000, 5000, n_regions), 'employment_rate': np.random.uniform(0.70, 0.95, n_regions), 'higher_ed_pct': np.random.uniform(0.15, 0.60, n_regions), 'health_index': np.random.normal(100, 10, n_regions), 'housing_cost_ratio': np.random.normal(7.5, 1.5, n_regions)}
    df_raw = pd.DataFrame(data)
    df_raw.to_csv('data/raw/regional_stats.csv', index=False) # 2. Preprocess & Scale
    preprocessor = DataPreprocessor('data/raw/regional_stats.csv')
    scaled_df, feature_names = preprocessor.load_and_scale(region_col='region') # 3. Dimensionality Reduction (PCA)
    reduced_df = apply_pca(scaled_df, variance_threshold=0.90) # 4. Optimal Clustering
    clusters, optimal_k = optimize_and_cluster(reduced_df, max_k=5) # 5. Profiling & Heatmap Output
    generate_profile_heatmap(scaled_df, clusters)
if __name__ == "__main__":
    run_pipeline()