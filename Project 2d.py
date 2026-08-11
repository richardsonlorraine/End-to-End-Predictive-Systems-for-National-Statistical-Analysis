import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def generate_profile_heatmap(original_scaled_df: pd.DataFrame, clusters: pd.Series):
    analysis_df = original_scaled_df.copy()
    analysis_df['cluster'] = clusters # Compute mean values of raw/scaled features per cluster
    cluster_profiles = analysis_df.groupby('cluster').mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_profiles, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Socio-Economic Profiles across Regional Clusters")
    plt.xlabel("Indicators")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.savefig("data/processed/cluster_profiles.png")
    plt.show()