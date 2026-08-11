from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
def apply_pca(scaled_df: pd.DataFrame, variance_threshold: float = 0.85) -> pd.DataFrame:
    pca_full = PCA().fit(scaled_df)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Selecting {n_components} components to explain {variance_threshold*100}% of variance.")
    pca = PCA(n_components=n_components)
    reduced_array = pca.fit_transform(scaled_df)
    columns = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(reduced_array, index=scaled_df.index, columns=columns)