from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
def optimize_and_cluster(X: pd.DataFrame, max_k: int = 6) -> tuple[pd.Series, int]:
    best_score = -1
    best_k = 2
    print("Evaluating cluster counts (k)...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k={k} | Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Optimal cluster count selected: {best_k}")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X)
    return pd.Series(cluster_labels, index=X.index, name='cluster'), best_k