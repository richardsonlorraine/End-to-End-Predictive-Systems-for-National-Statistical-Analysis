from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
def compute_skills_gap(vacancy_corpus: list[str], workforce_corpus: list[str], vectorizer) -> pd.Series:
    """Computes cosine similarity between aggregate employer demand and workforce profiles."""
    vac_matrix = vectorizer.transform(vacancy_corpus)
    work_matrix = vectorizer.transform(workforce_corpus) # Calculate mean vector for demand vs supply
    mean_demand = np.asarray(vac_matrix.mean(axis=0))
    mean_supply = np.asarray(work_matrix.mean(axis=0))
    similarity = cosine_similarity(mean_demand, mean_supply)[0][0]
    print(f"Overall Workforce-Employer Alignment Score (Cosine Similarity): {similarity:.4f}") # Per-document gap mapping
    doc_similarities = cosine_similarity(vac_matrix, work_matrix).diagonal()
    return pd.Series(doc_similarities, name='alignment_score')