from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
class SkillsModeler:
    def __init__(self, n_topics: int = 4):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    def extract_topics(self, corpus: list[str]) -> tuple[pd.DataFrame, list[str]]:
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        lda_matrix = self.lda.fit_transform(tfidf_matrix)
        feature_names = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(lda_matrix), feature_names
    def print_top_words(self, feature_names: list[str], n_top_words: int = 5):
        for topic_idx, topic in enumerate(self.lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f"Topic #{topic_idx + 1}: {', '.join(top_features)}")