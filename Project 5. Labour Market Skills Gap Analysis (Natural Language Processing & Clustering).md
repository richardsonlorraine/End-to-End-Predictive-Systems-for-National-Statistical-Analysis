Project 5. Labour Market Skills Gap Analysis (Natural Language Processing & Clustering)

* Objective: Process national labour force survey narratives, occupational classifications, or job vacancy datasets to map out evolving employer skill demands against current workforce competencies.
* Algorithms to Use: TF-IDF vectorization, Word Embeddings/Transformers, and Latent Dirichlet Allocation (LDA) for topic modeling, followed by cosine similarity mapping.
* Data Sources: National labour force surveys, occupational coding indexes (e.g., Standard Occupational Classification), and aggregated job market vacancy reports.
* Key Focus Areas: Text preprocessing (tokenization, lemmatization, stop-word removal), extracting semantic topics from unstructured survey fields, and visualizing structural shifts in labour requirements over time.

Labour Market Skills Gap Analysis Architecture: An end-to-end  implementation for processing unstructured labour market descriptions and job vacancy narratives using text preprocessing, TF-IDF vectorization, Latent Dirichlet Allocation (LDA) for topic modeling, and cosine similarity mapping to quantify skills alignment.

1. Project Structure

		skills_gap_analysis/
		│
		├── data/
		│   ├── raw/                 # Unstructured survey narratives and vacancy text
		│   └── processed/           # Cleaned tokens, vector matrices, and topic outputs
		│
		├── src/
		│   ├── __init__.py
		│   ├── preprocessing.py     # Tokenization, lemmatization, and stop-word removal
		│   ├── modeling.py          # TF-IDF vectorization and LDA topic modeling
		│   └── alignment.py         # Cosine similarity mapping and gap quantification
		│
		├── main.py                  # Orchestration script
		└── requirements.txt         # Project dependencies

2. Dependencies (requirements.txt)

		pandas>=2.0.0
		numpy>=1.24.0
		scikit-learn>=1.3.0
		gensim>=4.3.0
		nltk>=3.8.0
		matplotlib>=3.7.0
		seaborn>=0.12.0

3. Core Implementation Modules

* Text Preprocessing (src/preprocessing.py): Natural language text from labour surveys requires rigorous cleaning—removing punctuation, lowercasing, dropping domain-specific stop words, and applying lemmatization to normalize word variants.

		import re
		import nltk
		from nltk.corpus import stopwords
		from nltk.stem import WordNetLemmatizer # Ensure required NLTK resources are available
		nltk.download('stopwords', quiet=True)
		nltk.download('wordnet', quiet=True)
		class TextPreprocessor:
		    def __init__(self):
		        self.lemmatizer = WordNetLemmatizer()
		        self.stop_words = set(stopwords.words('english')).union({'experience', 'role', 'position', 'candidate', 'skills', 'work', 'job'})
		    def clean_text(self, text: str) -> str:
		        if not isinstance(text, str):
		            return "" # Lowercase and remove non-alphabetical characters
		        text = text.lower()
		        text = re.sub(r'[^a-z\s]', '', text) # Tokenize and lemmatize
		        tokens = text.split()
		        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
		        return " ".join(cleaned_tokens)

* Modeling & Topic Extraction (src/modeling.py): We apply TF-IDF vectorization followed by Latent Dirichlet Allocation (LDA) to surface underlying skill clusters or thematic requirements across market documents.

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

* Skills Alignment & Gap Quantification (src/alignment.py): By transforming both employer vacancy demands and workforce competencies into vector space, we calculate cosine similarity scores to flag specific skill shortages or surpluses.

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

4. Orchestration Script (main.py)

		import pandas as pd
		import numpy as np
		from src.preprocessing import TextPreprocessor
		from src.modeling import SkillsModeler
		from src.alignment import compute_skills_gap
		def run_pipeline(): # 1. Synthesize sample survey and vacancy text datasets
		    np.random.seed(42)
		    vacancies = ["Seeking experienced data engineer proficient in  sql pipeline architecture and cloud deployment", "Looking for machine learning specialist with strong background in natural language processing transformers pytorch", "Junior data analyst required with strong excel powerbi statistics and basic  skills",
"Senior cloud architect with expertise in kubernetes docker microservices and infrastructure automation"]
		    workforce_profiles = ["Data analyst skilled in excel sql basic  and statistical reporting", "Software engineer with java  cloud services and relational database experience", "Junior researcher with statistics background  and data visualization tools", "IT support specialist with windows networking and hardware troubleshooting expertise"]
		    df = pd.DataFrame({'vacancy_text': vacancies3, 'workforce_text': workforce_profiles3}) # 2. Preprocessing Execution
		    preprocessor = TextPreprocessor()
		    df['clean_vacancy'] = df['vacancy_text'].apply(preprocessor.clean_text)
		    df['clean_workforce'] = df['workforce_text'].apply(preprocessor.clean_text) # 3. Topic Modeling on Vacancies
		    modeler = SkillsModeler(n_topics=3)
		    lda_df, feature_names = modeler.extract_topics(df['clean_vacancy'].tolist())
		    print("--- Extracted Market Demand Topics ---")
		    modeler.print_top_words(feature_names) # 4. Alignment & Gap Analysis using shared vectorizer fit on entire corpus
		    full_corpus = df['clean_vacancy'].tolist() + df['clean_workforce'].tolist()
		    modeler.vectorizer.fit(full_corpus)
		    print("\n--- Computing Skills Gap ---")
		    alignment_scores = compute_skills_gap(df['clean_vacancy'].tolist(), df['clean_workforce'].tolist(),  modeler.vectorizer)
		    df['alignment_score'] = alignment_scores
		    df.to_csv('data/processed/skills_gap_results.csv', index=False)
		    print("\nSample Output Mapping:\n", df[['vacancy_text', 'alignment_score']].head())
		if __name__ == "__main__":
		    run_pipeline()

5. Key Extension Steps for Production

* Transformer Embeddings: Replace TF-IDF/LDA pipelines with pre-trained sentence transformers (e.g., sentence-transformers/all-MiniLM-L6-v2) to capture deeper semantic relationships and synonyms that keyword models miss.
* Temporal Tracking: Index job vacancy texts by quarter or year to track shifts in employer skill requirements over time against historical labour force surveys.

6. Project Summary: This project provides an advanced Natural Language Processing (NLP) framework for quantifying structural alignments between employer skill demands and workforce competencies. Utilizing NLTK for rigorous text preprocessing, TF-IDF vectorization, and Latent Dirichlet Allocation (LDA) for topic modeling, the pipeline extracts thematic market requirements from unstructured survey and vacancy narratives. Cosine similarity mapping is then applied to measure overall and document-level skills gaps.