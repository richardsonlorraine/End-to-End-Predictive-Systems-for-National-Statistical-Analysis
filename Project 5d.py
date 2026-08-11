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