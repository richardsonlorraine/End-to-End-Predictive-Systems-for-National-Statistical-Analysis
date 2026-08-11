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