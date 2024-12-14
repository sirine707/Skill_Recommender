import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load Spacy model for French
nlp = spacy.load('fr_core_news_sm')

# Define French stopwords
french_stopwords = set(stopwords.words('french'))

def preprocess_text(text):
    """Preprocess the text data."""
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    text = ' '.join(word for word in text.split() if word not in french_stopwords)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text