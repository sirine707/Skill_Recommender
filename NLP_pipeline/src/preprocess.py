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
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    text=re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in french_stopwords]
    # Lemmatize
    doc = nlp(' '.join(words))
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)