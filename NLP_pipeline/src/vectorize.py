import numpy as np
import gensim.downloader as api

# Load the pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

def vectorize_text(text, model=word2vec_model):
    words = text.split()  # Split text into words
    vectors = []

    for word in words:
        if word in model:  # If the word exists in the model's vocabulary
            vectors.append(model[word])
        else:
            vectors.append(np.zeros(300))  # Zero vector for unknown words

    # Average the word vectors to get a single vector
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)  # Return a zero vector if the text is empty