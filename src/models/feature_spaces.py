from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import gensim

def create_tfidf(fit=False, X_train=None, X_test=None):

    tfidf_vectorizer = TfidfVectorizer()

    if fit:
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        return tfidf_vectorizer, X_test_tfidf, X_test_tfidf

    return tfidf_vectorizer

def create_avg_word_embeddings(vectors_path, generate_missing=False, k=300, fit=False, X_train=None, X_test=None):

    vectors = gensim.models.KeyedVectors.load_word2vec_format(vectors_path, binary=True)

    avg_word_embeddings_transformer = AvgWordEmbeddingsTransformer(vectors, generate_missing)

    if fit:
        X_train_tfidf = avg_word_embeddings_transformer.fit_transform(X_train)
        X_test_tfidf = avg_word_embeddings_transformer.transform(X_test)

        return avg_word_embeddings_transformer, X_test_tfidf, X_test_tfidf

    return avg_word_embeddings_transformer

class AvgWordEmbeddingsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vectors, generate_missing=True, k=300):
        self.vectors = vectors
        self.generate_missing = generate_missing
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        embeddings = X.apply(lambda x: get_average_word2vec(x, self.vectors, generate_missing=self.generate_missing))
        return list(embeddings)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged