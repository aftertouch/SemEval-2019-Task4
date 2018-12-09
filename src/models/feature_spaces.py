"""
@author: Jonathan
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
from glob import glob
import multiprocessing
import logging
import pandas as pd
import ast
from tqdm import tqdm


# Create TF-IDF feature space
def create_tfidf(fit=False, DATA_INTERIM_PATH=None):

    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit vectorizer on training data and transform training data
    if fit:
        train = pd.read_csv(DATA_INTERIM_PATH + 'train_p.csv', usecols=['preprocessed_text'])
        val = pd.read_csv(DATA_INTERIM_PATH + 'val_p.csv', usecols=['preprocessed_text'])

        X_train_tfidf = tfidf_vectorizer.fit_transform(train['preprocessed_text'])
        X_test_tfidf = tfidf_vectorizer.transform(val['preprocessed_text'])

        return tfidf_vectorizer, X_train_tfidf, X_test_tfidf

    return tfidf_vectorizer


# Create TaggedDocuments for doc2vec
def create_tagged_documents(DATA_PATH):

    DATA_INTERIM_PATH = DATA_PATH + 'interim/'
    DATA_PROCESSED_PATH = DATA_PATH + 'processed/'

    train = pd.read_csv(DATA_INTERIM_PATH + 'train_p.csv', usecols=['tokens', 'hyperpartisan'])
    val = pd.read_csv(DATA_INTERIM_PATH + 'val_p.csv', usecols=['tokens', 'hyperpartisan'])

    # Set breaks for train
    breaks = [0, 150000, 300000, 450000, train.shape[0]]

    # Empty list for train tagged docs
    tagged_documents_train = []

    tqdm.pandas()

    # Iterate over breaks and create tagged documents for training set
    for i in range(0,len(breaks)-1):
        tagged_documents_train_temp = train.iloc[breaks[i]:breaks[i+1]].progress_apply(
            lambda x: TaggedDocument(words=ast.literal_eval(x['tokens']), tags=[x['hyperpartisan']]),
            axis=1
        )

        with open(DATA_PROCESSED_PATH + "tagged_documents_train{}.txt".format(i), "wb") as internal_filename:
            pickle.dump(tagged_documents_train_temp, internal_filename)

        tagged_documents_train.extend(tagged_documents_train_temp)

    # Create tagged documents for test set
    tagged_documents_test = val.progress_apply(
        lambda x: TaggedDocument(words=ast.literal_eval(x['tokens']), tags=[x['hyperpartisan']]),
        axis=1
    )

    with open(DATA_PROCESSED_PATH + "tagged_documents_test.txt", "wb") as internal_filename:
        pickle.dump(tagged_documents_test, internal_filename)

    return tagged_documents_train, tagged_documents_test


# Create doc2vec model using tagged documents
def create_docvec_model(DATA_PROCESSED_PATH, MODEL_PATH):

    # Set logging info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Create a generator for training corpus
    tagged_documents_train = TrainCorpus(DATA_PROCESSED_PATH)

    # Fit and save doc2vec model
    cores = multiprocessing.cpu_count()
    doc2vec_model = Doc2Vec(tagged_documents_train, workers=cores, vector_size=300, epochs=16, dm=1, min_count=3)
    doc2vec_model.save(MODEL_PATH + 'd2v300')

    return doc2vec_model


# Infer document vectors from doc2vec model
def infer_docvecs(doc2vec_model, tagged_documents_train, tagged_documents_test, DATA_PROCESSED_PATH):

    train_targets, train_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=16)) for doc in tagged_documents_train])

    test_targets, test_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=16)) for doc in tagged_documents_test])

    # Save inferred vectors
    with open(DATA_PROCESSED_PATH + "inferred_doc_vectors_train.txt", "wb") as internal_filename:
        pickle.dump(train_regressors, internal_filename)

    with open(DATA_PROCESSED_PATH + "inferred_doc_vectors_test.txt", "wb") as internal_filename:
        pickle.dump(test_regressors, internal_filename)

# Load document vectors
def load_docvecs(DATA_PROCESSED_PATH):

    with open(DATA_PROCESSED_PATH + "inferred_doc_vectors_train.txt", "rb") as internal_filename:
        X_train_doc2vec = pickle.load(internal_filename)

    with open(DATA_PROCESSED_PATH + "inferred_doc_vectors_test.txt", "rb") as internal_filename:
        X_test_doc2vec = pickle.load(internal_filename)

    return X_train_doc2vec, X_test_doc2vec


# Generator for streaming training documents
class TrainCorpus(object):

    def __init__(self, filepath):

        # Set filepath, infer filenames
        self.filepath = filepath
        self.filenames = glob(self.filepath + "tagged_documents_train*").sort()

    def __iter__(self):

        # Iterate over training files and yield
        for filename in self.filenames:
            with open(filename, "rb") as internal_filename:
                f = pickle.load(internal_filename)
                for i, line in enumerate(f):

                    if (i % 10000 == 0):
                        logging.info("read {0} docs".format(i))
                    yield line