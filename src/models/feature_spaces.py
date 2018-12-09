from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
from glob import glob

import logging


def create_tfidf(fit=False, X_train=None, X_test=None):
    tfidf_vectorizer = TfidfVectorizer()

    if fit:
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        return tfidf_vectorizer, X_train_tfidf, X_test_tfidf

    return tfidf_vectorizer


def create_tagged_documents():
    pass


def create_docvec_model(DATA_PROCESSED_PATH, MODEL_PATH):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    tagged_documents_train = TrainCorpus(DATA_PROCESSED_PATH)
    doc2vec_model = Doc2Vec(tagged_documents_train, workers=8, vector_size=300, epochs=16, dm=1, min_count=3)
    doc2vec_model.save(MODEL_PATH + 'd2v300')


class TrainCorpus(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.filenames = glob(self.filepath + "tagged_documents_train*").sort()

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, "rb") as internal_filename:
                f = pickle.load(internal_filename)
                for i, line in enumerate(f):

                    if (i % 10000 == 0):
                        logging.info("read {0} docs".format(i))
                    yield line