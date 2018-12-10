"""
@author: Jonathan
"""

import os

import pandas as pd
from sklearn.externals import joblib

from datatasks.custom_features import generate_custom_features
from datatasks.preprocess import preprocess, create_tagged_documents
from datatasks.parse_xml import parse_provided
from datatasks.remove_articles import remove_articles
from models.models import run_models, calculate_baseline
from models.feature_spaces import create_tfidf, create_docvec_model, infer_docvecs, load_docvecs
from models.plot import plot_confusion_matrix, plot_correct_per_publisher
from models.EDA import create_ydf


def main():

    # Store data paths for convenience
    DATA_PATH = '../data/'
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'
    DATA_PROCESSED_PATH = DATA_PATH + 'processed/'
    MODEL_PATH = '../model/'
    UTIL_PATH = '../util/'

    # Create data subfolders if they do not exist
    data_folders = ['interim', 'processed', 'external']
    try:
        for folder in data_folders:
            if not os.path.isdir(DATA_PATH + folder + '/'):
                os.mkdir(DATA_PATH + folder + '/')
    except:
        print('Data folder not found. Did you create it in the root directory?')

    # Check for parsed XML data. If not, parse.
    if not os.path.exists(DATA_INTERIM_PATH + 'train.csv') or not os.path.exists(DATA_INTERIM_PATH + 'val.csv'):
        print('Parsing XML')
        parse_provided(DATA_PATH)

    # Delete duplicate articles from training data
    if not os.path.exists(DATA_INTERIM_PATH + 'train_reduced.csv'):
        print('Deleting duplicate articles')
        remove_articles(DATA_INTERIM_PATH, remove_nonenglish=False)

    # Generate custom features
    if not os.path.exists(DATA_INTERIM_PATH + 'train_c.csv') or not os.path.exists(DATA_INTERIM_PATH + 'val_c.csv'):
        print('Generating custom features')
        generate_custom_features(DATA_INTERIM_PATH, UTIL_PATH)

    # Preprocess text
    if not os.path.exists(DATA_INTERIM_PATH + 'train_p.csv') or not os.path.exists(DATA_INTERIM_PATH + 'val_p.csv'):
        print('Preprocessing Text')
        train_tokens, val_tokens = preprocess(DATA_PATH)
        print('Creating Tagged Documents')
        create_tagged_documents(train_tokens, 'train', DATA_PROCESSED_PATH)
        create_tagged_documents(val_tokens, 'val', DATA_PROCESSED_PATH)

    if not os.path.exists(MODEL_PATH + 'd2v300'):
        print('Creating doc2vec model')
        create_docvec_model(DATA_PROCESSED_PATH, MODEL_PATH)

    # Create Document Vectors
    if not os.path.exists(DATA_PROCESSED_PATH + 'inferred_doc_vectors_train.txt'):
        print('Inferring Document Vectors')
        infer_docvecs(DATA_PROCESSED_PATH, MODEL_PATH)

    # Create TF-IDF Features
    print('Creating TF-IDF Features')
    tfidf_vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf(fit=True, DATA_INTERIM_PATH=DATA_INTERIM_PATH)

    # Load inferred document vectors
    print('Loading document vectors')
    X_train_doc2vec, X_test_doc2vec = load_docvecs(DATA_PROCESSED_PATH)

    # Load targets
    y_train = pd.read_csv(DATA_INTERIM_PATH + 'train_p.csv', usecols=['hyperpartisan'])['hyperpartisan']
    y_test = pd.read_csv(DATA_INTERIM_PATH + 'val_p.csv', usecols=['hyperpartisan'])['hyperpartisan']

    # Calculate Baseline
    print('Calculating baseline')
    calculate_baseline(y_train)

    # Initialize best model
    best_model = {
        'model': None,
        'type': '',
        'accuracy': 0,
        'predictions': None
    }

    # Evaluate models
    models_list = ['lr', 'sgd', 'rf']
    best_model = run_models('tfidf', models_list, best_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    best_model = run_models('doc2vec', models_list, best_model, X_train_doc2vec, X_test_doc2vec, y_train, y_test)

    # Print best results
    print('Best model is {} with an accuracy score of {:.4f}'.format(best_model['type'], best_model['accuracy']))

    # Confusion Matrix
    plot_confusion_matrix(y_test, best_model['predictions'])

    # Plot correct per publisher
    ydf = create_ydf(DATA_INTERIM_PATH, best_model['predictions'])
    plot_correct_per_publisher(ydf)

    # Serialize and save best model
    joblib.dump(best_model['model'], MODEL_PATH + best_model['type'] + '.joblib')


if __name__ == '__main__':
    main()
