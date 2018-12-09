"""
@author: Jonathan
"""

import os

import pandas as pd
from sklearn.externals import joblib

from datatasks.custom_features import generate_custom_features
from datatasks.preprocess import preprocess
from datatasks.parse_xml import parse_provided
from datatasks.remove_articles import remove_articles
from models.models import run_models, calculate_baseline
from models.feature_spaces import create_tfidf
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
        preprocess(DATA_INTERIM_PATH)

    # Load training and validation data
    print('Loading data')
    train = pd.read_csv(DATA_INTERIM_PATH + 'train_p.csv')
    val = pd.read_csv(DATA_INTERIM_PATH + 'val_p.csv')

    # Split into X and y
    X_train = train.drop('hyperpartisan', axis=1)
    y_train = train['hyperpartisan']
    X_test = val.drop('hyperpartisan', axis=1)
    y_test = val['hyperpartisan']

    # Create TF-IDF Features
    print('Creating TF-IDF Features')
    tfidf_vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf(fit=True, X_train=X_train, X_test=X_test)

    # Create Document Vectors Features

    # Calculate Baseline
    print('Calculating baseline')
    calculate_baseline(train)

    # Evaluate Models
    models_list = ['lr']
    best_model, best_model_type, best_model_predictions, clf = run_models('tfidf', models_list,
                                                                          X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Confusion Matrix
    plot_confusion_matrix(y_test, best_model_predictions)

    # Plot correct per publisher
    ydf = create_ydf(val, best_model_predictions)
    plot_correct_per_publisher(ydf)

    # Serialize and save best model
    joblib.dump(best_model, MODEL_PATH + best_model_type + '.joblib')


if __name__ == '__main__':
    main()
