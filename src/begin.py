"""
@author: Jonathan
"""

import os

import pandas as pd
from sklearn.externals import joblib

import models.EDA
import models.plot
from datatasks.custom_features import generate_custom_features
from datatasks.preprocess import preprocess
from datatasks.parse_xml import parse_provided
from datatasks.remove_articles import remove_articles
from datatasks.sample_data import sample_data
from models.feature_spaces import create_tfidf
from models.models import run_models, calculate_baseline
from models.pipeline import make_features_pipeline


def main():
    DATA_PATH = '../data/'
    UTIL_PATH = '../util/'

    # Create data subfolders if they do not exist
    data_folders = ['interim', 'processed', 'external']
    try:
        for folder in data_folders:
            if not os.path.isdir(DATA_PATH + folder + '/'):
                os.mkdir(DATA_PATH + folder + '/')
    except:
        print('Data folder not found. Did you create it in the root directory?')

    # Check for existence of parsed XML files. If they do not exist, create them
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    # Check for parsed XML data. If not, parse.
    if not os.path.exists(DATA_INTERIM_PATH + 'train.csv') or not os.path.exists(DATA_INTERIM_PATH + 'val.csv'):
        print('Parsing XML')
        parse_provided(DATA_PATH)

    # Delete duplicate and nonenglish articles from training data
    if not os.path.exists(DATA_INTERIM_PATH + 'train_reduced.csv'):
        print('Deleting duplicate/nonenglish articles')
        remove_articles(DATA_INTERIM_PATH)

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

    # Optionally randomly sample datasets
    sample = True
    sample_size_train = 100000
    sample_size_test = 10000
    if sample:
        print('Sampling data')
        train = sample_data(train, sample_size_train, 'train')
        val = sample_data(val, sample_size_test, 'val')

    # Train test split
    X_train = train.drop('hyperpartisan', axis=1)
    y_train = train['hyperpartisan']
    X_test = val.drop('hyperpartisan', axis=1)
    y_test = val['hyperpartisan']

    # Create Feature set for text

    # Create Feature Union
    print('Creating feature union')
    tfidf_transformer = create_tfidf()
    feats = make_features_pipeline(tfidf_transformer, 'tokens')

    # Calculate Baseline
    print('Calculating baseline')
    baseline = calculate_baseline(train)

    # Evaluate Models
    model_list = ['lr']
    print('Evaluating models')
    best_tfidf_model, best_tfidf_model_type, best_tfidf_model_predictions = run_models(feats, model_list, X_train,
                                                                                       X_test, y_train, y_test)

    # Confusion Matrix
    models.plot.plot_confusion_matrix(y_test, best_tfidf_model_predictions)

    # Serialize and save best model
    MODEL_PATH = '../model/'
    joblib.dump(best_tfidf_model, MODEL_PATH + 'tfidf_' + best_tfidf_model_type + '.joblib')


if __name__ == '__main__':
    main()
