"""
@author: Jonathan
"""

from datatasks.parse_xml import parse_provided
from datatasks.custom_features import generate_custom_features
import datatasks.sample_data
from models.models import create_tfidf, run_models, calculate_baseline
from sklearn.externals import joblib
import os
import glob
import models.plot
import models.EDA
import pandas as pd

def main():

    DATA_PATH = '../data/'

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

    # Generate custom features
    print('Generating custom features')
    generate_custom_features(DATA_PATH)

    DATA_PROCESSED_PATH = DATA_PATH + 'processed/'

    # Sample the datasets and preprocess
    filepath = DATA_PROCESSED_PATH + '*.csv'
    if not glob.glob(filepath):
        print('Sampling and preprocessing training data')
        datatasks.sample_data.sample_preprocess_data(DATA_PATH, 1, 10000, 'train', save=True)
        print('Sampling and preprocessing validation data')
        datatasks.sample_data.sample_preprocess_data(DATA_PATH, 1, 2500, 'val', save=True)

    # Get training and test data
    train_path = glob.glob(DATA_PROCESSED_PATH + 'train*.csv')[0]
    val_path = glob.glob(DATA_PROCESSED_PATH + 'val*.csv')[0]

    # Load training and validation data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    # Calculate Baseline
    baseline = models.models.calculate_baseline(train)

    # TFIDF
    X_train, X_test, y_train, y_test, tfidf_vectorizer = create_tfidf(train, val)

    # TFIDF LSA
    models.plot.plot_LSA(X_train, y_train, title='TF-IDF LSA')

    # Evaluate Models
    model_list = ['nb', 'lr']
    best_tfidf_model, best_tfidf_model_type, best_tfidf_model_predictions = run_models(model_list, X_train, X_test, y_train, y_test, random_state=42)

    # Confusion Matrix
    models.plot.plot_confusion_matrix(y_test, best_tfidf_model_predictions)

    # Important Features
    importance = models.EDA.get_most_important_features(tfidf_vectorizer, best_tfidf_model, 10)
    models.plot.plot_important_words(importance, "Most important words for relevance")

    # Serialize and save best model
    MODEL_PATH = '../model/'
    joblib.dump(best_tfidf_model, MODEL_PATH + 'tfidf_' + best_tfidf_model_type + '.joblib')

if __name__ == '__main__':
    main()