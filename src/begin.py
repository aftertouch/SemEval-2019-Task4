"""
@author: Jonathan
"""

from datatasks.parse_xml import parse_provided
import datatasks.sample_data
from models.models import create_tfidf, run_models
from sklearn.externals import joblib
import os
import glob

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
        print('Done')

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

    # Create models and report results
    model_list = ['nb', 'lr']

    # TFIDF
    X_train, X_test, y_train, y_test = create_tfidf(train_path, val_path)
    best_tfidf_model, best_tfidf_model_type = run_models(model_list, X_train, X_test, y_train, y_test, random_state=42)

    # Serialize and save best model
    MODEL_PATH = '../model/'
    joblib.dump(best_tfidf_model, MODEL_PATH + 'tfidf_' + best_tfidf_model_type + '.joblib')

if __name__ == '__main__':
    main()