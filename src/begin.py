from datatasks.parse_xml import parse_provided
import datatasks.sample_data
from models.models import create_train_test_tfidf, run_models
import os

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

    # Sample the datasets and preprocess
    print('Sampling and preprocessing training data')
    datatasks.sample_data.sample_preprocess_data(DATA_PATH, 1, 100, 'train', save=True)
    print('Sampling and preprocessing validation data')
    datatasks.sample_data.sample_preprocess_data(DATA_PATH, 1, 20, 'val', save=True)

    # Create models and report results
    model_list = ['nb', 'lr', 'gb']

    DATA_PROCESSED_PATH = DATA_PATH + 'processed/'

    #TFIDF
    X_train, X_test, y_train, y_test = create_train_test_tfidf(DATA_PROCESSED_PATH + 'train100_0.csv', DATA_PROCESSED_PATH + 'val20_0.csv')
    run_models(model_list, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()