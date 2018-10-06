import xml.etree.cElementTree as et
import pandas as pd

def parse_provided_gt():

    # Get paths to data folders
    DATA_PATH = '../data/'
    DATA_RAW_PATH = DATA_PATH + 'raw/'
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    # Get paths and column names for data files
    try:
        gt_train_path = DATA_RAW_PATH + 'ground-truth-training-20180831.xml'
        gt_val_path = DATA_RAW_PATH + 'ground-truth-validation-20180831.xml'
    except:
        print('Data file(s) not found.')
    gt_cols = ['id', 'hyperpartisan', 'bias', 'url', 'labeled-by']

    # Parse ground truth files
    for gt_path in [gt_train_path, gt_val_path]:

        # Find all articles in XML tree
        tree = et.parse(gt_path)
        root = tree.getroot()
        articles = root.findall('.//article')

        # Get data for columns
        xml_data = [[article.get('id'), article.get('hyperpartisan'), article.get('bias'), article.get('url'), article.get('labeled-by')] 
                    for article in articles]

        # Create dataframes
        gt_df = pd.DataFrame(xml_data, columns=gt_cols)

        if gt_path == 'gt_train_path':
            gt_df.to_csv(DATA_INTERIM_PATH + 'gt_train.csv', index=False)
        elif gt_path == 'gt_val_path':
            gt_df.to_csv(DATA_INTERIM_PATH + 'gt_val.csv', index=False)

def parse_provided_text(file=None):

    # Get paths to data folders
    DATA_PATH = '../data/'
    DATA_RAW_PATH = DATA_PATH + 'raw/'
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    try:
        text_train_path = DATA_RAW_PATH + 'articles-training-20180831.xml'
        text_val_path = DATA_RAW_PATH + 'articles-validation-20180831.xml'
    except:
        print('Data file(s) not found.')


    # Parse article text files
    for text_path in [text_train_path, text_val_path]:

        text_df = parse_text(text_path)

        # Save as csv
        if text_path == 'text_train_path':
            text_df.to_csv(DATA_INTERIM_PATH + 'text_train.csv', index=False)
        elif text_path == 'text_val_path':
            text_df.to_csv(DATA_INTERIM_PATH + 'text_val.csv', index=False)

def parse_text(path):
    
    text_cols = ['id', 'published-at', 'title']

    #try:
    tree = et.parse(path)
    root = tree.getroot()
    articles = root.findall('.//article')

    # Get data for columns
    xml_data = [[article.get('id'), article.get('published-at'), article.get('title')] 
                for article in articles]

    # Create dataframes
    text_df = pd.DataFrame(xml_data, columns=text_cols)

    # Get article text for each article
    article_text_list = []
    for article in articles:
        text_string = article.get('title') + ' '
        for t in article.itertext():
            text_string += t
        article_text_list.append(text_string)

    # Add article text column to dataframe
    text_df['article_text'] = article_text_list

    return text_df
    #except:
        #print('Error: File either not XML or improperly formatted.')


def merge_gt_text():

    # Get interim data path
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    try:
        gt_train = pd.read_csv(DATA_INTERIM_PATH + 'gt_train.csv', dtype={'id' : str})
        gt_val = pd.read_csv(DATA_INTERIM_PATH + 'gt_val.csv', dtype={'id' : str})
    except:
        print('Ground truth files not found. Have you run parse_xml.parse_gt()?')

    try:
        text_train = pd.read_csv(DATA_INTERIM_PATH + 'text_train.csv', dtype={'id' : str})
        text_val = pd.read_csv(DATA_INTERIM_PATH + 'text_val.csv', dtype={'id' : str})
    except:
        print('Ground truth files not found. Have you run parse_xml.parse_gt()?')

    # Merge text and ground truth dataframes into single train/val dataframes
    train = text_train.merge(gt_train, on='id')
    val = text_val.merge(gt_val, on='id')

    # Save new data files as csv
    train.to_csv(DATA_INTERIM_PATH + 'train.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val.csv', index=False)