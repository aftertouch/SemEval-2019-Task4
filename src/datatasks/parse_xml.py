import xml.etree.cElementTree as et
import pandas as pd
from tqdm import tqdm

def parse_provided(DATA_PATH):

    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    gt_list = parse_provided_gt(DATA_PATH)
    text_list = parse_provided_text(DATA_PATH)

    train = text_list[0].merge(gt_list[0], on='id')
    val = text_list[1].merge(gt_list[1], on='id')

    print('Saving')

    train.to_csv(DATA_INTERIM_PATH + 'train.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val.csv', index=False)

def parse_provided_gt(DATA_PATH):

    # Get paths to data folders
    DATA_RAW_PATH = DATA_PATH + 'raw/'

    # Get paths and column names for data files
    try:
        gt_train_path = DATA_RAW_PATH + 'ground-truth-training-20180831.xml'
        gt_val_path = DATA_RAW_PATH + 'ground-truth-validation-20180831.xml'
    except:
        print('Data file(s) not found.')

    gt_cols = ['id', 'hyperpartisan', 'bias', 'url', 'labeled-by']
    gt_list = []

    print('Parsing ground truth')

    # Parse ground truth files
    for gt_path in [gt_train_path, gt_val_path]:

        # Find all articles in XML tree
        tree = et.parse(gt_path)
        root = tree.getroot()
        articles = root.findall('.//article')

        # Get data for columns
        xml_data = [[article.get('id'), article.get('hyperpartisan'), article.get('bias'), article.get('url'), article.get('labeled-by')] 
                    for article in tqdm(articles)]

        # Create dataframes
        gt_df = pd.DataFrame(xml_data, columns=gt_cols)

        gt_list.append(gt_df)

    return gt_list

def parse_provided_text(DATA_PATH):

    # Get paths to data folders
    DATA_RAW_PATH = DATA_PATH + 'raw/'

    try:
        text_train_path = DATA_RAW_PATH + 'articles-training-20180831.xml'
        text_val_path = DATA_RAW_PATH + 'articles-validation-20180831.xml'
    except:
        print('Data file(s) not found.')

    text_list = []


    print('Parsing Article Text')
    # Parse article text files
    for text_path in [text_train_path, text_val_path]:

        text_df = parse_text(text_path)

        text_list.append(text_df)

    return text_list

def parse_text(path):

    text_cols = ['id', 'published-at', 'title']

    #try:
    tree = et.parse(path)
    root = tree.getroot()
    articles = root.findall('.//article')

    # Get data for columns
    xml_data = [[article.get('id'), article.get('published-at'), article.get('title')] 
                for article in tqdm(articles)]

    # Create dataframes
    text_df = pd.DataFrame(xml_data, columns=text_cols)

    # Get article text for each article
    article_text_list = []
    for article in tqdm(articles):
        text_string = article.get('title') + ' '
        for t in article.itertext():
            text_string += t
        article_text_list.append(text_string)

    # Add article text column to dataframe
    text_df['article_text'] = article_text_list

    return text_df


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