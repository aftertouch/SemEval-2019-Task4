"""
@author: Jonathan
"""

import xml.etree.cElementTree as et
import pandas as pd
from tqdm import tqdm

# Main function for parsing competition data
def parse_provided(DATA_PATH):

    # Get interim path
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    # Parse GT and Article files for train and val
    gt_list = parse_provided_gt(DATA_PATH)
    text_list = parse_provided_text(DATA_PATH)

    # Merge GT and Article dataframes
    train = text_list[0].merge(gt_list[0], on='id')
    val = text_list[1].merge(gt_list[1], on='id')

    print('Saving')

    # Save
    train.to_csv(DATA_INTERIM_PATH + 'train.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val.csv', index=False)

# Parse ground truth XML files
def parse_provided_gt(DATA_PATH):

    # Get paths to data folders
    DATA_RAW_PATH = DATA_PATH + 'raw/'

    # Get paths and column names for data files
    try:
        gt_train_path = DATA_RAW_PATH + 'ground-truth-training-20180831.xml'
        gt_val_path = DATA_RAW_PATH + 'ground-truth-validation-20180831.xml'
    except:
        print('Data file(s) not found.')

    # Column names
    gt_cols = ['id', 'hyperpartisan', 'bias', 'url', 'labeled-by']
    gt_list = []

    print('Parsing ground truth')

    # Parse ground truth files
    for gt_path in [gt_train_path, gt_val_path]:

        # Find all articles in XML tree
        tree = et.parse(gt_path)
        root = tree.getroot()
        articles = root.findall('.//article')
        #

        # Get data for columns
        xml_data = [[article.get('id'), article.get('hyperpartisan'), article.get('bias'), article.get('url'), article.get('labeled-by')] 
                    for article in tqdm(articles)]

        # Create dataframes
        gt_df = pd.DataFrame(xml_data, columns=gt_cols)

        gt_list.append(gt_df)

    return gt_list

# Parse article text XML files
def parse_provided_text(DATA_PATH):

    # Get paths to data folders
    DATA_RAW_PATH = DATA_PATH + 'raw/'

    # Check if data files exist
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

# Subfunction of parse_provided_text, contains XML parsing logic
def parse_text(path):

    # Define columns
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

    # Get article text and links for each article
    articles_text_list = []
    external_links_lists = []
    internal_links_lists = []

    # Iterate over articles
    for article in tqdm(articles):

        # Get article title
        text_string = article.get('title') + ' '

        # Append article text to article title
        for t in article.itertext():
            text_string += t
        articles_text_list.append(text_string)

        # Create empty dictionaries for links
        external_links_dict = {}
        internal_links_dict = {}

        p_list = article.findall('p')

        # Iterate over paragraphs, find all links, then iterate over links per paragraph
        for p in p_list:
            links = p.findall('a')
            for a in links:

                # Get link and link's text
                link = a.get('href')
                text = a.text

                # Determine if link is internal or external
                if a.get('type') == 'external' and bool(link):
                    if link[0] == '/':
                        internal_links_dict[link] = text
                    else:
                        external_links_dict[link] = text
                            
        external_links_lists.append(external_links_dict)
        internal_links_lists.append(internal_links_dict)

    # Add article text and link columns to dataframe
    text_df['article_text'] = articles_text_list
    text_df['external_links'] = external_links_lists
    text_df['internal_links'] = internal_links_lists

    return text_df