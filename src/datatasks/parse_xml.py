"""
@author: Jonathan
"""

import xml.etree.cElementTree as et

import pandas as pd
from tqdm import tqdm


# Main function for parsing competition data
def parse_provided(data_path):
    # Get interim path
    data_interim_path = data_path + 'interim/'

    # Parse GT and Article files for train and val
    gt_list = parse_provided_gt(data_path)
    text_list = parse_provided_text(data_path)

    # Merge GT and Article dataframes
    train = text_list[0].merge(gt_list[0], on='id')
    val = text_list[1].merge(gt_list[1], on='id')

    print('Saving')

    # Save
    train.to_csv(data_interim_path + 'train.csv', index=False)
    val.to_csv(data_interim_path + 'val.csv', index=False)


# Parse ground truth XML files
def parse_provided_gt(data_path):
    # Get paths to data folders
    data_raw_path = data_path + 'raw/'

    # Get paths and column names for data files
    try:
        gt_train_path = data_raw_path + 'ground-truth-training-bypublisher-20181122.xml'
        gt_val_path = data_raw_path + 'ground-truth-validation-bypublisher-20181122.xml'
    except:
        print('Data file(s) not found.')

    # Column names
    gt_cols = ['id', 'hyperpartisan', 'bias', 'url', 'labeled-by']
    gt_list = []

    print('Parsing ground truth')

    # Parse ground truth files
    for gt_path in [gt_train_path, gt_val_path]:

        gt_df = parse_ground_truth(gt_path, gt_cols)

        gt_list.append(gt_df)

    return gt_list


def parse_ground_truth(gt_path, gt_cols):
    # Find all articles in XML tree
    tree = et.parse(gt_path)
    root = tree.getroot()
    articles = root.findall('.//article')
    #

    # Get data for columns
    xml_data = [[article.get('id'), article.get('hyperpartisan'), article.get('bias'), article.get('url'),
                 article.get('labeled-by')]
                for article in tqdm(articles)]

    # Create dataframes
    gt_df = pd.DataFrame(xml_data, columns=gt_cols)

    return gt_df


# Parse article text XML files
def parse_provided_text(DATA_PATH):
    # Get paths to data folders
    DATA_RAW_PATH = DATA_PATH + 'raw/'

    # Check if data files exist
    try:
        text_train_path = DATA_RAW_PATH + 'articles-training-bypublisher-20181122.xml'
        text_val_path = DATA_RAW_PATH + 'articles-validation-bypublisher-20181122.xml'
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

    # try:
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
        text_string = ''

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
