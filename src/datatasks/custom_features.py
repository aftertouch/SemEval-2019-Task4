"""
@author: Jonathan
"""

import ast
import tldextract
import pandas as pd
import pickle

def generate_custom_features(DATA_PATH):

    DATA_INTERIM_PATH = DATA_PATH + 'interim/'
    DATA_EXTERNAL_PATH = DATA_PATH + 'external/'

    # Load datasets
    train = pd.read_csv(DATA_INTERIM_PATH + 'train.csv')
    val = pd.read_csv(DATA_INTERIM_PATH + 'val.csv')

    for df in [train, val]:

        # Get numbers of links
        df = strip_chars(df)

    # Training set only tasks
    train = tld(train)
    train = domain_bias(train, DATA_EXTERNAL_PATH)

    # Save datasets
    print('Saving')
    train.to_csv(DATA_INTERIM_PATH + 'train_c.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val_c.csv', index=False)

# Strip unwanted characters
def strip_chars(df):
    df['article_text'] = df['article_text'].str.replace('\n','').str.replace('&#160', ' ')

    return df


# Count number of internal and external links
def domain_bias(train, path):

    pickle_in = open(path + "tld.pickle","rb")
    tlds = pickle.load(pickle_in)
    train[['HP_links_count', 'nonHP_links_count', 'unknown_links_count']] = train.loc[:,'external_links'].apply(external_links_bias, args=(tlds,))

def external_links_bias(external_links, tlds) :
    
    HP_links = 0
    nonHP_links = 0
    unknown_links = 0

    for url in ast.literal_eval(external_links):
        tld = extract(url)[1]
        try:
            bias = tlds[tld]
            if bias in ['left', 'right']:
                HP_links += 1
            elif bias in ['leftcenter', 'right-center', 'center']:
                nonHP_links += 1
        except:
            unknown_links += 1
            
    return pd.Series([HP_links, nonHP_links, unknown_links])


# Extract article publisher (TLD)
def tld(df):

    df['domain'] = df.loc[:,'url'].apply(lambda x: tldextract.extract(x)[1])

    return df