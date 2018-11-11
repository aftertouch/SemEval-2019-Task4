"""
@author: Jonathan

"""

import ast
import pickle

import pandas as pd
from tldextract import extract


# Main function for generating custom features
def generate_custom_features(data_interim_path, util_path):
    # Load datasets
    train = pd.read_csv(data_interim_path + 'train_reduced.csv')
    val = pd.read_csv(data_interim_path + 'val.csv')

    for df in [train, val]:
        # Get domain biases for linked articles
        df = domain_bias(df, util_path)

        # GROUND TRUTH TASKS - cannot be run on evaluation sets
        df = tld(df)

    # Save datasets
    print('Saving')
    train.to_csv(data_interim_path + 'train_c.csv', index=False)
    val.to_csv(data_interim_path + 'val_c.csv', index=False)


# Get bias for each linked domain in article
def domain_bias(df, util_path):
    # Load bias mapping
    pickle_in = open(util_path + "tld.pickle", "rb")
    tlds = pickle.load(pickle_in)

    # Create HP_links_count, nonHP_links_count, and unknown_links_count for each external link
    df[['HP_links_count', 'nonHP_links_count', 'unknown_links_count']] = df.loc[:, 'external_links'].apply(
        external_links_bias, args=(tlds,))

    return df


# Sub-function of domain_bias to be applied to dataframe
def external_links_bias(external_links, tlds):
    # Initialize link numbers
    hp_links = 0
    nonhp_links = 0
    unknown_links = 0

    # Iterate over external links list for each article
    for url in ast.literal_eval(external_links):

        # Get TLD of URL
        tld = extract(url)[1]

        # Check if TLD bias is known. Map appropriately.
        try:
            bias = tlds[tld]
            if bias in ['left', 'right']:
                hp_links += 1
            elif bias in ['leftcenter', 'right-center', 'center']:
                nonhp_links += 1
        except:
            unknown_links += 1

    return pd.Series([hp_links, nonhp_links, unknown_links])


# Extract article publisher (TLD)
def tld(df):
    df['domain'] = df.loc[:, 'url'].apply(lambda x: extract(x)[1])

    return df
