"""
@author: Jonathan
"""

import pandas as pd
from math import floor
from datatasks.preprocess import normalize_corpus
from nltk.tokenize import RegexpTokenizer

def sample_preprocess_data(DATA_PATH, n_samples, sample_size, train_or_val, save=False):

    # Get the processed folder paths
    DATA_INTERIM_PATH = DATA_PATH + 'interim/'
    DATA_PROCESSED_PATH = DATA_PATH + 'processed/'

    # Load dataframe to sample
    if train_or_val == 'train':
        df = pd.read_csv(DATA_INTERIM_PATH + 'train.csv')
    elif train_or_val == 'val':
        df = pd.read_csv(DATA_INTERIM_PATH + 'val.csv')
    
    for i in range(n_samples):
        
        # Create empty sample dataframe
        df_sample = pd.DataFrame()
        
        # Get bias column distribution
        bias = df['bias'].value_counts(normalize=True)
        gps = df.groupby('bias')
        
        # Iterate over bias groups
        for group in gps.groups.keys():
            
            # Get number of rows from df to sample
            nrows = floor(sample_size*bias[group])
            
            # Randomly sample group's rows and append to sample df
            sample = df.loc[gps.groups[group],:].sample(n=nrows, random_state=1)
            df_sample = df_sample.append(sample)
        
        # Preprocess article text and append to sample dataframe
        df_sample['preprocessed_text'] = normalize_corpus(df_sample['article_text'])
        
        # Reset index
        df_sample.reset_index(drop=True, inplace=True)
        
        # Save dataframe
        df_sample.to_csv(DATA_PROCESSED_PATH + train_or_val + str(sample_size) + '_' + str(i) + '.csv', index=False)