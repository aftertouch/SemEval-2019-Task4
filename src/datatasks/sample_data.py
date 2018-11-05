"""
@author: Jonathan
"""

import pandas as pd
from math import floor

# Sample the datasets
def sample_data(df, sample_size, train_or_val, random_state=1, save=False):
        
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
        sample = df.loc[gps.groups[group],:].sample(n=nrows, random_state=random_state)
        df_sample = df_sample.append(sample)
    
    # Reset index
    df_sample.reset_index(drop=True, inplace=True)
    
    # Save dataframe
    if save:
        df_sample.to_csv(DATA_PROCESSED_PATH + train_or_val + str(sample_size) + '_' + str(i) + '.csv', index=False)

    return df_sample