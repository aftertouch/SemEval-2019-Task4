"""
@author: Jonathan
"""

from math import floor
import pandas as pd


# Sample the datasets
def sample_data(df, sample_size, train_or_val, save=False, DATA_PATH=None):

    # Create empty sample dataframe
    df_sample = pd.DataFrame()

    # Get bias column distribution
    bias = df['bias'].value_counts(normalize=True)

    # Group dataframe by bias
    gps = df.groupby('bias')

    # Iterate over bias groups
    for group in gps.groups.keys():
        # Get number of rows from df to sample
        num_rows = floor(sample_size * bias[group])

        # Randomly sample group's rows and append to sample df
        sample = df.loc[gps.groups[group], :].sample(n=num_rows, random_state=1)
        df_sample = df_sample.append(sample)

    # Reset index
    df_sample.reset_index(drop=True, inplace=True)

    # Save dataframe
    if save:
        DATA_PROCESSED_PATH = DATA_PATH + 'processed/'
        df_sample.to_csv(DATA_PROCESSED_PATH + train_or_val + str(sample_size) + '.csv')

    return df_sample