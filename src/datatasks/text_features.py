"""
@author: Jonathan
"""

import pandas as pd
import numpy as np


def make_text_features(data_interim_path, train, val):

    for df in [train, val]:

        # Count number of characters in article text
        df['tokens_count'] = df['tokens'].apply(lambda x: len(x.split()))

        # Compute ratio of numbers to total characters
        df['numbers_to_tokens_ratio'] = df['preprocessed_text'].str.count('##+') / df['token_count']

    train.to_csv(data_interim_path + 'train_p.csv', index=False)
    val.to_csv(data_interim_path + 'val_p.csv', index=False)