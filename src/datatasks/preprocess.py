import re

import numpy as np
import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer

from datatasks.contractions import CONTRACTION_MAP


def preprocess(data_interim_path):
    train = pd.read_csv(data_interim_path + 'train_c.csv')
    val = pd.read_csv(data_interim_path + 'val_c.csv')

    for df in [train, val]:
        df['preprocessed_text'] = df.apply(add_title_to_article_text, axis=1)
        df['preprocessed_text'] = df['preprocessed_text'].apply(normalize_text)
        df['tokens'] = df['preprocessed_text'].apply(tokenize)

    print('Saving')
    train.to_csv(data_interim_path + 'train_p.csv', index=False)
    val.to_csv(data_interim_path + 'val_p.csv', index=False)


def normalize_text(text):
    text = replace_html_stuff(text)
    text = expand_contractions(text)
    text = mask_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = text.lower()

    return text


def add_title_to_article_text(article):
    if article['title'] is not np.nan:
        return str(article['title']) + ' \n ' + str(article['article_text'])
    else:
        return str(article['article_text'])


def replace_html_stuff(text):
    # Remove all between left and right brackets
    text = re.sub(r'(?s)&lt.*?&gt;', '', text)
    text = re.sub(r'(?s)lt;.*?gt;', '', text)

    # Remove all html tags
    text = re.sub(r'&(.{1,5});', '', text)
    text = re.sub(r'amp;', '', text)

    # Remove !function
    text = re.sub(r'(?s)!function.*?(\?|\")\);', '', text)

    # Remove all URLs
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

    # Remove all emails
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove all twitter usernames
    text = re.sub(r'@\S*\s?', '', text)

    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def mask_numbers(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)

    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_extra_whitespace(text):
    text = re.sub(' +', ' ', text)
    return text


def tokenize(text):
    tokenizer = ToktokTokenizer()

    tokens = tokenizer.tokenize(text)

    return tokens
