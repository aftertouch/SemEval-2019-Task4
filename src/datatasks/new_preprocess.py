import pandas as pd
import numpy as np
import re
from datatasks.contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer

def preprocess(DATA_INTERIM_PATH):

    train = pd.read_csv(DATA_INTERIM_PATH + 'train_c.csv')
    val = pd.read_csv(DATA_INTERIM_PATH + 'val_c.csv')

    for df in [train, val]:
        df['preprocessed_text'] = df.apply(add_title_to_article_text, axis=1)
        df['preprocessed_text'] = df['preprocessed_text'].apply(replace_html_stuff)
        df['preprocessed_text'] = df['preprocessed_text'].apply(expand_contractions)

    print('Saving')
    train.to_csv(DATA_INTERIM_PATH + 'train_p.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val_p.csv', index=False)

def tokenize(df):

    tokenizer = ToktokTokenizer()

    df['tokens'] = df['preprocessed_text'].str.lower().apply(lambda x: tokenizer.tokenize(x))

    return df

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

    # Remove all ?s
    text = text.replace('?', '')
    
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text