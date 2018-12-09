"""
@author: Jonathan
"""

import re
import numpy as np
import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer
from tqdm import tqdm
from datatasks.contractions import CONTRACTION_MAP

# Main preprocessing function
def preprocess(data_interim_path):

    # Load data
    print('Loading')
    train = pd.read_csv(data_interim_path + 'train_c.csv')
    val = pd.read_csv(data_interim_path + 'val_c.csv')

    tqdm.pandas()

    # Normalize Text for each article
    print('Normalizing Text')
    for df in [train, val]:
        df['preprocessed_text'] = df.progress_apply(add_title_to_article_text, axis=1)
        df['preprocessed_text'] = df['preprocessed_text'].progress_apply(normalize_text)

    # Replace publisher signatures for training set
    print('Replacing publisher signatures')
    train = train.progress_apply(replace_publisher_signatures, axis=1)

    # Tokenize and remove stopwords
    print('Tokenizing and removing stopwords')
    for df in [train, val]:
        df[['preprocessed_text', 'tokens']] = df.loc[:, 'preprocessed_text'].progress_apply(remove_stopwords_and_tokenize)

    print('Saving')
    train.to_csv(data_interim_path + 'train_p.csv', index=False)
    val.to_csv(data_interim_path + 'val_p.csv', index=False)


def normalize_text(text):

    text = text.lower()
    text = replace_html_stuff(text)
    text = expand_contractions(text)
    text = mask_numbers(text)
    text = remove_special_characters(text)
    text = replace_newlines(text)
    text = remove_extra_whitespace(text)

    return text

# Replace semiautonomously generated list of common noise phreases
def replace_publisher_signatures(article):
    removal_dict = {
        'foxbusiness': ['continue reading below', 'opens a new window', 'has no position in any of the stocks mentioned',
                        'the motley fool has a disclosure policy', 'dow jones newswires', 'copyright marketwatch inc'],
        'inthesetimes': ['your email your name recipients email comma separated message captcha',
                         'like what youve read subscribe to in these times magazine or make a taxdeductible donation to fund this reporting'],
        'truthdig': ['associated press', 'ap', 'read more', 'reuters'],
        'washingtonblade': ['washington', 'blade'],
        'feministing': ['header image'],
        'motherjones': ['mother jones'],
        'thedailywire': ['daily Wire'],
        'factcheck': ['the factcheck wire'],
        'pri': ['pris'],
        'abqjournal': ['new mexico', 'santa fe', 'albuquerque', ' nm '],
        'newsline': ['upi',
                     'fusion media or anyone involved with fusion media will not accept any liability for loss or damage as a result of reliance on the information including data quotes charts and buysell signals contained within this website please be fully informed regarding the risks and costs associated with trading the financial markets it is one of the riskiest investment forms possible'],
        'reuters': ['our standards the thomson reuters trust principles'],
        'thedailybeast': [
            'start and finish your day with the top stories from the daily beast a speedy smart summary of all the news you need to know and nothing you do not',
            'the daily beast']
    }

    # Check each article's domain. If it's in the removal dictionary, remove each noise phrase from article text
    if article['domain'] in removal_dict.keys():
        for phrase in removal_dict[article['domain']]:
            article['preprocessed_text'] = re.sub(phrase, '', article['preprocessed_text'])

    return article

# Add article title as string to article text
def add_title_to_article_text(article):
    if article['title'] is not np.nan:
        return str(article['title']) + ' ' + str(article['article_text'])
    else:
        return str(article['article_text'])

# Replace HTML artifacts
def replace_html_stuff(text):
    # Remove all between left and right brackets
    text = re.sub(r'(?s)&lt.*?&gt;', '', text)
    text = re.sub(r'(?s)lt;.*?gt;', '', text)

    # Remove all html tags
    text = re.sub(r'&(.{1,5});', '', text)
    text = re.sub(r'amp;', '', text)

    # Remove !function
    text = re.sub(r'(?s)!function.*?(\?|\")\);', '', text)

    # Remove everything between parenthesis
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove all URLs
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

    # Remove all emails
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove all twitter usernames
    text = re.sub(r'@\S*\s?', '', text)

    return text

# Expand contractions.
# Source: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
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

# Replace numbers with masking
def mask_numbers(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)

    return text

# Remove special characters
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9#\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# Remove extra whitespace
def remove_extra_whitespace(text):
    text = re.sub(' +', ' ', text)
    return text

# Replace newlines
def replace_newlines(text):
    text = re.sub(r'\n', ' ', text)

    return text

# Tokenize, then remove list of corpus specific stopwords
def remove_stopwords_and_tokenize(text):
    tokenizer = ToktokTokenizer()
    stopword_list = ['advertisement', 'via', 'image', 'source', 'click', 'video', 'editing', 'investingcom', '___', 'gmt', 'copyright', 'reporting', 'et', 'reprint', 'featured', 'embedded', 'journal',
             'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'mon',
             'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
             'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return pd.Series([filtered_text, filtered_tokens])
