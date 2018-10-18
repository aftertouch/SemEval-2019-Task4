"""
@author: Jonathan

Credits:
http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
"""

import ast
from tldextract import extract
import pandas as pd
import pickle

# Main function for generating custom features
def generate_custom_features(DATA_PATH, UTIL_PATH):

    DATA_INTERIM_PATH = DATA_PATH + 'interim/'

    # Load datasets
    train = pd.read_csv(DATA_INTERIM_PATH + 'train.csv')
    val = pd.read_csv(DATA_INTERIM_PATH + 'val.csv')

    for df in [train, val]:

        # Get numbers of links
        df = strip_chars(df)

        # Extract article publisher (TLD) from URL
        df = tld(df)

    # TRAINING SET ONLY TASKS

    # Get biases for each link per article
    train = domain_bias(train, UTIL_PATH)

    # Save datasets
    print('Saving')
    train.to_csv(DATA_INTERIM_PATH + 'train_c.csv', index=False)
    val.to_csv(DATA_INTERIM_PATH + 'val_c.csv', index=False)

# Strip unwanted characters
def strip_chars(df):

    # Newlines and link signifiers (&#160)
    df['article_text'] = df['article_text'].str.replace('\n','').str.replace('&#160', ' ')

    return df


# Get bias for each linked domain in article
def domain_bias(train, path):

    # Load bias mapping
    pickle_in = open(path + "tld.pickle","rb")
    tlds = pickle.load(pickle_in)

    # Create HP_links_count, nonHP_links_count, and unknown_links_count for each external link
    train[['HP_links_count', 'nonHP_links_count', 'unknown_links_count']] = train.loc[:,'external_links'].apply(external_links_bias, args=(tlds,))

    return train

# Sub-function of domain_bias to be applied to dataframe
def external_links_bias(external_links, tlds) :
    
    # Initialize link numbers
    HP_links = 0
    nonHP_links = 0
    unknown_links = 0

    # Iterate over external links list for each article
    for url in ast.literal_eval(external_links):

        # Get TLD of URL
        tld = extract(url)[1]

        # Check if TLD bias is known. Map appropriately.
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

    df['domain'] = df.loc[:,'url'].apply(lambda x: extract(x)[1])

    return df

# @credit: http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.
    
    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.
    
    @param text: Text whose language want to be detected
    @type text: str
    
    @return: Most scored language guessed
    @rtype: str
    """

    ratios = _calculate_languages_ratios(text)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language


# @credit: http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
def _calculate_languages_ratios(text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}
    
    @param text: Text whose language want to be detected
    @type text: str
    
    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}

    '''
    nltk.wordpunct_tokenize() splits all punctuations into separate tokens
    
    >>> wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
    ['That', "'", 's', 'thirty', 'minutes', 'away', '.', 'I', "'", 'll', 'be', 'there', 'in', 'ten', '.']
    '''

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios