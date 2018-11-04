"""
@author: Jonathan

Credits:
http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
"""

import pandas as pd
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

# Remove articles which should not be in the training set
def remove_articles(DATA_INTERIM_PATH):

    # Read training data
    train = pd.read_csv(DATA_INTERIM_PATH + 'train.csv')

    # Drop duplicate articles containing identical URLs
    train = train.drop_duplicates(subset='url', keep='first')

    # Detect language for each article using stopword ratios
    train['language'] = train['article_text'].apply(detect_language)

    # Drop all nonenglish articles from dataframe
    # Of note, many English articles are flagged as nonenglish, however,
    # most of them are non-content articles like lists of scores and are
    # almost entirely noise
    train = train[train['language'] == 'english']
    train.drop('language', axis=1, inplace=True)
    train.reset_index(inplace=True)

    # Save reduced training set
    train.to_csv(DATA_INTERIM_PATH + 'train_reduced.csv', index=False)

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