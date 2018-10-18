"""
@author: Jonathan

Credits:
https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#@credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# Gets largest coefficients for each class for a logistic regression model
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    
    return classes

# Convenience function for investigating the number of instances of a string in HP vs nonHP, train and val
def compare_text(text, train, val, text_type='article_text'):
    print('Train HP: ' + str(train[(train[text_type].str.contains(text)) & (train['hyperpartisan']==True)].shape[0]))
    print('Train nonHP: ' + str(train[(train[text_type].str.contains(text)) & (train['hyperpartisan']==False)].shape[0]))
    print('Val HP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan']==True)].shape[0]))
    print('Val nonHP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan']==False)].shape[0]))

# Extract and inspect words that appear in some (large) percentage of a publisher's articles
def find_domain_signatures(df, domain, thresh=0.6, ngram_range=(1,1), random_sample=True, sample_size=3000, should_print=True):
    
    # Get all articles by provided publisher
    df = df[df['domain'] == domain]

    # Sample subset of 3,000 if random_sample=True
    if random_sample and df.shape[0] > sample_size:
        df = df.sample(n=sample_size)
    
    # Create counter vectorizer
    vectorizer = CountVectorizer(stop_words='english',ngram_range=ngram_range, binary=True)
    X = vectorizer.fit_transform(df['article_text'])
    
    # Create sums vector with entry for each word
    sums = X.sum(axis=0)
    
    # Find all indices which exceed the given percentage for inclusion
    max_indices = np.argwhere(sums >= thresh*df.shape[0])
    
    signature_tokens = {}

    # Store feature names of vectorizer for convenience
    feature_names = vectorizer.get_feature_names()
    
    # Find words at indices and append to list
    for index in max_indices:
        feature = feature_names[index[1]]
        signature_tokens[feature] = round(sums[0,index[1]]/df.shape[0], 3)
    
    # Print and return
    if should_print:
        print(signature_tokens)
    return signature_tokens

# Filter common signature words
def filter_common_signatures(signatures, vectorizer, X, thresh=0.5):
    #vectorizer = CountVectorizer(stop_words='english',binary=True)
    #X = vectorizer.fit_transform(df['article_text'])

    # Store feature names for convenience
    features = vectorizer.get_feature_names()

    # Dictionary of dictionaries to ultimately be returned
    diffs = {}

    # Dictionary of word/sum pairs for convenience (to avoid summing common words multiple times)
    sums = {}

    # Iterate over domains
    for domain in signatures:

        # Dictionary of signature word/difference pairs
        diff = {}

        # Iterate over signature words for domain
        for key in signatures[domain].keys():

            # Check if word has already been summed. If so, return that
            if key in sums:
                percent = sums[key]/X.shape[0]
            else:
                # Get index of word
                index = features.index(key)

                # Sum word's column
                key_sum = X[:,index].sum()

                # Get percentage
                percent = key_sum/X.shape[0]

                # Store in convenience dictionary
                sums[key] = key_sum

            # Find difference between domain and corpus word usage
            diff_percent = round(float(signatures[domain][key]) - percent, 3)

            # If difference is larger than thresh, store it
            if diff_percent > thresh:
                diff[key] = diff_percent

        # Add domain to diffs dictionary
        diffs[domain] = diff

    return diffs