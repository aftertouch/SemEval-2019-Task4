"""
@author: Jonathan

Credits:
https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
"""

import pandas as pd
import numpy as np
import eli5
from sklearn.feature_extraction.text import CountVectorizer


# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# Gets largest coefficients for each class for a logistic regression model
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }

    return classes


# Convenience function for investigating the number of instances of a string in HP vs nonHP, train and val
def compare_text(text, train, val, text_type='article_text'):
    print('Train HP: ' + str(train[(train[text_type].str.contains(text)) & (train['hyperpartisan'] == True)].shape[0]))
    print('Train nonHP: ' + str(
        train[(train[text_type].str.contains(text)) & (train['hyperpartisan'] == False)].shape[0]))
    print('Val HP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan'] == True)].shape[0]))
    print('Val nonHP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan'] == False)].shape[0]))


# Extract and inspect words that appear in some (large) percentage of a publisher's articles
def find_domain_signatures(df, domain, thresh=0.6, ngram_range=(1, 1), random_sample=True, sample_size=3000,
                           should_print=True):
    # Get all articles by provided publisher
    df = df[df['domain'] == domain]

    # Sample subset of 3,000 if random_sample=True
    if random_sample and df.shape[0] > sample_size:
        df = df.sample(n=sample_size)

    # Create counter vectorizer
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range, binary=True)
    X = vectorizer.fit_transform(df['article_text'])

    # Create sums vector with entry for each word
    sums = X.sum(axis=0)

    # Find all indices which exceed the given percentage for inclusion
    max_indices = np.argwhere(sums >= thresh * df.shape[0])

    signature_tokens = {}

    # Store feature names of vectorizer for convenience
    feature_names = vectorizer.get_feature_names()

    # Find words at indices and append to list
    for index in max_indices:
        feature = feature_names[index[1]]
        signature_tokens[feature] = round(sums[0, index[1]] / df.shape[0], 3)

    # Print and return
    if should_print:
        print(signature_tokens)
    return signature_tokens


# Filter common signature words
def filter_common_signatures(signatures, vectorizer, X, thresh=0.5):
    # vectorizer = CountVectorizer(stop_words='english',binary=True)
    # X = vectorizer.fit_transform(df['article_text'])

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
                percent = sums[key] / X.shape[0]
            else:
                # Get index of word
                index = features.index(key)

                # Sum word's column
                key_sum = X[:, index].sum()

                # Get percentage
                percent = key_sum / X.shape[0]

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


def find_common_context_windows(term, domain, df, df_text_col, window):
    series = df[(df['preprocessed_text'].str.contains(' {} '.format(term))) &
                (df['domain'] == domain)].reset_index()[df_text_col]

    context = []

    for text in series:
        text_split = text.split(' ')

        indices = [i for i, x in enumerate(text_split) if x == term]

        for index in indices:
            context.append([catch(lambda: text_split[index + i]) for i in range(-window, window + 1)])

    context_df = pd.DataFrame(context)

    return context_df


def catch(func, handle=lambda e: e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return 'OOR'


def examine_top_weights(df, clf, vec, CONTEXT_THRESH, top_n_domains, n_features):
    weights = eli5.explain_weights_df(clf, vec=vec)
    weights['weight'] = np.absolute(weights['weight'])
    weights.sort_values('weight', inplace=True, ascending=False);
    base_props = df['domain'].value_counts(normalize=True)[0:top_n_domains]
    for i in range(0, n_features):  # len(features[0]['tops'][0])):
        row = weights.iloc[i]
        feature = row['feature']
        props = df[(df['preprocessed_text'].str.contains(" {} ".format(feature), regex=False)) &
                      (df['domain'].isin(base_props.keys()))]['domain'].value_counts(normalize=True)
        for key in props.keys():
            if props[key] > 2 * base_props[key]:

                # Examine context of term
                context = find_common_context_windows(feature, key, df, 'preprocessed_text', 1)
                forward_context = context[0].value_counts(normalize=True)
                if forward_context[0] > CONTEXT_THRESH:
                    print(feature.upper())
                    print(key,
                          "- Observed: {:.3f}, Expected: {:.3f}, Difference: {:.3f}".format(props[key], base_props[key],
                                                                                            props[key] / base_props[
                                                                                                key]))
                    print(forward_context[forward_context > CONTEXT_THRESH])
                    print('\n')
                backward_context = context[2].value_counts(normalize=True)
                if backward_context[0] > CONTEXT_THRESH:
                    print(feature.upper())
                    print(key,
                          "- Observed: {:.3f}, Expected: {:.3f}, Difference: {:.3f}".format(props[key], base_props[key],
                                                                                            props[key] / base_props[
                                                                                                key]))
                    print(backward_context[backward_context > CONTEXT_THRESH])
                    print('\n')

        print('\n')


def create_ydf(val, preds):
    ydf = pd.DataFrame(list(zip(val['hyperpartisan'], preds, val['preprocessed_text'], val['domain'])),
                       columns=['true', 'predicted', 'preprocessed_text', 'domain'], index=val.index)

    return ydf