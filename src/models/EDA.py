"""
@author: Jonathan
"""

import pandas as pd
import numpy as np
import eli5


# Convenience function for investigating the number of instances of a string in HP vs nonHP, train and val
def compare_text(text, train, val, text_type='article_text'):
    print('Train HP: ' + str(train[(train[text_type].str.contains(text)) & (train['hyperpartisan'] == True)].shape[0]))
    print('Train nonHP: ' + str(
        train[(train[text_type].str.contains(text)) & (train['hyperpartisan'] == False)].shape[0]))
    print('Val HP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan'] == True)].shape[0]))
    print('Val nonHP: ' + str(val[(val[text_type].str.contains(text)) & (val['hyperpartisan'] == False)].shape[0]))

# Return context windows around a term as a dataframe
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


# Examines the top weighted features of a classifier using eli5. Used to find publisher signatures
def examine_top_weights(df, clf, vec, CONTEXT_THRESH, top_n_domains, n_features):

    # Get top feature weights
    weights = eli5.explain_weights_df(clf, vec=vec)

    # Take the absolute value to distinguish between positive and negative predictors
    weights['weight'] = np.absolute(weights['weight'])

    # Sort from largest to smallest
    weights.sort_values('weight', inplace=True, ascending=False);

    # Get the percentage of representation of the top n domains
    base_props = df['domain'].value_counts(normalize=True)[0:top_n_domains]

    # Iterate over top n features
    for i in range(0, n_features):

        # Get feature weight
        row = weights.iloc[i]

        # Get feature name
        feature = row['feature']

        # Find domains whose articles contain top features and get the proportion of articles per domain which have it
        props = df[(df['preprocessed_text'].str.contains(" {} ".format(feature), regex=False)) &
                      (df['domain'].isin(base_props.keys()))]['domain'].value_counts(normalize=True)

        # Iterate over domains
        for key in props.keys():

            # If it shows up more than expected
            if props[key] > 2 * base_props[key]:

                # Examine forward and backward context of term, print common contexts
                context = find_common_context_windows(feature, key, df, 'preprocessed_text', 1)
                forward_context = context[0].value_counts(normalize=True)
                if forward_context[0] > CONTEXT_THRESH:
                    print(feature.upper())
                    print(key,
                          "- Observed: {:.3f}, Expected: {:.3f}, Difference: {:.3f}".format(props[key], base_props[key],
                                                                                            props[key] / base_props[key]))
                    print(forward_context[forward_context > CONTEXT_THRESH])
                    print('\n')
                backward_context = context[2].value_counts(normalize=True)
                if backward_context[0] > CONTEXT_THRESH:
                    print(feature.upper())
                    print(key,
                          "- Observed: {:.3f}, Expected: {:.3f}, Difference: {:.3f}".format(props[key], base_props[key],
                                                                                            props[key] / base_props[key]))
                    print(backward_context[backward_context > CONTEXT_THRESH])
                    print('\n')

        print('\n')


# Creates a convenience dataframe for comparing true and predicted labels for validation set
def create_ydf(DATA_INTERIM_PATH, preds):
    val = pd.read_csv(DATA_INTERIM_PATH + 'val_p.csv', usecols=['hyperpartisan', 'domain'])
    ydf = pd.DataFrame(list(zip(val['hyperpartisan'], preds, val['domain'])),
                       columns=['true', 'predicted', 'domain'], index=val.index)

    return ydf
