"""
@author: Jonathan
"""

import itertools

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import mplcursors

# COLOR PALETTES - From Vapeplot package
COOL_LIST = ["#FF6AD5", "#C774E8", "#AD8CFF", "#8795E8", "#94D0FF"]
COOL_PURPLE = '#AD8CFF'
COOL_PINK_BLUE = matplotlib.colors.ListedColormap(['#94D0FF', '#FF6AD5'])
COOL_LIST_MONO = ['#FFFFFF', COOL_PURPLE]
COOL_PAL = matplotlib.colors.ListedColormap(COOL_LIST)
COOL_PAL_CONT_MONO = matplotlib.colors.LinearSegmentedColormap.from_list('cool', COOL_LIST_MONO)


# Function to create Latent Semantic Analysis plot
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_LSA(X_train, test_labels, plot=True, title='LSA', save_path=None):
    fig = plt.figure(figsize=(12, 12))

    lsa = TruncatedSVD(n_components=2)
    lsa.fit(X_train)
    lsa_scores = lsa.transform(X_train)
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels, cmap=COOL_PINK_BLUE)
        red_patch = mpatches.Patch(color='#94D0FF', label='Non-HP')
        green_patch = mpatches.Patch(color='#FF6AD5', label='HP')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})
        plt.title(title, fontsize=30)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show(block=False)

    return lsa_scores


# Function to create Confusion Matrix plot
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_confusion_matrix(y_test, y_predicted_counts, normalize=False, title='Confusion Matrix', save_path=None):
    fig = plt.figure(figsize=(8, 8))

    cm = confusion_matrix(y_test, y_predicted_counts)
    classes = ['Non-HP', 'HP']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=COOL_PAL_CONT_MONO)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show(block=False)


# Function to plot largest coefficients of features of logistic regression model
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_important_words(importance, title):
    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]

    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', color=COOL_PURPLE, alpha=0.5)
    plt.title('Non-HP', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', color=COOL_PURPLE, alpha=0.5)
    plt.title('HP', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(title, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show(block=True)


def plot_correct_per_publisher(ydf, title='Percent Correct per Publisher', save_path=None):
    incorrect = ydf[~(ydf['true'] == ydf['predicted'])]
    preds_dict = {}
    for domain in ydf['domain'].value_counts().keys():
        total_articles = ydf[ydf['domain'] == domain].shape[0]
        correct_predictions = total_articles - incorrect[incorrect['domain'] == domain].shape[0]
        hyperpartisan = ydf[ydf['domain'] == domain]['true'].value_counts().keys()[0]
        preds_dict[domain] = {
            'total_articles': total_articles,
            'correct_predictions': correct_predictions,
            'pct_correct': correct_predictions / total_articles,
            'hyperpartisan': hyperpartisan
        }
    domains = list(preds_dict.keys())
    article_counts = np.array([preds_dict[domain]['total_articles'] for domain in domains])
    pct_correct = np.array([preds_dict[domain]['pct_correct'] for domain in domains])
    hyperpartisan = np.array([preds_dict[domain]['hyperpartisan'] for domain in domains])

    fig = plt.figure(figsize=(12, 12))
    scatter = plt.scatter(x=article_counts, y=pct_correct, c=hyperpartisan, s=100, cmap=COOL_PINK_BLUE)
    plt.title(title, fontsize=30)
    plt.xlabel('Number of Articles', fontsize=20)
    plt.ylabel('Percent Correct', fontsize=20)
    red_patch = mpatches.Patch(color='#94D0FF', label='Non-HP')
    green_patch = mpatches.Patch(color='#FF6AD5', label='HP')
    plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

    if save_path is not None:
        plt.savefig(save_path)

    mplcursors.cursor(scatter, hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(domains[sel.target.index]))

    plt.show()