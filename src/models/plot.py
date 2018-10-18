"""
@author: Jonathan
"""

import numpy as np
import itertools
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

# COLOR PALETTES - From Vapeplot package
COOL_LIST = ["#FF6AD5", "#C774E8", "#AD8CFF", "#8795E8", "#94D0FF"]
COOL_PURPLE = '#AD8CFF'
COOL_LIST_MONO = ['#FFFFFF', COOL_PURPLE]
COOL_PAL = matplotlib.colors.ListedColormap(COOL_LIST)
COOL_PAL_CONT_MONO = matplotlib.colors.LinearSegmentedColormap.from_list('cool', COOL_LIST_MONO)

# Function to create Latent Semantic Analysis plot
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_LSA(X_train, test_labels, plot=True, title='LSA'):

    fig = plt.figure(figsize=(16, 16))

    lsa = TruncatedSVD(n_components=2)
    lsa.fit(X_train)
    lsa_scores = lsa.transform(X_train)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue']
    if plot:
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=COOL_PAL)
        red_patch = mpatches.Patch(color='#FF6AD5', label='Non-HP')
        green_patch = mpatches.Patch(color='#94D0FF', label='HP')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})
        plt.title(title, fontsize=30)

    plt.show(block=False)

    return lsa_scores

# Function to create Confusion Matrix plot
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_confusion_matrix(y_test, y_predicted_counts, normalize=False, title='Confusion Matrix'):

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
        plt.text(j, i, format(cm[i,j], fmt), horizontalalignment="center",
                 color="black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)
    plt.show(block=True)

# Function to plot largest coefficients of features of logistic regression model
# @credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def plot_important_words(importance, title):

    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]

    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', color=COOL_PURPLE, alpha=0.5)
    plt.title('Non-HP', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', color=COOL_PURPLE, alpha=0.5)
    plt.title('HP', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(title, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show(block=True)