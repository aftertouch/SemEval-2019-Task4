"""
@author: Jonathan
@credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
"""

import numpy as np
import itertools
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

COOL_LIST = ["#FF6AD5", "#C774E8", "#AD8CFF", "#8795E8", "#94D0FF"]
COOL_PAL = matplotlib.colors.ListedColormap(COOL_LIST)
COOL_PAL_CONT = matplotlib.colors.LinearSegmentedColormap.from_list('cool', COOL_LIST)

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

def plot_confusion_matrix(y_test, y_predicted_counts, normalize=False, title='Confusion Matrix'):

    fig = plt.figure(figsize=(10, 10))

    cm = confusion_matrix(y_test, y_predicted_counts)
    classes = ['Non-HP', 'HP']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=COOL_PAL_CONT)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), horizontalalignment="center",
                 color="white" if cm[i,j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)
    plt.show(block=True)