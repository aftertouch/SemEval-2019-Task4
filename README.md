# SemEval 2019: Task 4 - Hyperpartisan News Detection

## Jonathan Miller and Negar Adyaniyazdi, VCU CMSC516, Fall 2018

This Github repository houses all code for our entry to the [2019 SemEval Task 4 Challenge](https://pan.webis.de/semeval19/semeval19-web/).
Here we aim to use statistical learning and Natural Language Processing techniques in order to
distinguish Hyperpartisan news from non-Hyperpartisan news. Hyperpartisan news is a growing problem
and a subproblem of the rapid spread of fake news throughout social media, which has shown power to influence the public negatively.
We would like to be able to automatically identify Hyperpartisan news in order to potentially
stop viral spread. The dataset provided for the challenge consists of 1,000,000 news articles
from various publishers, labeled Hyperpartisan or not, and evenly balanced. (500,000 articles each)

## Our solution

We plan to use NLP methods and various machine learning classifiers to distinguish Hyperpartisan articles from those
which are non-Hyperpartisan. The dataset provided contains a large amount of noise, so our solution will include
removing non-political articles, removing publisher 'signatures', removing URLs/emails/twitter usernames,
and preprocessing according to a fairly standard text preprocessing pipeline. We will then generate problem-specific
features using the data, then explore combinations of various NLP feature sets such as bag of words, TF-IDF bag of words,
and word embeddings in combination with custom features to train multiple classifiers.

The primary evaluation metric for this task is accuracy on a HP/non-HP-balanced evaluation set.
Additionally, Precision, Recall, and F1 scores are reported for the true class.

## Expected Input and Output

This program only needs the source code and competition data to run. The Anaconda distribution of Python, as 
well as an internet connection are required for setup as well.

Users can expect evaluation metric output for all currently implemented classifiers and feature sets, as well 
as LSA plots of these feature spaces and a confusion matrix for the best classifier as output.

## Getting Started

See [INSTALL.md](https://github.com/aftertouch/SemEval-2019-Task4/blob/master/INSTALL.md) 
for comprehensive instructions on setting up this project.

## Team Roles

Jonathan Miller - Application building, exploratory analysis, feature creation

Negar Adyaniyazdi - Model building, tuning, and selection, feature selection