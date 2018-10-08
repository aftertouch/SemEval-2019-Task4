"""
@author: Negar
"""

import pandas
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


###
#. Feature sets
###

# TF-IDF
def create_tfidf(trainFilePath, testFilePath):

    # Load training dataset
    train = pandas.read_csv(trainFilePath)

    calculate_baseline(train)

    # Create bag of words
    vectorizer = CountVectorizer()
    text = train.preprocessed_text.tolist()
    bag_of_words = vectorizer.fit(text)
    bag_of_words = vectorizer.transform(text)

    # Create TF-IDF from bag of words
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(bag_of_words)

    # Store response variable
    y_train = train.hyperpartisan.tolist()

    # Load test dataset
    test = pandas.read_csv(testFilePath)

    # Transform test set 
    testdata = test.preprocessed_text.tolist()
    testbag = vectorizer.transform(testdata)
    X_test_tfidf = tfidf_transformer.transform(testbag)

    # Store response variable
    y_test = test.hyperpartisan.tolist()

    return X_train_tfidf, X_test_tfidf, y_train, y_test

# Run all models. Takes a list of model types and data with a feature set
def run_models(model_list, X_train, X_test, y_train, y_test, random_state):

    # Set random state
    random_state = random_state
    
    # Convenience translation dictionary for printing
    model_dict ={
        'nb' : 'Multinomial Naive Bayes',
        'lr' : 'Logistic Regression',
        'gb' : 'Gradient Boosting Classifier'
    }

    # Initialize best model variables
    best_model = ''
    best_model_type = ''
    best_accuracy = 0
    
    # Iterate over list of model types
    for model_type in model_list:

        # Naive Bayes
        if model_type == 'nb':
            clf = MultinomialNB(alpha=0.1).fit(X_train, y_train)

        # Logistic Regression
        elif model_type == 'lr':
            clf = LogisticRegression(C=30.0, class_weight='None', solver='newton-cg')
            clf.fit(X_train, y_train)

        # Gradient Boosting
        elif model_type == 'gb':
            clf = GradientBoostingClassifier(learning_rate=0.7, max_depth=6, max_leaf_nodes=None, min_samples_leaf= 3, min_samples_split=2).fit(X_train, y_train)
        else:
            raise ValueError("No model type provided")   

        # Get predictions and evaluate     
        predicted = clf.predict(X_test)
        print(model_dict[model_type])
        accuracy = evaluate_model(predicted, y_test)

        # Update best performing model if necessary
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_type = model_type

    # Print best results
    print('Best model is {} with an accuracy score of {:.4f}'.format(model_dict[best_model_type], best_accuracy))

    # Return best model and type
    return best_model, best_model_type

# Evaluate models. Print classification report with precision, recall, f1, print accuracy, and return accuracy
def evaluate_model(predicted, y_test):
    print(classification_report(y_test, predicted))
    accuracy = accuracy_score(y_test, predicted)
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy

# Calculate baseline accuracy.
def calculate_baseline(train):

    # Get series counts of training data response. First item in value_counts will be
    # the majority class. Normalize returns accuracy as a percentage.
    baseline = train['hyperpartisan'].value_counts(normalize=True)[0]
    print('Majority baseline accuracy is {:.4f}'.format(baseline))