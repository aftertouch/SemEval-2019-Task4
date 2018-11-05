"""
@author: Negar
"""

import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np


###
#. Feature sets
###

# bag of words - tfidf
def create_tfidf(train, val):

    X_train = train['preprocessed_text']
    y_train = train['hyperpartisan'].tolist()

    X_test = val['preprocessed_text']
    y_test = val['hyperpartisan'].tolist()

    # tfidf for training set
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    # tfidf for test set - no need to fit for test set
    X_test_tfidf = tfidf_vectorizer.transform(X_test)


    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer


# TF-IDF vectorizer and fit-transformed data
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

# Word embeddings
# Credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def create_avg_word_embeddings(DATA_EXTERNAL_PATH, train, val):
    
    # Load pre-trained word embeddings
    word2vec_path = DATA_EXTERNAL_PATH + 'GoogleNews-vectors-negative300.bin.gz'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    
    # Create tokenizer
    toktokenizer = ToktokTokenizer()
    
    # Create empty embeddings list
    embeddings_list = []
    
    for df in [train, val]:
        
        # Tokenize
        tokens = df['preprocessed_text'].str.lower().apply(lambda x: toktokenizer.tokenize(x))
        
        embeddings = df['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
        
        embeddings_list.append(list(embeddings))
        
    return embeddings_list

# Helper function for create_avg_word_embeddings
# Credit: https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged
    

# Run all models and print evaluation metrics for all classifiers and also find and print best model
def run_models(feats, model_list, X_train, X_test, y_train, y_test, random_state=42):

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
    best_model_predictions = None
    best_accuracy = 0
    
    # Iterate over model_list
    for model_type in model_list:

        # Naive Bayes fit model
        if model_type == 'nb':
            clf = MultinomialNB(alpha=0.1)

        # Logistic Regression fit model
        elif model_type == 'lr':
            clf = LogisticRegression(C=30.0, class_weight='None', solver='newton-cg')

        # Gradient Boosting fit model
        elif model_type == 'gb':
            clf = GradientBoostingClassifier(n_estimators=170, max_depth=5, learning_rate=0.5, min_samples_leaf=3, min_samples_split=4)
        else:
            raise ValueError("No model type provided")

        # Create pipeline with selected features and classifier
        pipeline = Pipeline([
            ('features', feats),
            ('classifier', clf)
        ]) 

        print(X_train.shape, y_train.shape)
        # Fit classifier
        pipeline.fit(X_train, y_train)

        # predictions and evaluations
        predicted = pipeline.predict(X_test)
        print(model_dict[model_type])
        accuracy = evaluate_model(predicted, y_test)

        # Update best performing model if necessary
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_type = model_type
            best_model_predictions = predicted

    # Print best results
    print('Best model is {} with an accuracy score of {:.4f}'.format(model_dict[best_model_type], best_accuracy))

    # Return best model and type
    return best_model, best_model_type, best_model_predictions

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

    return baseline