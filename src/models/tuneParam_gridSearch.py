"""
@author: Negar
"""
import pandas
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def create_train_test_tfidf(trainFilePath):
    train = pandas.read_csv(trainFilePath)
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    text = train.preprocessed_text.tolist()
    bag_of_words = vectorizer.fit(text)
    bag_of_words = vectorizer.transform(text)
    # tf and tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(bag_of_words)
    y_train = train.hyperpartisan.tolist()
    # X_train_tfidf.shape

    return X_train_tfidf, y_train


def grid_search(model, param_grid, X, y, scorer='accuracy', n_jobs=3, cv=10):
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs,
                        scoring=scorer,
                        pre_dispatch=n_jobs)

    fit_m = grid.fit(X, y)
    return grid.best_params_, grid.best_score_


def tune_param(model_list, X_train, y_train):
    model_dict = {
        'nb': 'Multinomial Naive Bayes',
        'lr': 'LogisticRegression',
        'gb': 'GradientBoostingClassifier'
    }

    for model_type in model_list:
        if model_type == 'nb':
            nb_best_params, nb_best_score = grid_search(MultinomialNB(),
                                                        param_grid=dict(alpha=[10 ** a for a in range(-3, 4, 1)]),
                                                        X=X_train, y=y_train)
            print(nb_best_params)
            print(nb_best_score)
        elif model_type == 'lr':
            lr_best_params, lr_best_score = grid_search(LogisticRegression(),
                                                        param_grid=dict(C=[0.001, 0.01, 0.1, 1, 10, 30, 100, 1000],
                                                                        solver=('newton-cg', 'sag', 'saga'),
                                                                        class_weight=('balanced', 'None')),
                                                        X=X_train, y=y_train)
            print(lr_best_params)
            print(lr_best_score)
        elif model_type == 'gb':
            gb_best_params, gb_best_score = grid_search(GradientBoostingClassifier(),
                                                        param_grid=dict(
                                                            max_depth=list(range(3, 7, 1)),
                                                            max_leaf_nodes=[None],
                                                            min_samples_leaf=list(range(2, 4, 1)),
                                                            min_samples_split=list(range(2, 5, 2)),
                                                            learning_rate=np.arange(0.7, 1.5, 0.33)),
                                                        X=X_train, y=y_train)
            print(gb_best_params)
            print(gb_best_score)
        else:
            raise ValueError("No model type provided")

    return nb_best_params, gb_best_params
