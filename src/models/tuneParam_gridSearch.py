"""
@author: Negar
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# change CV to 4 or 3
# grid search to find the best parameters for the best accuracy with 10-fold cross validation on training set
# def grid_search(model, param_grid, X, y, scorer='accuracy', n_jobs=3, cv=10):
def grid_search(model, param_grid, X, y, scorer='accuracy', n_jobs=5, cv=3):
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs,
                        scoring=scorer,
                        pre_dispatch=n_jobs)

    # keep fitted model for future use -- not using it now TODO
    fit_model = grid.fit(X, y)
    return grid.best_params_, grid.best_score_


# Tune parameters for models and print best parameters
# @credit: got parameters for Multinamial Naive Bayes and Gradient Boosting and dt from https://github.com/Morgan243/CMSC516-SE-T8/blob/master/SemEvalEight/modeling/task1/task1_bag_of_words.py
def tune_param(model_list, X_train, y_train):
    model_dict = {
        'nb': 'Multinomial Naive Bayes',
        'lr': 'LogisticRegression',
        'gb': 'GradientBoostingClassifier',
        'dt': 'DecisionTreeClassifier',
        'rf': 'RandomForestClassifier',
        'svc': 'SVC'
    }

    for model_type in model_list:
        # grid search on Multinamial Naive Bayes
        if model_type == 'nb':
            nb_best_params, nb_best_score = grid_search(MultinomialNB(),
                                                        param_grid=dict(alpha=[10 ** a for a in range(-3, 4, 1)]),
                                                        X=X_train, y=y_train)
            print(nb_best_params)
            print(nb_best_score)

        # grid search on Logistic regression
        elif model_type == 'lr':
            lr_best_params, lr_best_score = grid_search(LogisticRegression(),
                                                        param_grid=dict(C=[0.001, 0.01, 0.1, 1, 10, 30, 100, 1000],
                                                                        solver=('newton-cg', 'sag', 'saga'),
                                                                        class_weight=('balanced', 'None')),
                                                        X=X_train, y=y_train)
            print(lr_best_params)
            print(lr_best_score)

        # grid search on Gradient Boosting
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
        elif model_type == 'dt':
            dt_best_params, dt_best_score = grid_search(DecisionTreeClassifier(),
                                                        param_grid=dict(
                                                            criterion=['gini', 'entropy'],
                                                            max_depth=[None] + list(range(24, 30, 3)),
                                                            max_leaf_nodes=[None] + list(range(13, 20, 2)),
                                                            min_samples_leaf=list(range(2, 5, 2)),
                                                            min_samples_split=list(range(2, 5, 2))), X=X_train,
                                                        y=y_train)
            print(dt_best_params)
            print(dt_best_score)
        elif model_type == 'rf':
            rf_best_params, rf_best_score = grid_search(RandomForestClassifier(),
                                                        param_grid=dict(bootstrap=[True, False],
                                                                        max_depth=[3, 20, 35, None],
                                                                        max_features=['auto', 'sqrt'],
                                                                        min_samples_leaf=[1, 2, 4],
                                                                        min_samples_split=[2, 5, 10],
                                                                        n_estimators=[25, 200, 230]), X=X_train,
                                                        y=y_train)
            print(rf_best_params)
            print(rf_best_score)
        elif model_type == 'svc':
            svc_best_params, svc_best_score = grid_search(SVC(),
                                                          param_grid=dict(C=[0.01, 0.1, 1, 10],
                                                                          gamma=[0.001, 0.01, 0.1, 1],
                                                                          kernel=['linear', 'rbf', 'sigmoid']),
                                                          X=X_train, y=y_train)
            print(svc_best_params)
            print(svc_best_score)

        else:
            raise ValueError("No model type provided")

            # return nb_best_params, gb_best_params
