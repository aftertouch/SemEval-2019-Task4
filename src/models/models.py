"""
@author: Negar
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Run all models and print evaluation metrics for all classifiers and also find and print best model
def run_models(features_name, model_list, best_model, X_train, X_test, y_train, y_test, random_state=42):
    # Set random state
    random_state = random_state

    # Convenience translation dictionary for printing
    model_dict = {
        'lr': 'Logistic Regression',
        'sgd': 'Stochastic Gradient Descent',
        'rf': 'Random Forest',
        'dnn': 'Dense Neural Network'
    }

    # Dictionary of pre-determined hyperparameters for models
    hyperparams_dict = {
        'tfidf': {
            'lr': {
                'C': 30.0,
                'class_weight': 'None',
                'solver': 'newton-cg'
            },
            'sgd': {
                'tol': 1e-3,
                'max_iter': 1000,
                'penalty': 'l1'
            },
            'rf': {
                'bootstrap': False,
                'n_estimators': 200,
                'max_depth': 35,
                'max_features': 'sqrt',
                'min_samples_leaf': 1,
                'min_samples_split': 10,
            }
        },
        'doc2vec': {
            'lr': {
                'C': 0.01,
                'class_weight': 'balanced',
                'solver': 'sag'
            },
            'sgd': {
                'tol': 1e-3,
                'max_iter': 1000,
                'penalty': 'l1'
            },
            'rf': {
                'bootstrap': True,
                'n_estimators': 230,
                'max_depth': 35,
                'max_features': 'auto',
                'min_samples_leaf': 4,
                'min_samples_split': 10,
            }
        }
    }

    # Iterate over model_list
    for model_type in model_list:

        # Logistic Regression fit model
        if model_type == 'lr':
            clf = LogisticRegression(C=hyperparams_dict[features_name][model_type]['C'],
                                     class_weight=hyperparams_dict[features_name][model_type]['class_weight'],
                                     solver=hyperparams_dict[features_name][model_type]['solver'],
                                     n_jobs=-1, random_state=random_state)

        elif model_type == 'sgd':
            clf = SGDClassifier(tol=hyperparams_dict[features_name][model_type]['tol'],
                                max_iter=hyperparams_dict[features_name][model_type]['max_iter'],
                                penalty=hyperparams_dict[features_name][model_type]['penalty'],
                                n_jobs=-1, random_state=random_state)

        elif model_type == 'rf':
            clf = RandomForestClassifier(max_features=hyperparams_dict[features_name][model_type]['max_features'],
                                         min_samples_leaf=hyperparams_dict[features_name][model_type]['min_samples_leaf'],
                                         n_estimators=hyperparams_dict[features_name][model_type]['n_estimators'],
                                         bootstrap=hyperparams_dict[features_name][model_type]['bootstrap'],
                                         min_samples_split=hyperparams_dict[features_name][model_type]['min_samples_split'],
                                         max_depth=hyperparams_dict[features_name][model_type]['max_depth'],
                                         n_jobs=-1, random_state=random_state)
        else:
            raise ValueError("No model type provided")

        # Fit classifier
        print(model_dict[model_type])
        clf.fit(X_train, y_train)

        # predictions and evaluations
        predicted = clf.predict(X_test)
        accuracy = evaluate_model(predicted, y_test)

        # Update best performing model if necessary
        if accuracy > best_model['accuracy']:
            best_model['accuracy'] = accuracy
            best_model['model'] = clf
            best_model['type'] = model_type
            best_model['predictions'] = predicted

    # Return best model and type
    return best_model


# Evaluate models. Print classification report with precision, recall, f1, print accuracy, and return accuracy
def evaluate_model(y_test, predicted):
    print(classification_report(y_test, predicted))
    accuracy = accuracy_score(y_test, predicted)
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy


# Calculate baseline accuracy.
def calculate_baseline(y_train):
    # Get series counts of training data response. First item in value_counts will be
    # the majority class. Normalize returns accuracy as a percentage.
    baseline = y_train.value_counts(normalize=True)[0]
    print('Majority baseline accuracy is {:.4f}'.format(baseline))