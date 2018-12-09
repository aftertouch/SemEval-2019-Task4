"""
@author: Negar
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Run all models and print evaluation metrics for all classifiers and also find and print best model
def run_models(features_name, model_list, X_train, X_test, y_train, y_test, random_state=42):
    # Set random state
    random_state = random_state

    # Convenience translation dictionary for printing
    model_dict = {
        'lr': 'Logistic Regression',
        'sgd' : 'Stochastic Gradient Descent',
        'rf' : 'Random Forest',
        'dnn' : 'Dense Neural Network'
    }

    hyperparams_dict = {
        'tfidf': {
            'lr': {
                'C': 1000.0,
                'class_weight': 'None',
                'solver': 'newton-cg'
            }
        }
    }

    # Initialize best model variables
    best_model = ''
    best_model_type = ''
    best_model_predictions = None
    best_accuracy = 0

    # Iterate over model_list
    for model_type in model_list:

        # Logistic Regression fit model
        if model_type == 'lr':
            clf = LogisticRegression(C=hyperparams_dict[features_name][model_type]['C'],
                                     class_weight=hyperparams_dict[features_name][model_type]['class_weight'],
                                     solver=hyperparams_dict[features_name][model_type]['solver'],
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_type = model_type
            best_model_predictions = predicted

    # Print best results
    print('Best model is {} with an accuracy score of {:.4f}'.format(model_dict[best_model_type], best_accuracy))

    # Return best model and type
    return best_model, best_model_type, best_model_predictions, clf


# Evaluate models. Print classification report with precision, recall, f1, print accuracy, and return accuracy
def evaluate_model(y_test, predicted):
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