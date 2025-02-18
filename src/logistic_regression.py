from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#using logistic regression as an algorithm, optimizing with GridSearchCV

def log_regression(x_train, y_train, x_test):
    #tuning hyperparameters
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'class_weight': ['balanced']
         }
    ]
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg_cv = GridSearchCV(logreg, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)

    #training the model
    best_logreg = logreg_cv.fit(x_train, y_train)

    print(f"Best parameters: {best_logreg.best_params_}")
    print(f"Best model: {best_logreg.best_estimator_}")
    return best_logreg.best_estimator_

#evaluating the performance of the model
def evaluate_model(model, x_test ,y_true):
    y_predict = model.predict(x_test)
    report = classification_report(y_true, y_predict)
    return y_predict, report