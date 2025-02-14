from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def log_regression(x_train, y_train, x_test):
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'class_weight': ['balanced']
         }
    ]
    logreg_0 = LogisticRegression(random_state=16, max_iter=1000)
    logreg_cv = GridSearchCV(logreg_0, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_logreg = logreg_cv.fit(x_train, y_train)
    print(best_logreg.best_params_)
    print(best_logreg.best_estimator_)
    y_predict = best_logreg.predict(x_test)
    return y_predict

def evaluate_model(y_true, y_predict):
    report = classification_report(y_true, y_predict)
    return report



