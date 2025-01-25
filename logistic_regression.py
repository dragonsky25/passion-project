from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def log_regression(x_train, y_train, x_test):
    logreg = LogisticRegression(random_state=16)
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg.fit(x_train, y_train)
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'max_iter': [100, 1000, 2500, 5000, 10000],
         'class_weight': ['balanced']
         }
    logreg = LogisticRegression(random_state=16, max_iter=10000)
    clf = GridSearchCV(logreg, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print(f'Accuracy - : {best_clf.score(x_train, y_train):.3f}')
    return y_pred

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    return report


