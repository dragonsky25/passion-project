from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def log_regression(x_train, y_train, x_test):
    logreg = LogisticRegression(random_state=16)
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    return y_pred

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    return report

