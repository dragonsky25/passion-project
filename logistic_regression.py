from sklearn.linear_model import LogisticRegression

def log_regression(x_train, y_train, x_test):
    logreg = LogisticRegression(random_state=16)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    return y_pred

