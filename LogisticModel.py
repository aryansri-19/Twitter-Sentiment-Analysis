from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
def LRprediction(X_train, y_train, test):
    model.fit(X=X_train, y=y_train)
    prediction_value = model.predict(test)
    return prediction_value