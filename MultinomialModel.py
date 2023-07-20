from sklearn.naive_bayes import MultinomialNB


model = MultinomialNB()
def MNBprediction(X_train, y_train, test):
    model.fit(X=X_train, y=y_train)
    prediction_value = model.predict(test)
    return prediction_value