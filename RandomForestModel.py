from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
def RFprediction(X_train, y_train, test):
    model.fit(X=X_train, y=y_train)
    prediction_value = model.predict(test)
    return prediction_value