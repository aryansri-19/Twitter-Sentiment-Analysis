from sklearn.svm import SVC

model = SVC(kernel='linear', random_state=25)
def SVMprediction(X_train, y_train, test):
    model.fit(X=X_train, y=y_train)
    prediction_value = model.predict(test)
    return prediction_value