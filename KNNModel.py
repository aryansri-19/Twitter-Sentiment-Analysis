from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

def KNNprediction(X_train, y_train, test):
    model.fit(X=X_train, y=y_train)
    prediction_value = model.predict(test)
    return prediction_value