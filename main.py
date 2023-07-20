import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import preprocessing
from KNNModel import KNNprediction
from LogisticModel import LRprediction
from MultinomialModel import MNBprediction
from RandomForestModel import RFprediction
from SVMModel import SVMprediction

def findAccuracy(y_test, pred):
  total = len(y_test)
  count = sum(1 for y, p in zip(y_test, pred) if y == p)
  acc = count/total * 100
  return acc

dataset = pd.read_csv('./Twitter_Data.csv', names=['Tweet', 'Sentiment'])
dataset = dataset[1:]
dataset = dataset.dropna(subset=['Tweet', 'Sentiment'])

print(dataset.describe())

dataset['Tweet'] = dataset['Tweet'].apply(lambda x: np.str_(x))
X_train = dataset['Tweet'].sample(frac=0.75, random_state=25)
indexes = X_train.index
X_test = dataset['Tweet'].drop(indexes)
y_train = dataset.loc[indexes, 'Sentiment']
y_test = dataset['Sentiment'].drop(y_train.index)

X_train = preprocessing(X_train)
count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train.apply(lambda x: np.str_(x)))
tfidf_vect = TfidfTransformer()
X_train_final = tfidf_vect.fit_transform(X_train_count)

X_test = preprocessing(X_test)
valid_count = count_vect.transform(X_test.apply(lambda x: np.str_(x)))
valid_final = tfidf_vect.transform(valid_count)

#Analysis of data in the form of 2D curve
y_axis = [dataset["Sentiment"].value_counts()[-1], dataset["Sentiment"].value_counts()[1], dataset["Sentiment"].value_counts()[0]]
x_axis = [-1, 0, 1]
plt.plot(x_axis, y_axis)
plt.show()

#In the form of Bar graph
sns.set_theme(style="darkgrid")
sns.countplot(x="Sentiment",data=dataset)
plt.show()

#KNN prediction
KNNpred = KNNprediction(X_train_final, y_train, valid_final)
print("Accuracy for KNN model is: ", findAccuracy(y_test, KNNpred))

#Logistic Regression
LRpred = LRprediction(X_train_final, y_train, valid_final)
print("Accuracy for Logistic Regression model is: ", findAccuracy(y_test, LRpred))

#Random Forest Classifier
RFpred = RFprediction(X_train_final, y_train, valid_final)
print("Accuracy for Random Forest Classifier model is: ", findAccuracy(y_test, RFpred))

#MultinomialNB 
MNBpred = MNBprediction(X_train_final, y_train, valid_final)
print("Accuracy for Multinomial Naive Bayes model is: ", findAccuracy(y_test, MNBpred))

#Support Vector Machine
SVCpred = SVMprediction(X_train_final, y_train, valid_final)
print("Accuracy for Logistic Regression model is: ", findAccuracy(y_test, SVCpred))
