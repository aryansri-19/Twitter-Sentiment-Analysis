# Twitter-Sentiment-Analysis
A machine learning project to analyze tweets and also predict the sentiment behind them.

Tweets are divided into 3 classes (categories) :
1. Positive (1)
2. Neutral (0)
3. Negative (-1)

The project uses 5 most commonly used models to predict the sentiment and analyse the tweets.
Models are K-Nearest Neighbor, Logistic Regression, Random Forest, Multinomial Naive Bayes and Support Vector Machine.

Preprocessing of data is done by nltk library and CountVectorizer and TF-IDF Vectorizer.

To get started:
```
git-clone https://github.com/aryansri-19/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
python main.py
```

The data is of Indian National Election 2019, gathered from Kaggle.
https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset
