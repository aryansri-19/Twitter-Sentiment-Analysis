import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocessing(X_train):
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    X_train = X_train.str.lower()
    X_train = X_train.apply(word_tokenize)
    X_train = X_train.apply(lambda tokens: [x for x in tokens if x not in stop])
    X_train = X_train.apply(lambda tokens: [lemma.lemmatize(x) for x in tokens])
    return X_train