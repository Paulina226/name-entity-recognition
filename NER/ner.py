from collections import Counter

import eli5 as eli5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from FeatureExtractor import FeatureExtractor
from SentenceExtractor import SentenceExtractor

data_frame = pd.read_csv('../ner_dataset.csv', encoding='ISO-8859-1')
data_frame = data_frame[:10000]
# structure of data set
print(data_frame.head())

print('Number of NaN values: ', data_frame.isnull().sum())

# fill NaN values with preceding data
data_frame = data_frame.fillna(method='ffill')
print(data_frame.head())
print(data_frame['Sentence #'].nunique(), data_frame.Word.nunique(), data_frame.Tag.nunique())

# ==== TRANSFORMING TEXT TO VECTOR ====
vector = DictVectorizer(sparse=False)

# splitting data frame into data and class columns
X = data_frame.drop('Tag', axis='columns')
X = vector.fit_transform(X.to_dict('records'))
y = data_frame.Tag.values

# ==== TRAIN-TEST SPLIT ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ==== CLASSIFICATION USING REGULAR CLASSIFIERS ====

# ** Random Forest **

# very basic, naive approach - we assume the word has a features like if it's an uppercase, title and so on
def get_feature_from_word(word: str):
    return np.array([len(word), word.isdigit(), word.isupper(), word.islower()])


words = [get_feature_from_word(word) for word in data_frame["Word"].values.tolist()]
# tags are classes in the model
tags = data_frame["Tag"].values.tolist()
rf = RandomForestClassifier(n_estimators=20)
rf_y_pred = cross_val_predict(rf,  X=words, y=tags)

print(classification_report(y_pred=rf_y_pred, y_true=tags))

# ** Multi-layer Percepton **
perceptron = Perceptron(verbose=20, n_jobs=-1)
tags = np.unique(y)
tags = tags.tolist()
# use of partial_fit (out-of-core algorithm) to process data with limited amount of RAM
perceptron.partial_fit(X_train, y_train, tags)

# remove the most common tag 'O' not to disturb evaluation metrics
new_tags = tags.copy()
del new_tags[-1]

perceptron_y_pred = perceptron.predict(X_test)
# classification report
print(classification_report(y_pred=perceptron_y_pred, y_true=y_test, labels=new_tags))

# ** Stochastic Gradient Descend **

sgd = SGDClassifier()
sgd.partial_fit(X_train, y_train, tags)
sgd_y_pred = sgd.predict(X_test)
print(classification_report(y_pred=sgd_y_pred, y_true=y_test, labels=new_tags))
