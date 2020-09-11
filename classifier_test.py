__author__ = 'xead'
# coding: utf-8

from sentiment_classifier import SentimentClassifier
from sklearn.externals import joblib
from eli5.lime import TextExplainer

#clf = SentimentClassifier()

#pred = clf.get_prediction_message("Хороший телефон")
text = 'Хороший был у меня телефон 5 лет назад'

pipe = joblib.load("./pipe6.pkl")
te = TextExplainer(random_state=42)
te.fit(text, pipe.predict_proba)
res = te.show_prediction(target_names=['negative', 'positive'], top=25)

print (res)