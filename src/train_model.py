from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

def logistic_reg(df):
  tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3)) #1-3 character/words, 
  X_train, X_test, y_train, y_test = train_test_split(df['memo'],df['new_category'],test_size=0.3, random_state=42)
  tfidf.fit(X_train)
  x_train = tfidf.transform(X_train)
  x_test = tfidf.transform(X_test)
  lr=LogisticRegression()
  lr.fit(x_train,y_train)
  ypred = lr.predict(x_test)
  print('model accuracy: '+str(metrics.accuracy_score(y_test,ypred)))
