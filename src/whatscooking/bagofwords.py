__author__ = 'phx'
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import metrics
from gridcvsearch import grid_CV_Search
train_file ="/Users/phx/downloads/competetion/recipe/train.json"
df = pd.read_json(train_file)
mask = np.random.rand(len(df)) < 0.7
train = df[mask]
test = df[~mask]



print('training data size : %d, test data size : %d' % (len(train),len(test)))

train_ingredients = train['ingredients']
train_word_list = [" ".join(x) for x in train_ingredients]

transformer = TfidfTransformer()

test_ingredients = test['ingredients']
test_word_list = [" ".join(x) for x in test_ingredients]

vectorizer = CountVectorizer()
vectorizer.fit(train_word_list)
bag_of_words = vectorizer.transform(train_word_list).toarray()

tfidf= transformer.fit_transform(bag_of_words)

clf = RandomForestClassifier(n_estimators=500,n_jobs=3,max_features="log2")
#clf = SGDClassifier(loss='log')
#clf = GaussianNB()
#clf = BernoulliNB(alpha=0.5)
#clf.fit(bag_of_words,train['cuisine'])
#clf = GradientBoostingClassifier(n_estimators=750)
#clf = xgb.XGBClassifier(n_estimators=750)

test_ingredients_array = vectorizer.transform(test_word_list).toarray()
test_tfidf = transformer.transform(test_ingredients_array)

#result = clf.predict(test_ingredients_array)

def benchmark(clf,X,y,testX,testY):
    print("_"*80)
    print("training")
    print(clf)
    from time import time
    t0 = time()
    clf.fit(X,y)
    print("training time: %0.3fs" % (time() - t0))
    t0 = time()
    pred = clf.predict(testX)
    print("testing time: %0.3fs" % (time() -t0))
    score = metrics.accuracy_score(testY,pred)
    print("accuracy: %0.2f%%" % (score*100))

benchmark(clf,bag_of_words,train['cuisine'],test_ingredients_array,test['cuisine'])
#benchmark(clf,tfidf,train['cuisine'],test_tfidf,test['cuisine'])

clfs=[
      (RandomForestClassifier(),
       [{
          'criterion':['gini','entropy'],
          'n_estimators':[10,100,500,1000],
          'max_features':["sqrt","log2"]
       }]
      ),
      (MultinomialNB(),
       [{
          'alpha':[0.1,0.2,0.4,0.5,0.7,0.9,1.0],
          'fit_prior':[True,False]
       }]
      ),
      (BernoulliNB(),
       [{
          'alpha':[0.1,0.2,0.4,0.5,0.7,0.9,1.0],
          'fit_prior':[True,False]
       }]
      ),
      (SGDClassifier(),
       [{'loss':['log','hinge','perceptron','modified_hube','squared_hinge'],
         'alpha':[0.00001,0.0001,0.001,0.01,0.1],
         'penalty':['l2','l1','elasticnet']
        }]
      )
    ]
#grid_CV_Search(clfs,bag_of_words,train['cuisine'],test_ingredients_array,test['cuisine'])