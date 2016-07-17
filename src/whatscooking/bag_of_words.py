__author__ = 'phx'
from classifier import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from os.path import  isfile
import pandas as pd

class BagOfWordsClassifier(Classifier):

    def _transform_data(self,X):
        return [" ".join(x) for x in X]

    def __init__(self):
        self.random_forest_pipe= Pipeline([
            ('vectorizer',CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('model', RandomForestClassifier(n_estimators=500,n_jobs=3,max_features="log2"))
        ])
    def predict(self,X):
        X = self._transform_data(X)
        return self.random_forest_pipe.predict(X)

    def fit(self,X,y):
        X = self._transform_data(X)
        self.random_forest_pipe.fit(X,y)

    def get_proba(self,X,y,test_X,prefix=''):
        if isfile(prefix+'bag_of_words_random_forest_train.pkl'):
            rf_proba_train = pd.read_pickle(prefix+'bag_of_words_random_forest_train.pkl')
            rf_proba_test =  pd.read_pickle(prefix+'bag_of_words_random_forest_test.pkl')
        else:
            self.fit(X,y)
            X = self._transform_data(X)
            test_X=self._transform_data(test_X)
            rf_proba_train = pd.DataFrame(self.random_forest_pipe.predict_proba(X))
            rf_proba_train.to_pickle(prefix+'bag_of_words_random_forest_train.pkl')
            rf_proba_test = pd.DataFrame(self.random_forest_pipe.predict_proba(test_X))
            rf_proba_test.to_pickle(prefix+'bag_of_words_random_forest_test.pkl')
        return [(rf_proba_train,rf_proba_test)]


    def test(self,X,y,test_X,test_Y):
        print("_"*80)
        print(self.random_forest_pipe)
        self.fit(X,y)
        pred = self.predict(test_X)
        print(metrics.accuracy_score(test_Y,pred))
        rf = self.random_forest_pipe.named_steps['model']
        print(rf.feature_importances_)


if __name__ == '__main__':
    from predict import get_test_data
    X,y,test_X,test_Y =get_test_data()
    bow = BagOfWordsClassifier()
    bow.test(X,y,test_X,test_Y)