__author__ = 'phx'
from classifier import Classifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from re import sub
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from os.path import isfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

removeableWords = set(['frozen','fresh','freshly','dried','cooked','chopped',
                        'ground','grated','crushed','minced','cracked','shredded',
                        'large','medium','small','warm','plain','low-fat','low-sodium',
                        'fat-free','sweetened','bottled','low','sodium','refried','flaked',
                        'whole','and','canned','boneless','sliced','of','fat free','nonfat',
                        'reduced','steel-cut','gluten-free'
                    ])

wnl = WordNetLemmatizer()
def lower(ingredients):
    return [ingredient.lower() for ingredient in ingredients]

def removeNonAlphabet(ingredients):
    return [sub('[^A-Za-z]+',' ',ingredient) for ingredient in ingredients]

def removeWords(ingredients):
    return [" ".join(filter(lambda x: x not in removeableWords,ingredient.split())) for ingredient in ingredients]

def lemmatize(ingredients):
    return [" ".join(map(lambda x:wnl.lemmatize(x),ingredient.split())) for ingredient in ingredients]


def getAttributeMap(train_X):
    result = {}
    serial = 0
    for ingredients in train_X:
        for ingredient in ingredients:
            if ingredient not in result:
                result[ingredient] = serial
                serial+=1
    return result

class DirectAttributeClassifier(Classifier):

    def _transform_data(self,X):
        attribute_size = len(self.ingredient_map)
        record_num = len(X)
        from scipy.sparse import lil_matrix
        import numpy
        #new_X = lil_matrix((record_num,attribute_size),dtype=numpy.int8)
        new_X = np.zeros((record_num,attribute_size))
        for i in xrange(0,record_num):
            ingredients = X[i]
            for ingredient in ingredients:
                if self.ingredient_map.get(ingredient):
                    new_X[i,self.ingredient_map.get(ingredient)]=1
        return new_X

    def _preprocess_data(self,X):
        X = [lower(ingredients) for ingredients in X]
        X = [removeNonAlphabet(ingredients) for ingredients in X]
        X = [removeWords(ingredients) for ingredients in X]
        X = [lemmatize(ingredients) for ingredients in X]
        return X

    def get_feature_and_importances(self,X,y,importance=True):
        print('get_feature_and_importances')
        X = self._preprocess_data(X)
        importances = {}
        if importance:
            self.ingredient_map = getAttributeMap(X)
            train_X = self._transform_data(X)
            rf = RandomForestClassifier(n_estimators=500)
            rf.fit(train_X,y)
            rf_feature_importances = rf.feature_importances_
            for key,value in self.ingredient_map.items():
                importances[key] = rf_feature_importances[value]
        return X,importances



    def __init__(self):
        self.sgd_pipe = Pipeline([('sgd',SGDClassifier(loss='log'))])
        self.mnb_pipe = Pipeline([('mnb',MultinomialNB(alpha=0.08))])
        #self.knn_pipe = Pipeline([('knn',KNeighborsClassifier(n_neighbors=10,metric='jaccard'))])
    def fit(self,X,y):

        X = self._preprocess_data(X)
        self.ingredient_map = getAttributeMap(X)
        X = self._transform_data(X)
        self.sgd_pipe.fit(X,y)
        self.mnb_pipe.fit(X,y)
        #self.knn_pipe.fit(X,y)

    def predict(self,X):
        X = self._preprocess_data(X)
        X = self._transform_data(X)
        return self.sgd_pipe.predict(X)

    def test(self,X,y,test_X,test_Y):
        self.fit(X,y)
        test_X = self._preprocess_data(test_X)
        test_X = self._transform_data(test_X)


        print(self.sgd_pipe)
        pred = self.sgd_pipe.predict(test_X)
        print(metrics.accuracy_score(test_Y,pred))

        print(self.mnb_pipe)
        pred = self.mnb_pipe.predict(test_X)
        print(metrics.accuracy_score(test_Y,pred))




    def get_proba(self,X,y,test_X,prefix=''):
        if isfile(prefix+'direct_attribute_sgd_train.pkl'):
            sgd_proba_train = pd.read_pickle(prefix+'direct_attribute_sgd_train.pkl')
            sgd_proba_test = pd.read_pickle(prefix+'direct_attribute_sgd_test.pkl')
            mnb_proba_train = pd.read_pickle(prefix+'direct_attribute_mnb_train.pkl')
            mnb_proba_test = pd.read_pickle(prefix+'direct_attribute_mnb_test.pkl')
        else:
            X = self._preprocess_data(X)
            self.ingredient_map = getAttributeMap(X)
            X = self._transform_data(X)
            self.sgd_pipe.fit(X,y)
            self.mnb_pipe.fit(X,y)
            test_X = self._preprocess_data(test_X)
            test_X = self._transform_data(test_X)

            sgd_proba_train = pd.DataFrame(self.sgd_pipe.predict_proba(X))
            sgd_proba_train.to_pickle(prefix+'direct_attribute_sgd_train.pkl')
            sgd_proba_test = pd.DataFrame(self.sgd_pipe.predict_proba(test_X))
            sgd_proba_test.to_pickle(prefix+'direct_attribute_sgd_test.pkl')
            mnb_proba_train = pd.DataFrame(self.mnb_pipe.predict_proba(X))
            mnb_proba_train.to_pickle(prefix+'direct_attribute_mnb_train.pkl')
            mnb_proba_test = pd.DataFrame(self.mnb_pipe.predict_proba(test_X))
            mnb_proba_test.to_pickle(prefix+'direct_attribute_mnb_test.pkl')
        return [(sgd_proba_train,sgd_proba_test),(mnb_proba_train,mnb_proba_test)]
