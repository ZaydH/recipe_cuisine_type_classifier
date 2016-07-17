__author__ = 'phx'
import pandas as pd
from os.path import isfile
import numpy as np
from bag_of_words import BagOfWordsClassifier
from direct_attribute import DirectAttributeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import grid_search
from sklearn import svm
from sklearn.linear_model import SGDClassifier,Perceptron
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_test_data():
    """
    Get data for test.
    70% as training data
    30% as test data
    Split only once so the training data and test data will be fixed

    """
    if isfile("t_train.pkl") and isfile("t_test.pkl"):
        train = pd.read_pickle("t_train.pkl")
        test = pd.read_pickle("t_test.pkl")
    else:
        train_file ="/Users/phx/downloads/competetion/recipe/train.json"
        df = pd.read_json(train_file)
        mask = np.random.rand(len(df)) < 0.7
        train = df[mask]
        test = df[~mask]
        train.to_pickle("t_train.pkl")
        test.to_pickle("t_test.pkl")
    return train['ingredients'],train['cuisine'],test['ingredients'],test['cuisine']


def get_predict_data():
    if isfile("train.pkl") and isfile("test.pkl"):
        train = pd.read_pickle("train.pkl")
        test = pd.read_pickle("test.pkl")
    else:
        train_file ="/Users/phx/downloads/competetion/recipe/train.json"
        test_file ="/Users/phx/downloads/competetion/recipe/test.json"
        train = pd.read_json(train_file)
        test = pd.read_json(test_file)
        train.to_pickle("train.pkl")
        test.to_pickle("test.pkl")
    return train['ingredients'],train['cuisine'],test['ingredients'],test['id']

def predict():
    X,y,test_X,ids =get_predict_data()
    print("bag of words")
    bow = BagOfWordsClassifier()
    bow_probs = bow.get_proba(X,y,test_X,prefix="p_")

    print("direct attribute")
    da = DirectAttributeClassifier()
    da_probs = da.get_proba(X,y,test_X,prefix="p_")

    probs = zip(*[item for p in [bow_probs,da_probs] for item in p])
    train_probs = probs[0]
    test_probs = probs[1]
    print(len(train_probs))
    for prob in train_probs:
        print(prob.shape)
        print(type(prob))

    train_attr = pd.concat(train_probs,axis=1)
    print(train_attr.shape)
    print(type(train_attr))

    test_attr = pd.concat(test_probs,axis=1)
    print(test_attr.shape)
    print(type(test_attr))

    clf = LogisticRegression()
    clf.fit(train_attr,y)
    pred=clf.predict(test_attr)
    result = pd.DataFrame({'id':ids,'cuisine':pred})
    result[['id','cuisine']].to_csv("av_submission.csv",index=False,cols=["id","cuisine"],engine='python')


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

def test_hard_vote():
    X,y,test_X,test_Y =get_test_data()

    print("bag of words")
    bow = BagOfWordsClassifier()
    bow_probs = bow.get_proba(X,y,test_X,prefix="t")

    print("direct attribute")
    da = DirectAttributeClassifier()
    da_probs = da.get_proba(X,y,test_X,prefix="t")

    probs = zip(*[item for p in [bow_probs,da_probs] for item in p])
    #train_probs = probs[0]
    test_probs = probs[1]
    print(len(test_probs))
    preds = [x.idxmax(1) for x in test_probs]
    pred = np.zeros(len(preds[0]),dtype=np.int8)
    print(len(pred))
    for i in range(len(preds[0])):
        votes = [p[i] for p in preds]
        print(votes)
        pred[i]= max(set(votes),key=votes.count)
        print(pred[i])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y)
    pred = le.inverse_transform(pred)

    print(metrics.accuracy_score(test_Y,pred))

    """
    0.779675275511
    """



def test_vote_soft():
    X,y,test_X,test_Y =get_test_data()

    print("bag of words")
    bow = BagOfWordsClassifier()
    bow_probs = bow.get_proba(X,y,test_X,prefix="t")

    print("direct attribute")
    da = DirectAttributeClassifier()
    da_probs = da.get_proba(X,y,test_X,prefix="t")

    probs = zip(*[item for p in [bow_probs,da_probs] for item in p])
    train_probs = probs[0]
    test_probs = probs[1]
    print(len(train_probs))
    for prob in train_probs:
        print(prob.shape)
        print(type(prob))
    #train_attr = reduce(lambda a,b:a+b,train_probs)
    test_attr = reduce(lambda a,b:a+b,test_probs)

    pred = test_attr.idxmax(1)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y)
    pred = le.inverse_transform(pred)

    print(metrics.accuracy_score(test_Y,pred))




def test():
    X,y,test_X,test_Y =get_test_data()

    print("bag of words")
    bow = BagOfWordsClassifier()
    bow_probs = bow.get_proba(X,y,test_X,prefix="t")

    print("direct attribute")
    da = DirectAttributeClassifier()
    da_probs = da.get_proba(X,y,test_X,prefix="t")

    probs = zip(*[item for p in [bow_probs,da_probs] for item in p])
    train_probs = probs[0]
    test_probs = probs[1]
    print(len(train_probs))
    for prob in train_probs:
        print(prob.shape)
        print(type(prob))

    train_attr = pd.concat(train_probs,axis=1)
    print(train_attr.shape)
    print(type(train_attr))

    test_attr = pd.concat(test_probs,axis=1)
    print(test_attr.shape)
    print(type(test_attr))


    #clf = LogisticRegression()
    #clf = svm.SVC()
    """
    params={'kernel':('rbf','linear','poly','sigmoid'),'C':[1,10]}
    clf = grid_search.GridSearchCV(svm.SVC(),params,cv=5)

    params={'penalty':('l1','l2'),'C':[1,10]}
    clf = grid_search.GridSearchCV(LogisticRegression(),params,cv=5)
    """
    #clf = SGDClassifier(loss="log")

    """
    params = {'loss':['hinge','log','modified_huber','squared_hinge','perceptron'],
              'penalty':['l1','l2','elasticnet'],
              'alpha':[0.0001,0.001,0.01,0.1]}
    clf = grid_search.GridSearchCV(SGDClassifier(),params,cv=5)
    """
    """
    clf.fit(train_attr,y)

    #print(clf.best_params_)
    pred = clf.predict(test_attr)
    print(clf)
    print(metrics.accuracy_score(test_Y,pred))

    """
    """
    clf = RandomForestClassifier(n_estimators=50)
    benchmark(clf,train_attr,y,test_attr,test_Y)
    """
    """
    clf = GradientBoostingClassifier(n_estimators=50)
    benchmark(clf,train_attr,y,test_attr,test_Y)


    clf=DecisionTreeClassifier()

    benchmark(clf,train_attr,y,test_attr,test_Y)
    """
    """
    clf = AdaBoostClassifier(base_estimator=SGDClassifier(loss="log"))
    benchmark(clf,train_attr,y,test_attr,test_Y)


    clf = BaggingClassifier(LogisticRegression())
    benchmark(clf,train_attr,y,test_attr,test_Y)

    clf = LogisticRegression()
    benchmark(clf,train_attr,y,test_attr,test_Y)


    clf = SGDClassifier(loss="log")
    benchmark(clf,train_attr,y,test_attr,test_Y)

    clf=Perceptron()
    benchmark(clf,train_attr,y,test_attr,test_Y)
    clf =GaussianNB()
    benchmark(clf,train_attr,y,test_attr,test_Y)

    """
    """
    gnb = GaussianNB()
    lgc = LogisticRegression()
    sgd = SGDClassifier(loss="log")
    from mlxtend.classifier import EnsembleClassifier
    clf = EnsembleClassifier(clfs=[sgd,lgc,gnb],voting='hard')
    benchmark(clf,train_attr,y,test_attr,test_Y)
    """


"""
________________________________________________________________________________
training
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
training time: 8.440s
testing time: 0.307s
accuracy: 69.29%

training
BaggingClassifier(base_estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False),
         bootstrap=True, bootstrap_features=False, max_features=1.0,
         max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False,
         random_state=None, verbose=0)
training time: 10.894s
testing time: 0.281s
accuracy: 79.50%

________________________________________________________________________________
training
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
training time: 1.195s
testing time: 0.021s
accuracy: 79.47%

________________________________________________________________________________
training
BaggingClassifier(base_estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0),
         bootstrap=True, bootstrap_features=False, max_features=1.0,
         max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False,
         random_state=None, verbose=0)
training time: 97.253s
testing time: 0.282s
accuracy: 79.24%

________________________________________________________________________________
training
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0)
training time: 9.343s
testing time: 0.025s
accuracy: 79.27%

training
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      n_iter=5, n_jobs=1, penalty=None, random_state=0, shuffle=True,
      verbose=0, warm_start=False)
training time: 0.744s
testing time: 0.024s
accuracy: 70.78%

________________________________________________________________________________
training
GaussianNB()
training time: 0.214s
testing time: 0.275s
accuracy: 75.32%

training
EnsembleClassifier(clfs=[SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False),...penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0), GaussianNB()],
          verbose=0, voting='hard', weights=None)
training time: 11.534s
testing time: 0.511s
accuracy: 79.57%
"""
    #bow.test(X,y,test_X,test_Y)


def test_direct_attribute():
    X,y,test_X,test_Y =get_test_data()
    da = DirectAttributeClassifier()
    da.test(X,y,test_X,test_Y)



if __name__ == "__main__":
    #test_direct_attribute()
    #test()
    #predict()
    #test_vote_soft()
    test_hard_vote()
