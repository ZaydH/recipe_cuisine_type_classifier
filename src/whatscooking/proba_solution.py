__author__ = 'phx'
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy
from preprocessor import preprocess,getDataSet,getLabelMap,getAttributeMap
import json
import pandas as pd

def get_proba(clf,X,y,testX,testY):
    print(clf)
    clf.fit(X,y)
    proba_train = clf.predict_proba(X)
    proba_test = clf.predict_proba(testX)
    return (pd.DataFrame(proba_train),pd.DataFrame(proba_test))

def generate_save_proba(clf,X,y,testX,testY,name):
    proba_train,proba_test = get_proba(clf,X,y,testX,testY)
    proba_train.to_pickle(name+"_train.pkl")
    proba_test.to_pickle(name+"_test.pkl")

def load_proba(name):
    proba_train = pd.read_pickle(name+"_train.pkl")
    proba_test = pd.read_pickle(name+"_test.pkl")
    return (proba_train,proba_test)

def get_data(clfs,X,y,testX,testY):
    probas = map(lambda x :get_proba(x,X,y,testX,testY),clfs)
    prob_train = probas[0][0]
    for i in range(1,len(probas)):
        prob_train = prob_train.merge(probas[i][0],how='left')
    print(prob_train.shape)
    prob_test = probas[0][1]

    for i in range(1,len(probas)):
        prob_test = prob_test.merge(probas[i][1],how='left')
    return (prob_train,y,prob_test,testY)


def main():
    train_file ="/Users/phx/downloads/competetion/recipe/train.json"
    with open(train_file) as file:
        data = json.load(file)
    print("size of dataset %d" % len(data))

    data = preprocess(data)
    data = preprocess(data)

    train_data = [data[i] for i in xrange(0,len(data)) if i%3 !=0]
    test_data = [data[i] for i in xrange(0,len(data)) if i%3 ==0]

    #test_data= preprocess(test_data)

    attribute_map = getAttributeMap(train_data,1)

    print('attribute number : %d' % len(attribute_map))
    print(attribute_map)


    label_map = getLabelMap(data)
    print('label number : %d' %len(label_map))
    print(label_map)
    X,y = getDataSet(train_data,attribute_map,label_map)
    testX,testY= getDataSet(test_data,attribute_map,label_map)
    sgd = SGDClassifier(loss='log')
    generate_save_proba(sgd,X,y,testX,testY,"SGDClassifier.loss_log")
    mnb = MultinomialNB(alpha=0.08, class_prior=None, fit_prior=True)
    generate_save_proba(mnb,X,y,testX,testY,"MultinomialNB.alpha_0.08")
    rf = RandomForestClassifier(n_estimators=500)
    generate_save_proba(rf,X,y,testX,testY,"RandomForestClassifier.n_estimators_500")
    """
    (m_X,m_y,m_tx,m_ty)= get_data([sgd,mnb,rf],X,y,testX,testY)

    clf = LogisticRegression()
    clf.fit(m_X,m_y)
    pred = clf.predict(m_tx)
    print(metrics.accuracy_score(m_ty,pred))
    """



if __name__ == '__main__':
    main()