__author__ = 'phx'

import numpy
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from preprocessor import preprocess,getAttributeMap,getLabelMap,getDataSet
import json
train_file ="train.json"
with open(train_file) as file:
     data = json.load(file)
print(len(data))

data = preprocess(data)
"""
train_data = [data[i] for i in xrange(0,len(data)) if i%4 !=0]
test_data = [data[i] for i in xrange(0,len(data)) if i%4 ==0]
"""
train_data = data


test_file ="/Users/phx/downloads/competetion/recipe/test.json"
with open(test_file) as file:
     test_data = json.load(file)
print(len(test_data))
test_data= preprocess(test_data)


attribute_map = getAttributeMap(train_data,1)

print('attribute number : %d' % len(attribute_map))
print(attribute_map)


label_map = getLabelMap(data)
print(len(label_map))

def getReverseLabelMap(label_map):
    reverse_map ={}
    for k,v in label_map.items():
        reverse_map[v] = k
    return reverse_map
reverse_map= getReverseLabelMap(label_map)


X,y = getDataSet(train_data,attribute_map,label_map)

testX,testY= getDataSet(test_data,attribute_map,label_map,False)

#clf = BernoulliNB(alpha=0.3,fit_prior=False)
#clf = BernoulliNB(alpha=0.05,fit_prior=True)
"""
clf = MultinomialNB(alpha=0.08,fit_prior=True)

clf.fit(X,y)
pred = clf.predict(testX)
"""

def combineVote(votes):
    pool = {}
    for vote in votes:
        if vote in pool:
            pool[vote] = pool[vote]+1
        else:
            pool[vote]=1

    max = 1
    result = votes[0]
    for k,v in pool.items():
        if v>max:
            max = v
            result=k
    return result

def bagging(clfs,X,y,testX):
    result =[]
    for clf in clfs:
        clf.fit(X,y)
        pred = clf.predict(testX)
        result.append(pred)
    votedResult=[]
    for i in xrange(0,len(test_data)):
        votes = []
        for r in result:
            votes.append(r[i])
        votedResult.append(combineVote(votes))

    return numpy.array(votedResult)
pred = bagging([SGDClassifier(loss='log'),SGDClassifier(),MultinomialNB(alpha=0.08, fit_prior=True),RandomForestClassifier(n_estimators=50),BernoulliNB(alpha=0.08)],X,y,testX)
print("pred : %d , test_data: %d" % (len(pred),len(test_data)))
f=open('bagging_submission.csv','w')
f.write('id,cuisine\n')
for i in xrange(0,len(test_data)):
    f.write(str(test_data[i]['id'])+','+reverse_map[pred[i]]+'\n')
f.close()







