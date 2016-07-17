__author__ = 'phx'

from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import numpy
from preprocessor import preprocess,getDataSet,getLabelMap,getAttributeMap
import json
train_file ="/Users/phx/downloads/competetion/recipe/train.json"
with open(train_file) as file:
     data = json.load(file)
print("size of dataset %d" % len(data))

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



def balanceData(data):
    categorized_data ={}
    for record in data:
        if not categorized_data.get(record['cuisine']):
            categorized_data[record['cuisine']] = [record]
        else:
            categorized_data[record['cuisine']].append(record)
    result = []
    for key,value in categorized_data.iteritems():
        #print("%s : %d" % (key, len(value)))
        if len(value)<=500:
            result.extend(value)
        else:
            result.extend(value[:501])
    return result




#train_data = balanceData(train_data)



X,y = getDataSet(train_data,attribute_map,label_map)

testX,testY= getDataSet(test_data,attribute_map,label_map)

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
"""
alpha = 0.01
while alpha <1:
    clf = BernoulliNB(alpha=alpha,fit_prior=False)
    benchmark(clf,X,y,testX,testY)
    alpha *=2
alpha = 0.01
while alpha <1:
    clf = BernoulliNB(alpha=alpha,fit_prior=True)
    benchmark(clf,X,y,testX,testY)
    alpha *=2
alpha = 0.01
while alpha <1:
    clf = MultinomialNB(alpha=alpha,fit_prior=True)
    benchmark(clf,X,y,testX,testY)
    alpha *=2

clf = RandomForestClassifier(n_estimators=10)
benchmark(clf,X,y,testX,testY)
clf = RandomForestClassifier(n_estimators=20)
benchmark(clf,X,y,testX,testY)
clf = RandomForestClassifier(n_estimators=50)
benchmark(clf,X,y,testX,testY)
clf = RandomForestClassifier(n_estimators=100)
benchmark(clf,X,y,testX,testY)

clf = SGDClassifier()
benchmark(clf,X,y,testX,testY)

clf = RandomForestClassifier(n_estimators=200)
benchmark(clf,X,y,testX,testY)
clf = RandomForestClassifier(n_estimators=300)

benchmark(clf,X,y,testX,testY)

clf = RandomForestClassifier(n_estimators=500)

benchmark(clf,X,y,testX,testY)

clf = BaggingClassifier(SGDClassifier(loss='log'),max_samples=0.5)
benchmark(clf,X,y,testX,testY)

clf =AdaBoostClassifier(SGDClassifier(loss='log'),n_estimators=100)
benchmark(clf,X,y,testX,testY)

clf = tree.DecisionTreeClassifier()
benchmark(clf,X,y,testX,testY)




clf = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=100)
benchmark(clf,X,y,testX,testY)
"""
"""
sgd = SGDClassifier(loss='log')
sgd.fit(X,y)
prob_sgd = sgd.predict_proba(X)
#print(prob)
print(prob_sgd.shape)

mnb=MultinomialNB(alpha=0.08, class_prior=None, fit_prior=True)
mnb.fit(X,y)
prob_mnb = mnb.predict_proba(X)

print(prob_mnb.shape)
"""
def mydist(x,y):
    print(x)
    print(y)
    return 1

knn = KNeighborsClassifier(n_neighbors=50,metric='pyfunc',func=mydist)
knn.fit(X,y)
knn.predict(testX)


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

def bagging(clfs,X,y,testX,testY):
    result =[]
    for clf in clfs:
        clf.fit(X,y)
        pred = clf.predict(testX)
        result.append(pred)
    votedResult=[]
    for i in xrange(len(testY)):
        votes = []
        for r in result:
            votes.append(r[i])
        votedResult.append(combineVote(votes))

    score = metrics.accuracy_score(testY,numpy.array(votedResult))
    print("accuracy: %0.2f%%" % (score*100))
#mnb = MultinomialNB(alpha=0.01,fit_prior=False)
#benchmark(mnb,X,y,testX,testY)



#bagging([SGDClassifier(loss='log'),SGDClassifier(),MultinomialNB(alpha=0.08, fit_prior=True),RandomForestClassifier(n_estimators=50),BernoulliNB(alpha=0.08)],X,y,testX,testY)
"""


#Result with clean ingredients

training
BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.147s
testing time: 0.070s
accuracy: 71.71%
________________________________________________________________________________
training
BernoulliNB(alpha=0.02, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.141s
testing time: 0.068s
accuracy: 71.84%
________________________________________________________________________________
training
BernoulliNB(alpha=0.04, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.140s
testing time: 0.069s
accuracy: 71.99%
________________________________________________________________________________
training
BernoulliNB(alpha=0.08, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.146s
testing time: 0.070s
accuracy: 72.36%
________________________________________________________________________________
training
BernoulliNB(alpha=0.16, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.148s
testing time: 0.070s
accuracy: 73.01%
________________________________________________________________________________
training
BernoulliNB(alpha=0.32, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.141s
testing time: 0.073s
accuracy: 73.90%
________________________________________________________________________________
training
BernoulliNB(alpha=0.64, binarize=0.0, class_prior=None, fit_prior=False)
training time: 0.149s
testing time: 0.074s
accuracy: 73.32%
________________________________________________________________________________
training
BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.138s
testing time: 0.066s
accuracy: 73.99%
________________________________________________________________________________
training
BernoulliNB(alpha=0.02, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.150s
testing time: 0.068s
accuracy: 74.32%
________________________________________________________________________________
training
BernoulliNB(alpha=0.04, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.142s
testing time: 0.069s
accuracy: 74.61%
________________________________________________________________________________
training
BernoulliNB(alpha=0.08, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.141s
testing time: 0.068s
accuracy: 74.71%
________________________________________________________________________________
training
BernoulliNB(alpha=0.16, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.140s
testing time: 0.073s
accuracy: 74.63%
________________________________________________________________________________
training
BernoulliNB(alpha=0.32, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.146s
testing time: 0.067s
accuracy: 74.54%
________________________________________________________________________________
training
BernoulliNB(alpha=0.64, binarize=0.0, class_prior=None, fit_prior=True)
training time: 0.141s
testing time: 0.066s
accuracy: 72.32%
________________________________________________________________________________
training
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
training time: 0.140s
testing time: 0.065s
accuracy: 74.40%
________________________________________________________________________________
training
MultinomialNB(alpha=0.02, class_prior=None, fit_prior=True)
training time: 0.139s
testing time: 0.067s
accuracy: 74.69%
________________________________________________________________________________
training
MultinomialNB(alpha=0.04, class_prior=None, fit_prior=True)
training time: 0.138s
testing time: 0.061s
accuracy: 74.98%
________________________________________________________________________________
training
MultinomialNB(alpha=0.08, class_prior=None, fit_prior=True)
training time: 0.146s
testing time: 0.061s
accuracy: 75.16%
________________________________________________________________________________
training
MultinomialNB(alpha=0.16, class_prior=None, fit_prior=True)
training time: 0.138s
testing time: 0.060s
accuracy: 75.15%
________________________________________________________________________________
training
MultinomialNB(alpha=0.32, class_prior=None, fit_prior=True)
training time: 0.138s
testing time: 0.061s
accuracy: 75.10%
________________________________________________________________________________
training
MultinomialNB(alpha=0.64, class_prior=None, fit_prior=True)
training time: 0.153s
testing time: 0.065s
accuracy: 74.20%
________________________________________________________________________________
training
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
training time: 7.993s
testing time: 0.159s
accuracy: 67.08%
________________________________________________________________________________
training
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
training time: 15.563s
testing time: 0.280s
accuracy: 69.73%
________________________________________________________________________________
training
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
training time: 39.342s
testing time: 0.585s
accuracy: 71.07%
________________________________________________________________________________
training
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
training time: 77.590s
testing time: 1.196s
accuracy: 71.62%
________________________________________________________________________________
training
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
training time: 0.571s
testing time: 0.061s
accuracy: 75.82%

________________________________________________________________________________
training
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
training time: 0.724s
testing time: 0.064s
accuracy: 76.83%
________________________________________________________________________________
training
BaggingClassifier(base_estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False),
         bootstrap=True, bootstrap_features=False, max_features=1.0,
         max_samples=0.5, n_estimators=10, n_jobs=1, oob_score=False,
         random_state=None, verbose=0)
training time: 5.378s
testing time: 0.226s
accuracy: 75.43%

________________________________________________________________________________
training
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False),
          learning_rate=1.0, n_estimators=100, random_state=None)
training time: 66.410s
testing time: 2.363s
accuracy: 47.96%

________________________________________________________________________________
training
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
training time: 26.599s
testing time: 0.079s
accuracy: 60.96%

________________________________________________________________________________
training
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best'),
          learning_rate=1.0, n_estimators=100, random_state=None)
training time: 4332.199s
testing time: 2.466s
accuracy: 68.30%

"""


