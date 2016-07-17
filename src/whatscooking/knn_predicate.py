__author__ = 'phx'
import json
import numpy
train_file ="/Users/phx/downloads/competetion/recipe/train.json"
with open(train_file) as file:
     data = json.load(file)
print(len(data))

train_data = [data[i] for i in xrange(0,len(data)) if i%3 !=0]
test_data = [data[i] for i in xrange(0,len(data)) if i%3 ==0]


print(len(train_data))

def getLabelMap(train_data):
    result = {}
    serial = 0;
    for record in train_data:
        if not result.get(record['cuisine']):
            result[record['cuisine']] = serial
            serial+=1
    return result

label_map = getLabelMap(data)

def balanceData(data):
    categorized_data ={}
    for record in data:
        if not categorized_data.get(record['cuisine']):
            categorized_data[record['cuisine']] = [record]
        else:
            categorized_data[record['cuisine']].append(record)
    result = []
    for key,value in categorized_data.iteritems():
        print("%s : %d" % (key, len(value)))
        if len(value)<=1000:
            result.extend(value)
        else:
            result.extend(value[:10001])
    return result




#train_data = balanceData(train_data)

def getDataSet(train_data,label_map):
    X = []
    y = []
    for record in train_data:
        y.append(label_map[record['cuisine']])
        X.append(record)
    return numpy.array(X,dtype=object),numpy.array(y)

X,y = getDataSet(train_data,label_map)


def knnPredict(X,y,test,testY,metric,K):
    result = []
    correct = 0
    for td in test:
        distances = map(lambda x : metric(x,td),X)
        votes = sorted(zip(distances,y),key = lambda x : x[0])[0:min(K,len(y))]
        voteResult = {}
        for vote in votes:
            if not voteResult.get(vote[1]):
                voteResult[vote[1]] = 1
            else:
                voteResult[vote[1]]=voteResult[vote[1]]+1
        max=0
        winner=-1
        for key,value in voteResult.iteritems():
            if value > max:
                max =value
                winner = key
        result.append(winner)
        print("%d : %d" % (winner,testY[len(result)-1]))

        if winner==testY[len(result)-1]:
            correct+=1
            print("correct : %d out of %d " % (correct,len(result)))

    return numpy.array(result)


from sklearn.neighbors import KNeighborsClassifier,DistanceMetric

def mySimilarity(a,b):
    sa= set(a['ingredients'])
    sb= set(b['ingredients'])
    return 1- len(sa.intersection(sb))*1.0/min(len(sa),len(sb))
#metric = DistanceMetric.get_metric('pyfunc', func=mySimilarity)
#neigh = KNeighborsClassifier(n_neighbors=10,metric=mySimilarity)
#neigh.fit(X,y)
testX,testY= getDataSet(test_data,label_map)
pred = knnPredict(X,y,testX,testY,mySimilarity,100)
print((pred!=testY).sum())

# k=10 correct : 7071 out of 13256
# k= 50 correct : 7655 out of 13256