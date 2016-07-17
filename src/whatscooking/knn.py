__author__ = 'phx'
from classifier import Classifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn import metrics


def weighted_overlap(a,b,feature_importance):
    aSet = set(a)
    bSet = set(b)
    intersection = aSet.intersection(bSet)
    difference = aSet.symmetric_difference(bSet)
    if len(intersection) == 0:
        return 99999999
    def get_importance(s):
        importances = [feature_importance[x] for x in s]
        importance = reduce(lambda a,b:a+b,importances,0)
        return importance
    return len(difference)/get_importance(intersection)

def overlap(a,b):
    aSet = set(a)
    bSet = set(b)
    return (len(aSet)+len(bSet))*1.0/(len(aSet.intersection(bSet))+1)

def uniformWeight(distance):
    return 1

def distanceWeight(distance):
    return 1.0/(distance**2)

class KNNClassifier(Classifier):

    def __init__(self,n_neighbors = 10,weight=uniformWeight):
        self.n_neighbors = n_neighbors
        self.weight = weight
        #self.knn_pipe = Pipeline([('knn',KNeighborsClassifier(n_neighbors=10,metric='pyfunc',func=weighted_overlap))])
        #metric = DistanceMetric.get_metric('pyfunc', func=weighted_overlap)
        #self.knn_pipe = Pipeline([('knn',KNeighborsClassifier(n_neighbors=10,metric=metric,algorithm='brute'))])

    def fit(self,X,y):
        from direct_attribute import DirectAttributeClassifier
        self.train_X,self.feature_importance = DirectAttributeClassifier().get_feature_and_importances(X,y)
        self.train_Y = y
        print("self.train_X type: %s shape %s" % (type(self.train_X),len(self.train_X)))
        print("self.train_Y type: %s shape %s" % (type(self.train_Y),self.train_Y.shape))
        #self.knn_pipe.fit(self.train_X,self.train_Y)
        #knn = self.knn_pipe.named_steps['knn']

    def predict(self,X):
        return [self.predict_record(x) for x in X]

    def get_proba(self,X):
        pass
    def predict_record(self,x):
        k_neighbors = self.get_k_nearest_neighbors(x)
        counter= {}
        for label,distance in k_neighbors:
            if label in counter:
                counter[label] = counter[label]+self.weight(distance)
            else:
                counter[label] = self.weight(distance)
        return sorted(counter.items(),key=lambda a : a[1],reverse=True)[0][0]



    def get_k_nearest_neighbors(self,x):
        distances = zip(self.train_Y,[overlap(x,a) for a in self.train_X])
        distances.sort(key=lambda a:a[1])
        k_neighbors = distances[:self.n_neighbors]
        return k_neighbors

    def test(self,X,y,test_X,test_Y):
        from direct_attribute import DirectAttributeClassifier
        self.fit(X,y)



        test_X,emptyImportance =  DirectAttributeClassifier().get_feature_and_importances(test_X,test_Y,importance=False)
        pred = self.predict(test_X)
        #print(self.knn_pipe)
        #pred = self.knn_pipe.predict(test_X)
        print(metrics.accuracy_score(test_Y,pred))
        """
        overlap k=10
        0.675612013124
        overlap distance weighted k=10
        0.6870530832
        overlap distance weighted k=50
        0.665853453352


        overlap importance weighted k=10
        0.63994279465
        """

def main():
    from predict import get_test_data
    X,y,test_X,test_Y =get_test_data()
    print(X)
    knn = KNNClassifier(n_neighbors=50,weight=distanceWeight)
    knn.test(X,y,test_X,test_Y)

if __name__ == '__main__':
    main()

