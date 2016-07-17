__author__ = 'phx'

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def grid_CV_Search(clfs,X_train,Y_train,X_test,Y_test,cv=5):
    print("grid searching")
    for clf, param_grid in clfs:
        print("_"*80)
        print(clf)
        grid_clf =GridSearchCV(clf,param_grid,cv=cv)
        grid_clf.fit(X_train,Y_train)
        print("Best parameters found on training set:")
        print(grid_clf.best_params_)
        for params,mean_score,scores in grid_clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score,scores.std()*2,params))
        Y_pred = grid_clf.predict(X_test)
        print("Detailed classification report:")
        print(classification_report(Y_test,Y_pred))