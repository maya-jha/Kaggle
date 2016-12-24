# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 15:46:32 2016

@author: Maya
"""
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from pandas import DataFrame
def preProcessData(trainingData,testData):    
    cols2Remove=[]
    #Find columns for which values are identical for all rows
    for col in trainingData.columns:    
        if trainingData[col].std()==0:        
            cols2Remove.append(col)    
    print "No of Columns before Identical Value Column removal {0}".format(trainingData.shape[1])
    trainingData.drop(cols2Remove,axis=1,inplace=True)
    testData.drop(cols2Remove,axis=1,inplace=True)
    print "No of Columns after Identical Value Column removal {0}".format(trainingData.shape[1])
    # remove duplicate columns
    cols2Remove = []
    columns=trainingData.columns    
    for i in range(len(columns)-1):
        v = trainingData[columns[i]].values
        for j in range(i+1,len(columns)):
            if np.array_equal(v,trainingData[columns[j]].values):
                cols2Remove.append(columns[j])
    trainingData.drop(cols2Remove,axis=1,inplace=True)
    testData.drop(cols2Remove,axis=1,inplace=True)
    print "No of Columns after Identical Value Column removal {0}".format(trainingData.shape[1])
    return (trainingData,testData)
if __name__=='__main__':
    trainingDataOrig=pd.read_csv('train.csv')
    testDataOrig=pd.read_csv('test.csv')    
    trainingDataProcessed,testDataProcessed=preProcessData(trainingDataOrig,testDataOrig)
    del trainingDataOrig
    del testDataOrig
    trainingX=trainingDataProcessed.drop(['ID','TARGET'],axis=1)
    testData=testDataProcessed.drop(['ID'],axis=1)
    trainingY=trainingDataProcessed['TARGET']
    trainingX=trainingDataProcessed.drop(['ID','TARGET'],axis=1)
    testData=testDataProcessed.drop(['ID'],axis=1)
    trainingY=trainingDataProcessed['TARGET']    
    test_IDS=testDataProcessed.ID
    del testDataProcessed
    del trainingDataProcessed
    forestClf= RandomForestClassifier(n_jobs = -1)    
    cv = StratifiedShuffleSplit(trainingY, n_iter=5, test_size=0.2, random_state=42)
    max_featuresRForest=['sqrt','log2',0.2]
    n_estimatorsRForest = [1000,1500,2000]
    sample_leaf_options = [5,10,50,100,200,500]
    param_grid = dict(	n_estimators=n_estimatorsRForest,max_features=max_featuresRForest, min_samples_leaf=sample_leaf_options)
    grid = GridSearchCV(forestClf, param_grid=param_grid, cv=cv)
    grid.fit(trainingX,trainingY)
    z=grid.predict_proba(testData)[:,1]
    submission=DataFrame({"ID":test_IDS, "TARGET":z})
    submission.to_csv('randForestCrossValidation.csv',sep=",",index=False)
    #Here is the best estimator parameters
    grid.best_estimator_
#    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features=0.2, max_leaf_nodes=None,
#            min_samples_leaf=5, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=-1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
    
    