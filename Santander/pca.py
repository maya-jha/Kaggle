# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 10:57:31 2016

@author: Maya
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import decomposition
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
def pca_analysis(n_components_pca,trainingX,trainingY,testData):
    # a number between 0 and 1 to select using explained variance.
    #n_components_pca
    pca=decomposition.PCA(n_components=n_components_pca)
    #Normalize acroos rows i.e. by columns
    #trainingXNorm=preprocessing.normalize(trainingX,axis=0)
    #Use standard scaler as same scaler needs to be applied to testData
    std_scale = preprocessing.StandardScaler().fit(trainingX)
    trainingXNorm=std_scale.transform(trainingX)
    testXNorm=std_scale.transform(testData)
    #PCA based on covraiance matrix
    #X_train_projected=pca.fit_transform(trainingX)
    #PCA based on correlation matrix
    #Use of correlation matix is better here as the inputs are in difference scales
    X_train_projected=pca.fit_transform(trainingXNorm)
    X_test_projeced=pca.transform(testXNorm)
    explainedVariancesRatio=pca.explained_variance_ratio_
    print "No of principal components selected {0}".format(len(explainedVariancesRatio))
    #ind=np.arange(1,len(explainedVariancesRatio)+1)
    #plt.bar(ind,explainedVariancesRatio)
    classes=np.sort(np.unique(trainingY))
    labels = ["Satisfied customers", "Unsatisfied customers"]
    #visualizePCA(X_train_projected,trainingY,pca,classes,labels)
    return (X_train_projected,X_test_projeced)
def visualizePCA(X_train_projected,trainingY,pca,classes,labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    colors = [(0.0, 0.63, 0.69), 'black']
    markers = ["o", "D"]
    for class_ix, marker, color, label in zip(
            classes, markers, colors, labels):
        ax.scatter(X_train_projected[np.where(trainingY == class_ix), 0],
                   X_train_projected[np.where(trainingY == class_ix), 1],
                   marker=marker, color=color, edgecolor='whitesmoke',
                   linewidth='1', alpha=0.9, label=label)
        ax.legend(loc='best')
    plt.title(
        "Scatter plot of the training data examples projected on the "
        "2 first principal components")
    plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[0] * 100.0))
    plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[1] * 100.0))
    plt.show()

    #plt.savefig("pca.pdf", format='pdf')
    #plt.savefig("pca.png", format='png')
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
def classifyLogReg(train_X,train_Y,test_X,test_ids,fileName):
    CValues = np.logspace(-4, 4, 10)    
    logReg=linear_model.LogisticRegressionCV(Cs=CValues)
    logReg.fit(train_X,train_Y)
    z=logReg.predict_proba(test_X)[:,1]
    submission=DataFrame({"ID":test_ids, "TARGET":z})
    submission.to_csv(fileName,sep=",",index=False)
def classifySVM(train_X,train_Y,test_X,test_ids,fileName):
    #CValues = np.logspace(-4, 4, 10)    
    svcClf=SVC(probability=True)
    svcClf.fit(train_X,train_Y)
    z=svcClf.predict_proba(test_X)[:,1]
    submission=DataFrame({"ID":test_ids, "TARGET":z})
    submission.to_csv(fileName,sep=",",index=False)
if __name__=='__main__':
    trainingDataOrig=pd.read_csv('train.csv')
    testDataOrig=pd.read_csv('test.csv')
    trainingDataProcessed,testDataProcessed=preProcessData(trainingDataOrig,testDataOrig)
    trainingX=trainingDataProcessed.drop(['ID','TARGET'],axis=1)
    testData=testDataProcessed.drop(['ID'],axis=1)
    trainingY=trainingDataProcessed['TARGET']
    n_components_pca=0.95
    X_train_projected,X_test_projeced=pca_analysis(n_components_pca,trainingX,trainingY,testData)
    #classifyLogReg(X_train_projected,trainingY,X_test_projeced,testDataProcessed.ID,'LogReg.csv')
    #classifySVM(X_train_projected,trainingY,X_test_projeced,testDataProcessed.ID,'SVM.csv')
    #linSVCClf=LinearSVC()
    #linSVCClf.fit(X_train_projected,trainingY)
    svcClf=SVC(kernel='linear',probability=True)
    svcClf.fit(X_train_projected,trainingY)
    
    
    
   

