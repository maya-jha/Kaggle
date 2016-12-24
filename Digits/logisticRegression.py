# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:03:47 2016

@author: Maya
"""
import pandas as pd
import numpy as np
import sklearn
import sklearn.datasets as datasets
trainData=pd.read_csv('train.csv')
trainData.head()
trainX=trainData.drop('label',axis=1)
trainY=trainData['label']
trainXArray=trainX.values
trainYArray=trainY.values
testData=pd.read_csv('test.csv')
testXArray=testData.values
from sklearn import linear_model
Cs = np.logspace(-4, 4, 10)
logReg=linear_model.LogisticRegressionCV(C=Cs)
logReg.fit(trainXArray,trainYArray)
z=logReg.predict(testXArray)
lenOutput=z.size
imageId=np.arange(1,lenOutput+1)
imageId
outputSub=zip(imageId,z)
outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
outputDf.head()
outputDf.to_csv("output.csv",sep=",",index=False)