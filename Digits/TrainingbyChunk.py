# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 16:46:03 2016

@author: Maya
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier()
classes=np.arange(0,10)

n_iter = 50
for n in range(n_iter):
    trainDataReader=pd.read_csv('train.csv',chunksize=1000)
    for chunk in trainDataReader:
        trainX=chunk.drop('label',axis=1)
        trainY=chunk['label']    
        clf.partial_fit(trainX,trainY,classes)
    
    
testData=pd.read_csv('test.csv')
testXArray=testData.values

z=clf.predict(testXArray)
lenOutput=z.size
imageId=np.arange(1,lenOutput+1)
#imageId
outputSub=zip(imageId,z)
#outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
#outputDf.head()
outputDf.to_csv("outputSGDClassifier.csv",sep=",",index=False)
    