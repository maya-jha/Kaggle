# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:19:00 2016

@author: Maya
"""
from sklearn import  svm
# Create a classifier: a support vector classifier
svmClassifier = svm.SVC(kernel='poly',degree=4)
#Preprocess this data
#This data should not be scaled probably
#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(trainXArray)
#scaler.transform(trainXArray)
#scaler.transform(trainXArray)
#The data should be in the same range. Divide by max: 255
trainXArray=trainXArray/255
testXArray=testXArray/255

svmClassifier.fit(trainXArray,trainYArray)
z=svmClassifier.predict(testXArray)
#lenOutput=z.size
imageId=np.arange(1,lenOutput+1)
#imageId
outputSub=zip(imageId,z)
#outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
#outputDf.head()
outputDf.to_csv("outputSVCDivided.csv",sep=",",index=False)