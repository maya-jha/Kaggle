# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:48:21 2016

@author: Maya
"""

#from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
#pca=decomposition.PCA()
forest = RandomForestClassifier()
pipe = Pipeline(steps=[('forest', forest)])
n_components = [20, 40, 64]
n_estimatorsRForest = [100,500,1500]
# Create the random forest object which will include all the parameters
# for the fit
estimatorRF = GridSearchCV(pipe,
                         dict(forest__n_estimators=n_estimatorsRForest))
estimatorRF.fit(trainXArray,trainYArray)
zRF=estimatorRF.predict(testXArray)
lenOutput=zRF.size
imageId=np.arange(1,lenOutput+1)
imageId
outputSub=zip(imageId,zRF)
outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
outputDf.head()
outputDf.to_csv("outputRandForestEstimator.csv",sep=",",index=False)