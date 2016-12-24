# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 17:40:18 2016

@author: Maya
"""

from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
pca=decomposition.PCA()
forest = RandomForestClassifier()
pipe = Pipeline(steps=[('pca', pca), ('forest', forest)])
n_components = [20, 40, 64]
n_estimatorsRForest = [100,300,500]
# Create the random forest object which will include all the parameters
# for the fit
estimatorRFPCA = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              forest__n_estimators=n_estimatorsRForest))
estimatorRFPCA.fit(trainXArray,trainYArray)
zRFPCA=estimatorRFPCA.predict(testXArray)
lenOutput=zRFPCA.size
imageId=np.arange(1,lenOutput+1)
imageId
outputSub=zip(imageId,zRFPCA)
outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
outputDf.head()
outputDf.to_csv("outputPCARandForest.csv",sep=",",index=False)
