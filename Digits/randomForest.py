# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:10:01 2016

@author: Maya
"""

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(trainXArray,trainYArray)
z=forest.predict(testXArray)
lenOutput=z.size
imageId=np.arange(1,lenOutput+1)
imageId
outputSub=zip(imageId,z)
outputSub
outputDf=pd.DataFrame(outputSub,columns=['ImageId','Label'])
outputDf.head()
outputDf.to_csv("outputRandForest.csv",sep=",",index=False)