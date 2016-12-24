# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:41:13 2016

@author: Maya
"""

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(train_Y, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
grid.fit(train_X,train_Y)
z=grid.predict_proba(test_X)[:,1]
submission=DataFrame({"ID":test_ids, "TARGET":z})
submission.to_csv(fileName,sep=",",index=False