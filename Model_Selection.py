# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:08:57 2019

@author: berkc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


#K-field cross validation
from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=4)
print(basari.mean())
print(basari.std())

#parameter optimization
from sklearn.model_selection import GridSearchCV
p = [ {'C':[1,2,3,4,5],'kernel':['linear']},
      {'C':[1,10,100,1000],'kernel':['rbf'],
       'gamma':[1,0.5,0.1,0.01,0.001]} ]
'''
estimator  : classification algorithm
param_grid : parameters
scoring    : depend on scoring for.ex : accuracy
cv         : how many folds 
n_jobs     : jobs on the same time
'''
gs = GridSearchCV(estimator=classifier,param_grid=p,
                  scoring='accuracy',cv=10,n_jobs=-1)
grid_search = gs.fit(X_train,y_train)
best_score  = grid_search.best_score_
best_parameter = grid_search.best_params_
print(best_score)
print(best_parameter)















