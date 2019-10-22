# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:06:23 2019

@author: berkc
"""

import pandas as pd

#Veri okuma ==================================================================
veriler = pd.read_csv('Churn_Modelling.csv')

#Eksik veriler ==============================================================
X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values

#Encoder : Kategorik -> Numeric =========================================
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
X[:,1] =le.fit_transform(X[:,1]) 

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])
print(X)

ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

#Verilerin test-train olarak ayrilmasi ====================================== 

from sklearn.model_selection._split import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)







