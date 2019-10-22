# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:16:48 2019

@author: berkc
"""

import pandas as pd
from sklearn.metrics import confusion_matrix


data = pd.read_csv('veriler.csv')

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,
                                                       test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# =============================================================================
# Classification Algorithms
# =============================================================================
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred_logr = logr.predict(X_test)
print(y_pred_logr)
print("----------")
print(y_test)


cm = confusion_matrix(y_test,y_pred_logr)
print(cm)


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test,y_pred_knn)
print("KNN Conf. Matrix")
print(cm_knn)


# SVC
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test,y_pred_svc)
print("SVC Conf. Matrix")
print(cm_svc)


# GNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred_gnb = gnb.predict(X_test)
cm_gnb = confusion_matrix(y_test,y_pred_gnb)
print("GNB Conf. Matrix")
print(cm_gnb)


# BNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)

y_pred_bnb = bnb.predict(X_test)
cm_bnb = confusion_matrix(y_test,y_pred_gnb)
print("BNB Conf. Matrix")
print(cm_bnb)


# DT Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)

y_pred_dtc = dtc.predict(X_test)
cm_dtc = confusion_matrix(y_test,y_pred_dtc)
print("DTC Conf. Matrix")
print(cm_dtc)


# RF Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='entropy',n_estimators=100)
rfc.fit(X_train,y_train)

y_pred_rfc = rfc.predict(X_test)
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
print("RFC Conf. Matrix")
print(cm_rfc)




#ROC
y_proba = rfc.predict_proba(X_test)
print(y_proba[:,0])
print(y_test)

from sklearn import metrics
fpr,tpr,thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(metrics.auc(fpr,tpr))
















