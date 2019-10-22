# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:00:51 2019

@author: berkc
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')

X = data.iloc[:,3:].values



#KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init = 'k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init = 'k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4,init = 'k-means++',random_state=123)
y_tahmin2=kmeans.fit_predict(X)
print(y_tahmin2)
plt.scatter(X[y_tahmin2==0,0],X[y_tahmin2==0,1],s=100,c='red')
plt.scatter(X[y_tahmin2==1,0],X[y_tahmin2==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin2==2,0],X[y_tahmin2==2,1],s=100,c='green')
plt.scatter(X[y_tahmin2==3,0],X[y_tahmin2==3,1],s=100,c='yellow')

plt.title("KMeans")
plt.show()

#Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward")
y_tahmin = ac.fit_predict(X)
print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='yellow')
plt.title("HC")
plt.show()



import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

























