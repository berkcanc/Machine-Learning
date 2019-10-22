# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:39:48 2019

@author: berkc
"""
import pandas as pd
import re
import nltk
durma = nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
yorumlar = pd.read_csv('Restaurant_Reviews.csv')
from nltk.corpus import stopwords

#PREPROCESSING
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    #KELİME KÖKLERİNE AYIRMA(STOPWORDS HARİÇ)
    yorum = [ps.stem(kelime) 
    for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)


#FEATURE EXTRACTION
#BAG OF WORDS(BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
#INDEPENDENT VARIABLE
X = cv.fit_transform(derlem).toarray()
#DEPENDENT VARIABLE
y = yorumlar.iloc[:,1].values


#MACHINE LEARNING
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#accuracy : %72.5








