# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:41:22 2019

@author: berkc
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values


# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("Linear R2 Score : ")
print(r2_score(Y,lin_reg.predict(X)))


# Polynomial Regression Degree 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)



# Polynomial Regression Degree 4
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


# Visualization
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X))
plt.show()

plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)))
plt.show()

plt.scatter(X,Y)
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)))
plt.show()


print(lin_reg.predict([[10]]))
print(lin_reg2.predict(poly_reg2.fit_transform([[11]])))

print("Polynomial R2 Score : ")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))))


# =============================================================================
# SVR
# =============================================================================

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


from sklearn.svm import SVR
svr_linear = SVR(kernel='linear')
svr_linear.fit(x_olcekli,y_olcekli)

svr_poly = SVR(kernel='poly')
svr_poly.fit(x_olcekli,y_olcekli)

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(x_olcekli,y_olcekli)


plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_linear.predict(x_olcekli))

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_poly.predict(x_olcekli))

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_rbf.predict(x_olcekli))
plt.show()

print(svr_linear.predict([[10]]))
print(svr_poly.predict([[10]]))
print(svr_rbf.predict([[10]]))

print("SVR RBF R2 Score : ")
print(r2_score(y_olcekli,svr_rbf.predict(x_olcekli)))

# =============================================================================
# DECISION TREE
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,r_dt.predict(X))
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print("Decision Tree R2 Score : ")
print(r2_score(Y,r_dt.predict(X)))


# =============================================================================
# RANDOM FOREST
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color = 'red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.show()

print("Random Forest R2 Score : ")
print(r2_score(Y,rf_reg.predict(X)))





















