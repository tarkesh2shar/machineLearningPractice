# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:20:28 2019
@author: tarkesh2shar
"""

# -------------------------------><-------------------------------

#   .............>y=b0 +b1x1 +b2x 1^2<..............


import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
dataset=pd.read_csv("Position_Salaries.csv");


#vector or matrix?? 

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split

#X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state=0)

#feature scaling ????????????

#Fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression;

linearregressor=LinearRegression();

linearregressor.fit(X,Y);

#fitting polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=4)

X_poly=poly_reg.fit_transform(X)


linreg2=LinearRegression()

linreg2.fit(X_poly,Y);

#Visulizing both models!!

plt.scatter(X,Y,color="red");
plt.plot(X,linearregressor.predict(X),color="blue")
plt.title("Truth or Bluff (Linear)")

plt.xlabel("Position Label");
plt.ylabel("Salary");
plt.show();

#!!!!!!!!!!regression Polynomial Here Enjoy!!!!!!!!!!!!!
#making more curvy!!!!
X_grid=np.arange(min(X),max(X),0.1)  #vector
X_grid=X_grid.reshape((len(X_grid), 1))  #vector to matrix

plt.scatter(X,Y,color="pink");
plt.plot(X,linreg2.predict(poly_reg.fit_transform(X)),color="green")
plt.title("Truth or Bluff (Polynomial)")
plt.xlabel("Position Label");
plt.ylabel("Salary");
plt.show();


#employee bluffing?
#predicting with linear model
linearregressor.predict([[6.5]])

#predicting with polynomial model Honesty !!!!!!! woww !!!!!!!!

linreg2.predict(poly_reg.fit_transform([[6.5]]))























