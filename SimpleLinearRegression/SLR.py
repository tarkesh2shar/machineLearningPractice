# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 03:49:14 2019

@author: tarkesh2shar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Salary_Data.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 1/3,random_state=0)

#simple linear regression doesnt require scaling now so enjoy this life dont stress!!
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)

#predicting test set results========> and compare in variables?
    
Y_pred=regressor.predict(X_test)  

#Plot our observations 

plt.scatter(X_train,Y_train,color="red")

#plot regression Line
#plt.plot(X_train,Y_train,color="green")

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

###########################################

plt.scatter(X_test,Y_test,color="red")

#plot regression Line
#plt.plot(X_train,Y_train,color="green")

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()




