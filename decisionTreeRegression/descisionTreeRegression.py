# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:14:34 2019

@author: tarkesh2shar
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt


#preferable in higher !! dimension !!!

dataset=pd.read_csv("Position_Salaries.csv");

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values



#Create a regressor

from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)


regressor.fit(X,y)

#non continous regresson model
y_pred=regressor.predict([[6.5]])

# Visualising the Regression results

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()












                   




