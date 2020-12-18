# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:56:35 2019

@author: tarkesh2shar
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

dataset=pd.read_csv("Position_Salaries.csv");

X=dataset.iloc[:,1:2].values

y=dataset.iloc[:,2].values



from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

X=scalerx.fit_transform(X)


scalery=StandardScaler();


scaledY=np.reshape(y,(-1,1))
y=scalery.fit_transform(scaledY);




from sklearn.svm import SVR;
regressor=SVR(kernel="rbf");
regressor.fit(X,y.ravel());



y_pred = scalery.inverse_transform(
        
        regressor.predict( 
                
                scalerx.transform([[6.5]])
                
                )
        
        
        )

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()









