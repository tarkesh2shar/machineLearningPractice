# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:53:55 2019

@author: tarkesh2shar
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

dataset=pd.read_csv("Mall_Customers.csv")

X=dataset.iloc[:,[1,2,3]];
Z=dataset.iloc[:,1:4].values;
Y=dataset.iloc[:,-1].values;


# we need to do something for the  CATEGORICAL dATA//


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data

"""
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
"""

#AVOIDING INDEPENDENT VARIABLE tRAP DROP 1 COLUMM NOW

X=X[:,[1,2,3]]; # a+bx+cd+....

"After backward elimintaion we can take out variable nume"


#nOW OUR MATRIX OF INDEPENDET MATRIX IS rEady , dependent vector is also Ready!!!!

from  sklearn.model_selection  import train_test_split;
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0);


# now time for all the Regression Here

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train);

#an awful test result for multiple linear Result here
predYMultipleLinear=regressor.predict(X_test);



#Checking with all other models 

#PolynomialRegression

#Better result than multiple Linear Regression //
from sklearn.preprocessing import PolynomialFeatures
poly_reg =PolynomialFeatures(degree=3)

X_poly=poly_reg.fit_transform(X_train)

polyreg=LinearRegression()
polyreg.fit(X_poly,Y_train)

predYPolynomailRegrssion=polyreg.predict(poly_reg.fit_transform(X_test))


#svr Regression Here!!


from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

XScaled=scalerx.fit_transform(X_train)


scalery=StandardScaler();


scaledY=np.reshape(Y_train,(-1,1))

YScaled=scalery.fit_transform(scaledY);


from sklearn.svm import SVR

svrRegressor=SVR()

svrRegressor.fit(XScaled,YScaled.ravel());


predYSVR=y_pred_svr = scalery.inverse_transform(
        
        svrRegressor.predict( 
                
                scalerx.transform(X_test)
                
                )
        
        
        )


from sklearn.tree import DecisionTreeRegressor

regressorDecision=DecisionTreeRegressor(random_state=0)


regressorDecision.fit(X_train,Y_train)

#non continous regresson model
y_pred_decision_tree=regressorDecision.predict(X_test)


from sklearn.ensemble import RandomForestRegressor

regressorForest=RandomForestRegressor(n_estimators=100)

regressorForest.fit(X_train,Y_train)

#non continous regresson model
y_pred_forest=regressorForest.predict(X_test)




#Now lets try to make our model better here by Backward Elimintaion  !!

 import statsmodels.api as sm;

 import statsmodels.regression.linear_model as sm;
 
 # Y= a0b0+a1b1+a2b2.....+anbn
 
 # putting a0 =1; Y=B0
 
# X=np.append(arr=X,values=np.ones((50,1)).astype(int),axis=1)
 #Putting 1 extra colummn for computing and taking care of constant b0;
  X=np.append(arr=np.ones((200,1)).astype(int),values=X,axis=1)
  
  X_opt=X[:,[0,1,2,3]]
  
  
  """
     Step 1 -->
  """
 
  # select a significance level to stay in the model (eg SL=0.05);
  
  #5%;
  """
    Step 2---> Fit the full model with all possible predictors!
  """
  regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
regressor_OLS.summary()

 """"""""""Again"""""""""""""""""""""
  """ Remove the independent variable with highest p value """
  
    X_opt=X[:,[0,1,2]]
    
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
regressor_OLS.summary();


 X_opt=X[:,[0,2]]
    
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
regressor_OLS.summary()
  

#  All right lets check our results with new Model 

      X_opt=X[:,[0,1,2]]
      
      regressor.fit(X_opt,Y_train);

#an awful test result for multiple linear Result here
predYAfterElimintaion=regressor.predict(X_test);
    
  
  




