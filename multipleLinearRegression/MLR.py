# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 06:51:42 2019

@author: tarkesh2shar
"""


#How to make a model !! Yikes!!



"""
All in  (!!throw in all variables!!)
Backward Elimination ()---->---------------------
Forward Selection---->  <---  STEP WISE ELEMINTAION -->
Biderectional eLIMINATION---->-------------------
ScoRE cOMPARIASION
"""

"""
2nd --> Backward Elimination Method <--
select a significance model 
fit the model with all possible predictors
Consider the predictor with the highest p value
       IF P>SL GO TO STEP 4 eLSE MODEL OUTPUT
Remove the predictor
Fit the model without this variable 
GO TO 3


"""

"""
3RD---->Forward Selection  
1) select SL=0.05,

2) fit all simple regression models  y=mx ,
    select the 1 with lowest p value
    
3) Keep this variable and fit all possible model with one
extra predictor added you already have

4)Consider the predictor with the lowest p value
    if P<SL go to step3.otherwise go to finish

5) KEEP THE PREVIOUS MODEL!!!



"""

##GET THE DATA##

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")
##MAKING INDEPENDENT AND DEPENDENT VARIABLES NOW!!
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,4]

## WE NEED TO TAKE CARE OF THE CATEGORICAL DATA

# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#   Avoiding the dummy varaibale Trap

        X=X[:,1:] 
        
        ### BUT THIS WILL BE TAKEN CARE BY THE PHYTHON LIBRARIES!! ###
        


#Spliting the dataset into training and testing purporses
        
 from sklearn.model_selection import train_test_split
 
 X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
 
 
 
 
 #simple linear regression
 
 from sklearn.linear_model import LinearRegression
 
 regressor=LinearRegression()
 
 regressor.fit(X_train,Y_train); # fitting the value here now !!
 
 
 #testing here now
 
 y_pred=regressor.predict(X_test) #predicting the value on 10 test x
 
 
 # backward eleimination 
 
 import statsmodels.api as sm;

 import statsmodels.regression.linear_model as sm;
 
 # Y= a0b0+a1b1+a2b2.....+anbn
 
 # putting a0 =1; Y=B0
 
# X=np.append(arr=X,values=np.ones((50,1)).astype(int),axis=1)
 
  X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
  
  X_opt=X[:,[0,1,2,3,4,5]]
  
   
  """
     Step 1 -->
  """
 
  # select a significance level to stay in the model (eg SL=0.05);
  
  #5%;
  """
     Step 2---> Fit the full model with all possible predictors!
  """
 regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
  """
     Step 3--->Look for the predictor with highest P value!
     
  """
  
  regressor_OLS.summary()
  
  """"""""""Again"""""""""""""""""""""
  """ Remove the independent variable with highest p value """
  
   X_opt=X[:,[0,1,3,4,5]]
  
   
 
    
 regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
  """
     Step 3--->Look for the predictor with highest P value!
     
  """
  
  regressor_OLS.summary()
  
  
  ###################################################################
    
   X_opt=X[:,[0,3,4,5]]
  
   
 
    
 regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
  
  """
     Step 3--->Look for the predictor with highest P value!
     
  """
  
  regressor_OLS.summary()
  
  
  #####################################################################
  
    
   X_opt=X[:,[0,3,5]]
   
 regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
 
 
  regressor_OLS.summary()
  
  ##################################################################
  
    X_opt=X[:,[0,3]]
    
     regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
 
 
  regressor_OLS.summary()
  
   """
  import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""
#    Automatic eliminations
"""

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""
  
  
  

  
  
      
  
  
 
 
 
 
 
 
 
 








