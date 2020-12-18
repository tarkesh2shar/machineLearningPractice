# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:27:14 2019

@author: tarkesh2shar
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset here now , dont be shy!!

dataset=pd.read_csv("Data.csv")

#independent columnn matrix

#CONVERT TO ARRAY!
X=dataset.iloc[:,:-1].values

#DEPENDENT VALUE VECTOR

Y=dataset.iloc[:,3].values



#HANDLING NAN DATA!! 
#preprocess anydataSet!!
from sklearn.preprocessing import Imputer 

#from sklearn import SimpleImputer as Imputer

#import impute.SimpleImputer from sklearn as Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(X[:,1:3]) #UPPER BOUND INCLUDED

X[:,1:3]=imputer.transform(X[:,1:3])

'''
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='')
imp_mean= imp_mean.fit(X[:,1:3])
X[:,1:3]=imp_mean.transform(X[:,1:3])

'''

#INCODING TEXT TO NUMBER CATEGORICAL DATA--->


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X=LabelEncoder()
#labelencoder_Y=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])#which column onehotencode

#Y[:,0]=labelencoder_Y.fit_transform(Y[:,0])
X=onehotencoder.fit_transform(X).toarray()

#problem still persists , 2>1 check the X matrix!!

labelencoder_Y=LabelEncoder()
#labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

'''
 uPDATE hERE

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

# Importing the libraries
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
'''



# SPLITTING THE DATA SET INTO TRAINING AND TEST SET !!! WOWW SO WONDERFUL!!!




#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state=0)


#putting variable into same Scale!!

#2ways here ---->1 Standardisation ----> 2 Normalisation

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler();

X_train=sc_X.fit_transform(X_train)

#sc_X=StandardScaler();

X_test=sc_X.transform(X_test)









