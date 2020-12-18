# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:10:46 2019

@author: tarkesh2shar
"""


import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np;

#classify as dibatic or not

"""

Number of times pregnant.
Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
Diastolic blood pressure (mm Hg).
Triceps skinfold thickness (mm).
2-Hour serum insulin (mu U/ml).
Body mass index (weight in kg/(height in m)^2).
Diabetes pedigree function.
Age (years).
Class variable (0 or 1).

The baseline performance of predicting the most prevalent class
 is a classification accuracy of approximately 65%.
 Top results achieve a classification accuracy of approximately 77%.


"""

dataset=pd.read_csv("dibatese.txt",header=None)
X=dataset.iloc[:,:-1].values
Y=dataset.loc[:,[8]].values
Z=np.asarray(Y).reshape(-1)


XforGraph=dataset.iloc[:,[5,7]].values
YforGraph=dataset.loc[:,[8]].values

YforGraph=np.asarray(YforGraph).reshape(-1)


#Splitting the dataset into training and testing
    
from sklearn.model_selection import train_test_split;

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2);

X_trainforGraph,X_testforGraph,Y_trainforGraph,Y_testforGraph=train_test_split(XforGraph,YforGraph,test_size=0.2);


from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

X_train=scalerx.fit_transform(X_train)
X_test=scalerx.fit_transform(X_test)

"""
Logistic Regression

"""
from sklearn.linear_model import LogisticRegression

classfierLogistic=LogisticRegression();


classfierLogistic.fit(X_train,Y_train.ravel());

y_pred_logistic=classfierLogistic.predict(X_test);

from sklearn.metrics import confusion_matrix;

logisticConfusion=confusion_matrix(Y_test,y_pred_logistic);

# ------> 70% accurate model PASSED 65% THRESHOLD here
#---->73%accurate 27thNovember


"""""""""""""""""""""

2) ----> predicting with k nearest neighbours!!

"""""""""""""""""""""
from sklearn.neighbors import KNeighborsClassifier
classifierKnn=KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
classifierKnn.fit(X_train,Y_train.ravel())

y_pred_knn=classifierKnn.predict(X_test)

KnnConfusion=confusion_matrix(Y_test,y_pred_knn);

#with 5 neigbors 67% accuracy!
#with 10 --> 68%accurate

#---->62%accurate 27thNovember

"""""""""""
Support Vector Machine

"""""""""""
from sklearn.svm import SVC

svcClassifier=SVC(kernel="poly",degree=10);

svcClassifier.fit(X_train,Y_train.ravel());

y_pred_svc=svcClassifier.predict(X_test);

svcConfusion=confusion_matrix(Y_test,y_pred_svc);

#---->75%accurate 27thNovember rbf//
#---->57% for poly with 10degree


"""
Naive baise Classifier//

"""

from sklearn.naive_bayes import GaussianNB

naiveClassifier=GaussianNB()


naiveClassifier.fit(X_train,Y_train);

y_pred_naive=naiveClassifier.predict(X_test);

svcNaive=confusion_matrix(Y_test,y_pred_naive);
"""
----->68%correct 27th November
"""

"""
 DecisionTreeClassifier
"""

from sklearn.tree import DecisionTreeClassifier

deClassifier=DecisionTreeClassifier(criterion="entropy")


deClassifier.fit(X_train,Y_train);


ypred_Decision=deClassifier.predict(X_test);
confusionDecison=confusion_matrix(Y_test,ypred_Decision);

"""
59% 27thNovember
"""

"""
    RandomForestClassifier
    
"""
from sklearn.ensemble import RandomForestClassifier

randomforestClassifier=RandomForestClassifier(n_estimators=100)
randomforestClassifier.fit(X_train,Y_train)
ypredRandomForest=randomforestClassifier.predict(X_test)
confusionRandom=confusion_matrix(Y_test,ypredRandomForest)

"""
78% 27thNovember
"""

#graph with 2 variables here

X_trainforGraph,X_testforGraph,Y_trainforGraph,Y_testforGraph=train_test_split(XforGraph,YforGraph,test_size=0.2);
scalerxforGraph=StandardScaler();
X_trainforGraph=scalerxforGraph.fit_transform(X_trainforGraph);
X_testforGraph=scalerxforGraph.fit_transform(X_testforGraph);

classfierLogisticForGraph=LogisticRegression();


classfierLogisticForGraph.fit(X_trainforGraph,Y_trainforGraph.ravel());

y_pred_logisticForGraph=classfierLogisticForGraph.predict(X_testforGraph);



logisticConfusionForGraph=confusion_matrix(Y_testforGraph,y_pred_logisticForGraph);

#50%accurate here


#knnnnn

classifierKnnforGraph=KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
classifierKnnforGraph.fit(X_trainforGraph,Y_trainforGraph.ravel())

y_pred_knn_graph=classifierKnnforGraph.predict(X_testforGraph)

KnnConfusiongraph=confusion_matrix(Y_testforGraph,y_pred_knn_graph);


#svc

classifiersvcforGraph=SVC(kernel="poly",degree=6)
classifiersvcforGraph.fit(X_trainforGraph,Y_trainforGraph.ravel())

y_pred_SVC_graph=classifiersvcforGraph.predict(X_testforGraph)

SVCConfusiongraph=confusion_matrix(Y_testforGraph,y_pred_SVC_graph);


#naive


naiveClassifierGraph=GaussianNB()


naiveClassifierGraph.fit(X_trainforGraph,Y_trainforGraph.ravel());

y_pred_naive_graph=naiveClassifierGraph.predict(X_testforGraph);

svcNaiveGraph=confusion_matrix(Y_testforGraph,y_pred_naive_graph);


#decisionTree

deClassifierGraph=DecisionTreeClassifier(criterion="entropy")


deClassifierGraph.fit(X_trainforGraph,Y_trainforGraph.ravel());


ypred_Decision_Graph=deClassifierGraph.predict(X_testforGraph);
confusionDecisonGraph=confusion_matrix(Y_testforGraph,ypred_Decision_Graph);

#randomForest

randomforestClassifierGraph=RandomForestClassifier(n_estimators=100)
randomforestClassifierGraph.fit(X_trainforGraph,Y_trainforGraph.ravel())
ypredRandomForestGraph=randomforestClassifierGraph.predict(X_testforGraph)
confusionRandomGraph=confusion_matrix(Y_testforGraph,ypredRandomForestGraph)


#------->for logisitc graph<-------
from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classfierLogisticForGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()



#knn

from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifierKnnforGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()


#svc



from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifiersvcforGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()


"""
naiveClassifierGraph
"""

from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, naiveClassifierGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()


"""
deClassifierGraph
"""

from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, deClassifierGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()


"""
randomforestClassifierGraph
"""


from matplotlib.colors import ListedColormap
X_set, y_set = X_testforGraph, Y_testforGraph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, randomforestClassifierGraph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
  
"""
plt.scatter(X_set[:,[0]],X_set[:,[1]],)
"""
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()












