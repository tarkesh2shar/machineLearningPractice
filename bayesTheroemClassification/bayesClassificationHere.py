# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:49:46 2019

@author: tarkesh2shar
"""
import matplotlib.pyplot as plt;
import pandas as pd;
import numpy as np;

dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values


#Split the data Now

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


#fEATURE sCALING hERE nOW//

from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

X_train=scalerx.fit_transform(X_train)
X_test=scalerx.fit_transform(X_test)

###############tIME fOR A cLASSIFIER HERE nOW################

from  sklearn.naive_bayes import GaussianNB

classifier=GaussianNB();

classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)



from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#mng = plt.get_current_fig_manager()

#mng.frame.Maximize(True)


plt.figure()
plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode

plt.show()





