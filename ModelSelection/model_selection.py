
import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

dataset=pd.read_csv("Social_Network_Ads.csv");

X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values



from sklearn.model_selection import train_test_split;

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.25)


#scaling is Required here

from sklearn.preprocessing import StandardScaler

scalerx=StandardScaler();

X_train=scalerx.fit_transform(X_train)
X_test=scalerx.fit_transform(X_test)


#Here the classifier

#!!!!!!!!!!!!!!!Kernal Tricks!!!!!!!!!!!!!!!
"""

k(x,l)= e^-(||x-l ||^2/2sigma^2)

"""

from sklearn.svm import SVC

classifier=SVC(kernel="rbf",random_state=0);
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,y_pred);


#Applying k-fold cross validation

from sklearn.model_selection  import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)

accuracies.mean()
accuracies.std()







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