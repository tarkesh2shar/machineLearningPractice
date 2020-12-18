# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 02:58:07 2019

@author: tarkesh2shar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


#plot dendogram

import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("dendogram")
plt.xlabel("Customeres")
plt.ylabel("Euclidean distances")
plt.show()

#Fitting clusturing

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5);

y_hc=hc.fit_predict(X);

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

#plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
