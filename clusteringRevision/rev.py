# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:31:30 2019

@author: tarkesh2shar
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

dataset=pd.read_csv("D.csv")

X=dataset.iloc[:,1:].values

#Y=dataset.iloc[:,-1].values
#z=dataset.iloc[:,:].values

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
"""

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean= imp_mean.fit(X)
X=imp_mean.transform(X)



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show();

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


#hierarchicle Clustering 
#need to create Dendogram

import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method="ward"))

plt.title("Dendogram")
plt.show()


#optimum clusters=3

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")

y_hc=hc.fit_predict(X)



#So we know that we have 3 solid categories in the dataset so whats next ?
#maybe dimesional reduction will provide more answers 


"""
next step dimensionsReduction voila balle balle wow woww
"""

"""
import seaborn as sns
dataset["cluster"] = y_kmeans
cols = list(dataset.columns)
cols.remove("CUST_ID")

sns.pairplot( dataset[ cols ], hue="cluster")

"""
"""
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
"""
