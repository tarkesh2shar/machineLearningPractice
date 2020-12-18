# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:41:31 2019

@author: tarkesh2shar
"""

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
xdataFrame=dataset.iloc[:,1:];

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


#feature scaling here ?




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


fig_size = plt.gcf().get_size_inches() #Get current size
sizefactor = 2 #Set a zoom factor
# Modify the current size by the factor
plt.gcf().set_size_inches(sizefactor * fig_size) 


"what each cluster signifies ??"
labels=kmeans.labels_
import seaborn as sns
clusters=pd.concat([xdataFrame, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()
for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)





"Dimensionality Reduction here !!"


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Y=sc.fit_transform(X)

#Applying pca to reduce dimensions of Y!!

from sklearn.decomposition import PCA

pca=PCA(n_components=None)


Ypca=pca.fit_transform(Y)

explained_variance=pca.explained_variance_ratio_
"""
q=pca.explained_variance_
w=pca.components_
"""

pca2=PCA(n_components=2)
Ypca2=pca2.fit_transform(Y)

pca3=PCA(n_components=3)
Ypca3=pca3.fit_transform(Y)


fig_size = plt.gcf().get_size_inches() #Get current size
sizefactor = 2 #Set a zoom factor
# Modify the current size by the factor
plt.gcf().set_size_inches(sizefactor * fig_size) 



plt.scatter(Ypca2[y_kmeans == 0, 0], Ypca2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Rich people ')
plt.scatter(Ypca2[y_kmeans == 1, 0], Ypca2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Ypca2[y_kmeans == 2, 0], Ypca2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Ypca2[y_kmeans == 3, 0], Ypca2[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(Ypca2[y_kmeans == 4, 0], Ypca2[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(Ypca2[y_kmeans == 5, 0], Ypca2[y_kmeans == 5, 1], s = 100, c = 'orange', label = 'Cluster 6')

#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()

from mpl_toolkits import mplot3d


fig = plt.figure()
ax = plt.axes(projection='3d')


fig_size = plt.gcf().get_size_inches() #Get current size
sizefactor = 2 #Set a zoom factor
# Modify the current size by the factor
plt.gcf().set_size_inches(sizefactor * fig_size) 

# Data for a three-dimensional line

"""
3d graph here
"""
ax.scatter3D(Ypca3[y_kmeans == 0, 0], Ypca3[y_kmeans == 0, 1], Ypca3[y_kmeans == 0, 2],'red')
ax.scatter3D(Ypca3[y_kmeans == 1, 0], Ypca3[y_kmeans == 1, 1], Ypca3[y_kmeans == 1, 2], 'blue')
ax.scatter3D(Ypca3[y_kmeans == 2, 0], Ypca3[y_kmeans == 2, 1], Ypca3[y_kmeans == 2, 2], 'green')
ax.scatter3D(Ypca3[y_kmeans == 3, 0], Ypca3[y_kmeans == 3, 1], Ypca3[y_kmeans == 3, 2], 'cyan')

ax.scatter3D(Ypca3[y_kmeans == 4, 0], Ypca3[y_kmeans == 4, 1], Ypca3[y_kmeans == 4, 2],'magenta')
ax.scatter3D(Ypca3[y_kmeans == 5, 0], Ypca3[y_kmeans == 5, 1], Ypca3[y_kmeans == 5, 2], 'orange')

#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('pc1')
plt.ylabel('pc2')

plt.legend()
plt.show()












"""
next step dimensionsReduction voila balle balle wow woww

reduce the dimensions of x from 16 to 2 holyshit 
and cluster mark in 2d holy double fucking shit

"""








#hierarchicle Clustering 
#need to create Dendogram
"""
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

"""
import seaborn as sns
dataset["cluster"] = y_kmeans
cols = list(dataset.columns)
cols.remove("CUST_ID")

sns.pairplot( dataset[ cols ], hue="cluster")

"""

