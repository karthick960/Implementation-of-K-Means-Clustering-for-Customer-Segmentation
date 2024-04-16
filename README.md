# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Pick customer segment quantity (k)
2. Seed cluster centers with random data points.
3. Assign customers to closest centers.
4. Re-center clusters and repeat until stable.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: KARTHICK K
RegisterNumber: 212222040070
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()

k=5
kmeans = KMeans(n_clusters=k)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroid:")
print(centroids)
print("Labels:")
print(labels)

colors =['r','g','b','c','m']
for i in range(k):
  cluster_points =x[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],color=colors[i], label = f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
  
plt.scatter(centroids[:,0], centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show
```
## Output:
![image](https://github.com/karthick960/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121215938/4544a008-ba88-4bae-89ab-a704f234635d)
![image](https://github.com/karthick960/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121215938/7fdfe46b-0fb3-46fa-8e5f-85df010ac812)
![image](https://github.com/karthick960/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121215938/b1cfd0f1-e02c-4365-a806-3d79baca2beb)
![image](https://github.com/karthick960/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121215938/a7da6a85-e417-482d-96dd-c83a19e4abb8)
![image](https://github.com/karthick960/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/121215938/3dc799ef-3f92-4035-a381-97a2cbc415ff)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
