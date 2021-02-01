from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#第一个数据
data=np.genfromtxt('kmeans.txt',delimiter=' ')
modle=DBSCAN(eps=1.5,min_samples=4)
modle.fit(data)
result=modle.fit_predict(data)
print(result)
mark=['or','ob','og','oy','ok','om']
for i,d in enumerate(data):
    plt.plot(d[0],d[1],mark[result[i]])
plt.show()

#第二个数据
x1,y1=datasets.make_circles(n_samples=2000,noise=0.05,factor=0.5)       #生成一内圈一外圈，factor ：外圈与内圈的尺度因子<1
x2,y2=datasets.make_blobs(n_samples=100,centers=[[1.2,1.2]],cluster_std=0.1)
x=np.concatenate((x1,x2))
plt.scatter(x[:,0],x[:,1],marker='o')
plt.show()

from sklearn.cluster import KMeans
modle=KMeans(n_clusters=3)
y_pred=modle.fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()

modle=DBSCAN(eps=0.2)
y_pred=modle.fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()