import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

data=np.genfromtxt('kmeans.txt',delimiter=' ')
k=4
modle=MiniBatchKMeans(n_clusters=k)
modle.fit(data)

centroids=modle.cluster_centers_
print(centroids)
result=modle.predict(data)
print(result)

mark=['or','ob','og','oy']          #画出每个点所属的簇
for i,d in enumerate(data):
    plt.plot(d[0],d[1],mark[result[i]])

mark=['*r','*b','*g','*y']
for i ,d in enumerate(centroids):
    plt.plot(d[0],d[1],mark[i],markersize=20)

def plot(data,modle):
    x_min,x_max=data[:,0].min()-1,data[:,0].max()+1
    y_min,y_max=data[:,1].min()-1,data[:,1].max()+1

    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                      np.arange(y_min,y_max,0.02))
    z=modle.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    cs=plt.contourf(xx,yy,z)

plot(data,modle)
plt.show()
