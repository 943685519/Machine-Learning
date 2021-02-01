from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('kmeans.txt',delimiter=' ')
k=4
modle=KMeans(n_clusters=4)
modle.fit(data)

centers=modle.cluster_centers_
print(centers)

result=modle.predict(data)
print(result)


mark=['or','ob','og','oy']
for i,d in enumerate(data):
    plt.plot(d[0],d[1],mark[result[i]])

mark=['*r','*b','*g','*y']
for i,d in enumerate(centers):
    plt.plot(d[0],d[1],mark[i],markersize=20)

def plot(data,modle):
    #两个特征的数据构成的XOY坐标系（取值范围）
    x_min,x_max=data[:,0].min()-1,data[:,0].max()+1
    y_min,y_max=data[:,1].min()-1,data[:,1].max()+1
    #生成网格矩阵,x_min,x_max为x坐标轴方向，arange间隔0。02生成等差数列，y轴同理
    # a = np.array([1, 2, 3])
    # b = np.array([7, 8])
    # res = np.meshgrid(a, b)
    # 返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                      np.arange(y_min,y_max,0.02))
    #求z轴，在XOY坐标系中的所有点，对每个点进行z的预测，那么就得到了等高面z，
    #ravel()与flatten()类似，多维数据转一维。flatten不改变原数据，ravel改变
    #np.c_[]是按colum组合两个数组，可以得到XOY中所有的点坐标
    z=modle.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    #等高线图
    cs=plt.contourf(xx,yy,z)

plot(data,modle)
plt.show()