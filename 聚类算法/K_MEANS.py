import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')
data=np.genfromtxt('E:\机器学习\聚类算法\kmeans.txt',delimiter=' ')
# plt.scatter(data[:,0],data[:,1])
# plt.show()

def euclDistance(vector1,vector2):
    return np.sqrt(sum((vector2-vector1)**2))

def initCentroids(data,k):                          #初始化质心函数
    numSamples,dim=data.shape
    centroids=np.zeros((k,dim))                     #k个质心，dim个列
    for i in range(k):
        index=random.randint(0,numSamples)
        centroids[i,:]=data[index,:]
    return centroids

def kmeans(data,k):                                 #Kmeans算法
    numSample=data.shape[0]                         #计算样本的个数
    clusterData=np.array(np.zeros((numSample,2)))   #样本的属性，第一列保存样本属于哪个簇，第二列保存该样本跟它所属簇的距离
    clusterChanged=True                             #决定质心是否要改变
    #初始化质心
    centroids=initCentroids(data,k)
    while clusterChanged:
        clusterChanged=False
        for i in range(numSample):                  #计算每个样本到每个质心的距离
            minDistance=100000
            minIndex=0
            for j in range(k):                      #计算每个样本属于的簇和到该簇的最小距离
                distance=euclDistance(centroids[j,:],data[i,:])
                if distance<minDistance:
                    minDistance=distance
                    clusterData[i,1]=minDistance
                    minIndex=j
            if clusterData[i,0]!=minIndex:
                clusterChanged=True
                clusterData[i,0]=minIndex
        for j in range(k):                          #求新的质心
            cluster_index=np.nonzero(clusterData[:,0]==j)               #np.nonezero:返回数组中不为0的元素下标
            pointsInCluster=data[cluster_index]
            centroids[j,:]=np.mean(pointsInCluster,axis=0)
    return centroids,clusterData

def showCluster(data,k,centroids,clusterData):
    numsamples,dim=data.shape
    if dim!=2:
        print('dimension of your data is not 2!')
        return 1

    mark=['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
    if k>len(mark):
        print('k is too large')
        return 1

    for i in range(numsamples):
        markIndex=int(clusterData[i,0])
        plt.plot(data[i,0],data[i,1],mark[markIndex])

    mark=['*r','*b','*g','*k','^b','sb','db','<b','pb']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=20)
    plt.show()


# 单个样本测试
# test=[1,2]
# testData=np.tile(test,(k,1))
# distance=np.sqrt(((testData-centroids)**2).sum(axis=1))
# clusterIndex=np.argmin(distance)
# print(clusterIndex)

def predict(datas):          #测试多组数据算法
    return np.array([np.argmin(((np.tile(data,(k,1))-centroids)**2).sum(axis=1)) for data in datas])

def plot(data):
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
    z=predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    #等高线图
    cs=plt.contourf(xx,yy,z)

def advance(data,k):
    min_centroids=np.array([])      #存放多次训练下来最好的质心
    min_clusterData=np.array([])    #存放多次训练下来最好的样本归类信息
    min_loss=10000
    for i in range(0,50):
        centroids,clusterData = kmeans(data,k)
        check(centroids)
        loss=sum(clusterData[:,1])/data.shape[0]
        if loss<min_loss:
            min_loss=loss
            min_centroids=centroids
            min_clusterData=clusterData
    return min_centroids,min_clusterData

def check(centroids):
    if np.isnan(centroids).any():       #只要质心数组中有任何空值
            print('error')
    else:
        print('cluster complete!')


k=4
centroids,clusterData=advance(data,k)
plot(data)
showCluster(data,k,centroids,clusterData)

