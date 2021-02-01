import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('data.csv',delimiter=',')
x_data=data[:,0]
y_data=data[:,1]
plt.scatter(x_data,y_data)
# plt.show()
print(x_data.shape)

#去中心化
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData,meanVal

#求协方差矩阵
def covMat(newData):
    covmat=np.cov(newData,rowvar=0)
    return covmat

#特征值，特征向量
def eig(covmav):
    eigVals,eigVects=np.linalg.eig(np.mat(covmav))
    return eigVals,eigVects

newData,meanVal=zeroMean(data)              #得到中心化后的矩阵
covmat=covMat(newData)                      #得到协方差矩阵
eigVals,eigVects=eig(covmat)
#得到特征值，特征向量
print(covmat)
print(eigVals)
print(eigVects)

eigValIndice=np.argsort(eigVals)            #排序特征值，从小到大
print(eigValIndice)

top=1
n_eigValIndice=eigValIndice[-1:-(top+1):-1] #得到最大的top个特征值下标
n_eigVect=eigVects[:,n_eigValIndice]        #最大的top个特征值对应的特征向量矩阵

lowDataMat=newData*n_eigVect                #降维后的数据，就是原数据在最大特征向量矩阵上的投影
print('数据的维度',newData.shape)
print('特征向量矩阵的维度',n_eigVect.shape)
print('协方差矩阵维度',covmat.shape)
print('低维数据维度',lowDataMat.shape)
print('前x个特征向量组成的矩阵维度',n_eigVect.shape)

#一维数据在特征向量上的投影
reconData=lowDataMat*n_eigVect.T+meanVal
x_data=np.array(reconData)[:,0]
y_data=np.array(reconData)[:,1]
plt.scatter(x_data,y_data,c='r')
plt.show()