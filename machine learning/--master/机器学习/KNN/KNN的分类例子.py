import matplotlib.pyplot as plt
import numpy as np
import operator

#1类型的数据，3个样本
x1=np.array([3,2,1])
y1=np.array([104,100,81])
#2类型的数据，3个样本
x2=np.array([101,99,98])
y2=np.array([10,5,2])

scatter1=plt.scatter(x1,y1,c='r')
scatter2=plt.scatter(x2,y2,c='b')

#未知数据
x=np.array([18])
y=np.array([90])
scatter3=plt.scatter(x,y,c='k')

#画图例
plt.legend(handles=[scatter1,scatter2,scatter3],labels=['lableA','lableB','X'],loc='best')
plt.show()

x_data=np.array([[3,104],
                 [2,100],
                 [1,81],
                 [101,10],
                 [99,5],
                 [81,2]])
y_data=np.array(['A','A','A','B','B','B'])
x_test=np.array([18,90])

x_test=np.tile(x_test,(x_data.shape[0],1))      #测试数据矩阵
diffMat=x_test-x_data                           #测试数据矩阵与样本矩阵的差
sqDiffMat=diffMat**2
sqDistances=sqDiffMat.sum(axis=1)
distance=sqDistances**0.5
print(distance)

list=distance.argsort()
print(list)

classCount={}
for i in range(5):
    sign=y_data[list[i]]
    classCount[sign]=classCount.get(sign,0) + 1     #字典中的sign对应的值等于得到这个sign的值，没有就赋值0，得到了就让sign的值加一
print(classCount)

max=0
maxname='C'
for i in classCount:
    if int(classCount[i])>max:
        max=int(classCount[i])
        maxname=i
print(maxname)