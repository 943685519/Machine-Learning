import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split        #sklearn切分数据集模块
from sklearn.metrics import classification_report,confusion_matrix      #测试结果指标
import operator
import random

def KNN(x_test,x_data,y_data,k):
    x_data_size=x_data.shape[0]
    X_test=np.tile(x_test,(x_data_size,1))
    diffMat=X_test-x_data
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistances=distances.argsort()         #该排序是以原列表的元素下标值来排序
    classCount={}
    for i in range(k):
        label=y_data[sortedDistances[i]]
        classCount[label]=classCount.get(label,0)+1
    #classCount.items()是得到字典的可迭代对象
    #operator.itemgetter是获取对象的第几维的数据，等于1时就获取到了标签对应的值
    #reverse是排序方式，True是降序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

iris=datasets.load_iris()
#切分数据集函数 x_train , x_test , y_train , y_test=train_test_split(iris.data,iris.target,test_size=0.2)
#打乱数据
data_size=iris.data.shape[0]
index=[i for i in range(data_size)]
random.shuffle(index)               #打乱该列表中的值
iris.data=iris.data[index]
iris.target=iris.target[index]
#切分数据
test_size=40
x_train=iris.data[test_size:]
x_test=iris.data[:test_size]
y_train=iris.target[test_size:]
y_test=iris.target[:test_size]
predictions=[]
for i in range(x_test.shape[0]):
    predictions.append(KNN(x_test[i],x_train,y_train,5))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))