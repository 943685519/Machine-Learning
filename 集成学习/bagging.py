from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris=datasets.load_iris()
x_data=iris.data[:,:2]
y_data=iris.target
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)

def plot(model):
    #两个特征的数据构成的XOY坐标系（取值范围）
    x_min,x_max=x_data[:,0].min()-1,x_data[:,0].max()+1
    y_min,y_max=x_data[:,1].min()-1,x_data[:,1].max()+1

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
    z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    #等高线图
    cs=plt.contourf(xx,yy,z)

#knn预测
knn=neighbors.KNeighborsClassifier()
knn.fit(x_train,y_train)
plot(knn)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()
print('knn-score:',knn.score(x_test,y_test))             #准确率

#dtree预测
dtree=tree.DecisionTreeClassifier()
dtree.fit(x_train,y_train)
plot(dtree)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()
print('dtree-score:',dtree.score(x_test,y_test))           #准确率

#bagging_knn预测
bagging_knn=BaggingClassifier(knn,n_estimators=100)
bagging_knn.fit(x_train,y_train)
plot(bagging_knn)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()
print('bagging_knn-score:',bagging_knn.score(x_test,y_test))

#bagging_dtree预测
bagging_dtree=BaggingClassifier(dtree,n_estimators=100)
bagging_dtree.fit(x_train,y_train)
plot(bagging_dtree)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()
print('bagging_dtree-score:',bagging_dtree.score(x_test,y_test))













