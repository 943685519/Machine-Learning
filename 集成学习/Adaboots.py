import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier

#生成2维正态分布，生成的数据按分位数分为两类，500个样本，2个样本特征
x1,y1=make_gaussian_quantiles(n_samples=500,n_features=2,n_classes=2)
#生成2维正态分布，生成的数据按分位数分为两类，500个样本，2个样本特征，中心点为（3，3）
x2,y2=make_gaussian_quantiles(n_samples=500,n_features=2,n_classes=2,mean=(3,3))
x_data=np.concatenate((x1,x2))
y_data=np.concatenate((y1,-y2+1))

plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

def plot(modle):
    x_min,x_max=x_data[:,0].min()-1,x_data[:,0].max()+1
    y_min,y_max=x_data[:,1].min()-1,x_data[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
    z=modle.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    cs=plt.contourf(xx,yy,z)
    plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
    plt.show()

dtree=tree.DecisionTreeClassifier(max_depth=3)
dtree.fit(x_data,y_data)
plot(dtree)
print(dtree.score(x_data,y_data))

Adaboots=AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3),n_estimators=10)
Adaboots.fit(x_data,y_data)
plot(Adaboots)
print(Adaboots.score(x_data,y_data))