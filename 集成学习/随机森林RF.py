from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=np.genfromtxt('E:\\机器学习\决策树\LR-testSet2.txt',delimiter=',')
x_data=data[:,:-1]
y_data=data[:,-1]
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.5)

def plot(modle):
    x_min,x_max=x_data[:,0].min()-1,x_data[:,0].max()+1
    y_min,y_max=x_data[:,1].min()-1,x_data[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
    z=modle.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    cs=plt.contourf(xx,yy,z)
    plt.scatter(x_test[:,0],x_test[:,1],c=y_test)
    plt.show()

CART_tree=tree.DecisionTreeClassifier()
CART_tree.fit(x_train,y_train)
plot(CART_tree)
print(CART_tree.score(x_test,y_test))

RF=RandomForestClassifier(n_estimators=50)
RF.fit(x_train,y_train)
plot(RF)
print(RF.score(x_test,y_test))