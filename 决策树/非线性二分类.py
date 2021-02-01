import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split

data=np.genfromtxt('E:\\机器学习\决策树\LR-testSet2.txt',delimiter=',')
x_data=data[:,:-1]
y_data=data[:,-1]
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)
model=tree.DecisionTreeClassifier(max_depth=6,min_samples_split=4)      #树的深度,内部节点再划分所需最小样本数，就是该点共3个样本，即使可以再分支，也不分了
model.fit(x_train,y_train)

import graphviz
dot_data=tree.export_graphviz(model,
                              out_file=None,
                              feature_names=['x','y'],
                              class_names=['label0','label1'],
                              filled=True,
                              rounded=True,
                              special_characters=True)
graph=graphviz.Source(dot_data)
graph.view()

x_min,x_max=x_data[:,0].min()-1,x_data[:,0].max()+1
y_min,y_max=x_data[:,1].min()-1,x_data[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                  np.arange(y_min,y_max,0.02))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
cs=plt.contourf(xx,yy,z)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

predictions=model.predict(x_train)
print(classification_report(predictions,y_train))

predictions=model.predict(x_test)
print(classification_report(predictions,y_test))
