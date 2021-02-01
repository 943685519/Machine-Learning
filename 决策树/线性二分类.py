import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import tree

data=np.genfromtxt('E:\\机器学习\决策树\LR-testSet.csv',delimiter=',')
x_data=data[:,:-1]
y_data=data[:,-1]
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

modle=tree.DecisionTreeClassifier()
modle.fit(x_data,y_data)

import graphviz
dot_data=tree.export_graphviz(modle,
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
z=modle.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
cs=plt.contourf(xx,yy,z)
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

predictions=modle.predict(x_data)
print(classification_report(predictions,y_data))