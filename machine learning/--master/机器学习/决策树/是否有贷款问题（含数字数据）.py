from sklearn import tree
import numpy as np

data=np.genfromtxt('E:\\机器学习\决策树\cart.csv',delimiter=',')
x_data=data[1:,1:-1]
y_data=data[1:,-1]

model=tree.DecisionTreeClassifier()     #criterion不赋值就默认是Gini系数
model.fit(x_data,y_data)

import graphviz
dot_data=tree.export_graphviz(model,
                              out_file=None,
                              feature_names=['house_yes','house_no','single','married','divorced','income'],
                              class_names=['no','yes'],
                              filled=True,
                              rounded=True,
                              special_characters=True)
graph=graphviz.Source(dot_data)
graph.render('cart')
graph.view()