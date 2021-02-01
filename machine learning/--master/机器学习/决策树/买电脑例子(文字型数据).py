from sklearn.feature_extraction import DictVectorizer   #特征提取
from sklearn import tree                            #决策树库
from sklearn import preprocessing                   #数据预处理库
import csv      #用于读取CSV文件的库

#读取数据
Dtree=open(r'AllElectronics.csv')       #以读的方式打开文件
reader=csv.reader(Dtree)                #用csv读取文件，reader对象存放所有数据

#获取reader的第一行数据,用作标签
headers=reader.__next__()
print(headers)

#定义两个列表
featureList=[]          #特征值列表
labelList=[]            #标签列表

#给列表赋值
for row in reader:
    labelList.append(row[-1])           #将每个的类别放在labellist中
    rowDict={}                          #存放每个个体的信息字典
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]      #将属性和对应的值放在字典中
    featureList.append(rowDict)         #将每条数据对应的字典加入列表
print(featureList)
print(labelList)

#将文字数据表示为0和1（字典转数组）
vec=DictVectorizer()                               #将特征与值的映射字典组成的列表转换成向量的对象。
x_data=vec.fit_transform(featureList).toarray()    #将列表中的字典转化为0和1的形式
print("x_data:\n",(x_data))
#打印特征名称
print(vec.get_feature_names())
#打印标签
print('labelList:',(labelList))

#把标签列表转化为0和1表示（列表中字符转0，1）
lb=preprocessing.LabelBinarizer()
y_data=lb.fit_transform(labelList)
print('y_data',y_data)

#创建决策树模型，使用信息熵来构建树
model=tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_data,y_data)

#做测试
x_test=x_data[0]
print('x_test:',x_test)
predict=model.predict(x_test.reshape(1,-1))     #在列表两边加括号变为二维的数据
print('predict:',predict)

#导出决策树
import graphviz
dot_data=tree.export_graphviz(model,
                              out_file=None,
                              feature_names=vec.get_feature_names(),
                              class_names=lb.classes_,
                              filled=True,
                              rounded=True,
                              special_characters=True)
graph=graphviz.Source(dot_data)
graph.render('computer')
graph.view()
# print(dir(graph))
print(vec.get_feature_names())
