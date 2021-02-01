from sklearn import svm

x=[[3,3],[4,3],[1,1]]
y=[1,1,-1]

model=svm.SVC(kernel='linear')
model.fit(x,y)

print(model.support_vectors_)           #打印支持向量

print(model.support_)                   #属于支持向量的样本的下标

print(model.n_support_)                 #支持向量的个数

print(model.predict([[4,3]]))

print(model.coef_)                      #超平面的方向
print(model.intercept_)                 #超平面的截距