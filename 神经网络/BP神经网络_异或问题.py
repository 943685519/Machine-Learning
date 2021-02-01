import numpy as np

X=np.array([[1,0,0],
           [1,0,1],
           [1,1,0],
           [1,1,1]])
Y=np.array([[0,1,1,0]])
V=np.random.random((3,4))*2-1   #隐含层的系数矩阵
W=np.random.random((4,1))*2-1   #输出层的系数矩阵
lr=0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def updata():
    global X,Y,W,V,lr

    L1=sigmoid(np.dot(X,V))     #隐含层的输出y，4*4矩阵
    L2=sigmoid(np.dot(L1,W))    #输出层的输出O，4*1矩阵

    L2_delta=(Y.T-L2)*dsigmoid(L2)      # 4*1
    L1_delta=L2_delta*W.T*dsigmoid(L1)  # 4*4

    W_C=lr*L1.T.dot(L2_delta)           # 4*1
    V_C=lr*X.T.dot(L1_delta)            # 3*4

    W = W + W_C
    V = V + V_C

for i in range(20000):
    updata()
    if i%500==0:
        L1 = sigmoid(np.dot(X, V))  # 隐含层的输出y，4*4矩阵
        L2 = sigmoid(np.dot(L1, W))  # 输出层的输出O，4*1矩阵
        print('error:',np.mean(np.abs(Y.T-L2)))

L1 = sigmoid(np.dot(X, V))  # 隐含层的输出y，4*4矩阵
L2 = sigmoid(np.dot(L1, W))  # 输出层的输出O，4*1矩阵

print(L2)