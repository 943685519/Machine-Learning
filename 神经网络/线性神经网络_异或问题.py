# x0w0 +x1w1 + x2w2 + x1^2*w3 + x2^*w5 +x1x2w4 = 0
#线性神经网络做异或问题时需要加入非线性输入，使模型的维度变大
import numpy as np
import matplotlib.pyplot as plt

X_data = np.array([[1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 1, 0, 0],
                   [1, 0, 1, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1]])
Y_data = np.array([[-1],
                   [1],
                   [1],
                   [-1]])
W = (np.random.random([6, 1]) - 0.5) * 2  #random自动生成一个6行一列的，取值在0~1之间的矩阵
print(W)

lr = 0.11
# 神经网络的输出
O = 0


def updata():
    global X_data, Y_data, W, lr, O
    O = np.dot(X_data, W)
    W_C = lr * (X_data.T.dot(Y_data - O)) / int(X_data.shape[0])
    W = W + W_C


for i in range(1000):
    updata()

x1 = [0, 1]
y1 = [1, 0]

x2 = [1, 0]
y2 = [1, 0]

#计算y的值
def caculate(x, root):
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


xdata = np.linspace(-1, 2)

plt.figure()
plt.plot(xdata, caculate(xdata, 1), 'r')
plt.plot(xdata, caculate(xdata, 2), 'r')
plt.scatter(x1, y1, c='b', )
plt.scatter(x2, y2, c='y')
plt.show()
print(np.dot(X_data, W))
