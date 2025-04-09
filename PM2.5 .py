import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_pre_processing():
    data = pd.read_csv('train.csv', usecols=np.arange(3, 27))  # 利用pandas库读取数据.这里使用usecols参数来读取第3-27列
    data = data.replace(['NR'], [0])  # 把值为NR的（大多集中在RAINFALL）替换为0.0
    data = data.to_numpy(dtype='float')  # 把所有的数据都转换为float的类型
    train = {}  # 构建训练集

    # 每个月前20天的数据是连续24小时进行的，为了得到多笔数据，将每个月20天数据连起来:
    for i in range(12):
        # 每个月的数据是一个18 * (20 * 24)的数据块
        temp = np.empty([18, 20 * 24])
        for j in range(20):
            temp[:, j * 24:(j + 1) * 24] = data[18 * (20 * i + j): 18 * (20 * i + j + 1), :]
        # 每隔18取一次PM2.5，分成12月（组），每组480小时
        train[i] = temp
    # 我们再把每个月中的天淡化，共480个小时，可以出471个训练数据，每个训练数据18 * 9个feature，1个label。
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    i, j = 0, 0
    k = 20 * 24 - 9
    while i < 12:
        while j < k:
            # 生成相邻九小时的数据
            x[k * i + j, :] = train[i][:, j:j + 9].reshape(1, -1)
            # 生成一个第十小时的label
            y[k * i + j, :] = train[i][9, j + 9]
            j += 1
        i += 1
    return x, y


# 数据标准化,模型更容易收敛、且收敛更快
def static(x):
    # axis=0表明计算的是列均值
    mean = np.mean(x, axis=0)  # 均值
    std = np.std(x, axis=0)  # 标准差
    # 让数据减去均值再除以方差
    i, j = 0, 0
    while (i < len(x)):
        while (j < len(x[0])):
            if std[j] != 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]
            j += 1
        i += 1


# 线性回归
def BGD(x):
    w = np.zeros([18 * 9 + 1, 1])  # 初始化数组
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # 为1那一列是bias前的系数
    learning_rate = 0.03  # 学习率
    number = 1000  # 设置迭代次数
    loss = np.zeros([number, 1])
    for i in range(number):  # 每次迭代都调用梯度下降算法，计算此时的梯度w和b
        y1 = np.dot(x, w)  # 需要用到求向量积的方法
        loss[i] = np.sqrt(np.sum(np.power(y1 - y, 2)) / 12 * 471)
        gradient = 2 * np.dot(x.transpose(), y1 - y) / 12 * 471  # 求偏导，多除一个N可以利用更大的学习率
        w = w - learning_rate * gradient
    plt.plot(loss)
    plt.show()


# Adagrad梯度下降优化
def Adagrad(x):
    w = np.zeros([18 * 9 + 1, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # 为1那一列是bias前的系数
    learning_rate = 0.03  # 学习率
    number = 1000  # 设置迭代次数
    loss = np.zeros([number, 1])
    adagrad = np.zeros([18 * 9 + 1, 1])
    eps = 0.000000001
    for i in range(number):  # 梯度更新，求解最佳梯度
        y1 = np.dot(x, w)
        loss[i] = np.sqrt(np.sum(np.power(y1 - y, 2)) / 12 * 471)
        gradient = 2 * np.dot(x.transpose(), y1 - y) / 12 * 471  # 多除一个N可以利用更大的学习率
        adagrad += pow(gradient, 2)
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # eps为了防止分母为零的一个很小的数
    plt.plot(loss)
    plt.show()
    return w


# 矩阵加速
def speed(xl, yl, b0, w0):
    # 对18个特征的每九个参数进行计算偏导数
    b2 = np.zeros(1)
    w2 = np.zeros((18, 9))
    z = yl - (xl * w0).sum(axis=1).sum(axis=1)
    b2 = (-2.0) * (z.sum() / 5652 - b0)
    w2 = (-2.0) * xl.T.dot(z - b0).T / 5652
    return b2, w2


x, y = data_pre_processing()
static(x)
# w = BGD(x)
w = Adagrad(x)
np.save('weight.npy', w)  # save方法需指定文件保存路径，如果未设置，保存到默认路径。其文件拓展名为.npy
