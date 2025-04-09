import numpy as np
import pandas
import time


def readcsv1(name):
    # 调用pandas库对文件进行预处理
    data = pandas.read_csv(name)
    data = data.replace(['NR'], [0.0])
    data = np.array(data)
    data = data[:, 3:].astype(float)
    # 取出数值部分，并对NR处进行处理
    xlist = []
    ylist = []
    for i in range(0, 4320, 18):
        for j in range(15):
            mat = np.array(data[i:i + 18, j:j + 9])
            label = np.array(data[i + 9, j + 9])
            xlist.append(mat)
            ylist.append(label)
    # 选取九个时候的数据同第十个时候的PM2.5值作为特征和标签
    x = np.array(xlist)
    y = np.array(ylist)
    return x, y


def readcsv(name):
    # 5652个参数
    data = pandas.read_csv(name)
    data = data.replace(['NR'], [0.0])
    data = np.array(data)
    data = data[:, 3:].astype(float)
    month_data = []
    for i in range(12):  # 每个月的数据
        sub_data = np.empty([18, 20 * 24])
        # 建立空矩阵，一个个填进去
        for j in range(20):  # 每日的数据
            sub_data[:, j * 24:(j + 1) * 24] = data[(i * 18 * 20 + j * 18):(i * 18 * 20 + (j + 1) * 18), :]
        month_data.append(sub_data)
    # 将12个月中的数据连成一长串，产生12组数据集
    xlist = []
    ylist = []
    for i in range(12):
        for j in range(471):
            xlist.append(month_data[i][:, j:j + 9])
            ylist.append(month_data[i][9, j + 9])
    x = np.array(xlist)
    y = np.array(ylist)
    return x, y


def getGrad(xl, yl, b0, w0):
    # 计算参数的偏导数
    # b2=np.zeros(1)
    # w2=np.zeros(9)
    # b2=(-2.0)*((np.subtract(yl,np.dot(xl,w0)).sum())/3600-b0)
    # w2=(-2.0)*np.sum(xl.T*np.subtract(yl,b0+np.dot(xl,w0)).T,axis=1)/3600
    # return b2,w2
    # 上面为计算一个特征九个参数时候的偏导数，下面将对18个特征的每九个参数进行计算偏导数
    b2 = np.zeros(1)
    w2 = np.zeros((18, 9))
    z = yl - (xl * w0).sum(axis=1).sum(axis=1)
    b2 = (-2.0) * (z.sum() / 5652 - b0)
    w2 = (-2.0) * xl.T.dot(z - b0).T / 5652
    return b2, w2


def Feature(narray):
    # nmean=narray.mean(axis = 0)
    # nm=nmean.mean(axis = 1)
    # nt=[]
    # for i in range(18):
    #     nt.append(narray[:,i,:].mean())
    # #求取了3600个数据的18个特征参数的平均值
    # ns=[]
    # for i in range(18):
    #     ns.append(narray[:,i,:].std())
    # 求取了3600个数据的18个特征参数的标准差
    # 上面是一开始分步操作时候的方法，下面后来想到的一步到位的操作方法
    for i in range(18):
        narray[:, i, :] = (narray[:, i, :] - narray[:, i, :].mean()) / narray[:, i, :].std()
    return narray


def test(name, w0, b0):
    # 对test文件进行预测
    data = pandas.read_csv(name, header=None)
    data = data.replace(['NR'], [0.0])
    data = np.array(data)
    idlist = []
    for i in range(0, 4320, 18):
        idlist.append(data[i, 0])
    idlist = np.array(idlist)
    data = data[:, 2:].astype(float)
    xlist = []
    for i in range(0, 4320, 18):
        mat = np.array(data[i:i + 18, :])
        xlist.append(mat)
    x = np.array(xlist)
    x = Feature(x)
    # 前面为对文件进行处理，同一开始相同
    z = (x * w0)
    y = z.sum(axis=1).sum(axis=1) + b0
    dataframe = pandas.DataFrame({'id_n': idlist, 'PM2.5': y})
    dataframe.to_csv("result.csv", index=False, sep=',')


st = time.time()
xarray, yarray = readcsv('train.csv')
xarray = Feature(xarray)  # 对数据进行标准化
# xarray1=xarray[0:3200,:,:]
# yarray1=yarray[0:3200]
b = np.zeros(1)
w = np.zeros((18, 9))
lr = 1
bg_sum = np.zeros(1)
wg_sum = np.zeros((18, 9))
# 参数和参数平方和的初始化
for i in range(15000):
    # 迭代，并利用adagrad算法来更新学习率
    b1, w1 = getGrad(xarray, yarray, b, w)

    bg_sum += b1 ** 2
    wg_sum += w1 ** 2
    # 更新权重和偏置
    b -= lr / bg_sum ** 0.5 * b1
    w -= lr / wg_sum ** 0.5 * w1
loss = (np.power(yarray - (xarray * w).sum(axis=1).sum(axis=1) - b, 2) / 5652).sum()
print("loss:", loss)
print("use time:", time.time() - st, "s")

test('test.csv', w, b)
print("use time:", time.time() - st, "s")
