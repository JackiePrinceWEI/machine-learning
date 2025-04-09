import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# 用于测试集的预测输出
shuchu_path = 'output_{}.csv'

# 把csv文件转换成numpy的数组
with open('X_train.csv') as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[0:] for line in f], dtype=float)
with open('Y_train.csv') as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[0] for line in f], dtype=float)
with open('X_test.csv') as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[0:] for line in f], dtype=float)


# 数据标准化
def standard_(X, train=True, column_specified=None, mean_of_X=None, X_std=None):
    if column_specified == None:
        column_specified = np.arange(X.shape[1])
    if train:
        # 计算每个数据的平均值和标准差
        mean_of_X = np.mean(X[:, column_specified], 0).reshape(1, -1)
        X_std = np.std(X[:, column_specified], 0).reshape(1, -1)
    # 归一化数据
    X[:, column_specified] = (X[:, column_specified] - mean_of_X) / (X_std + 1e-8)
    # 返回归一化后的数据，均值，标准差
    return X, mean_of_X, X_std


# 归一化数据
X_train, mean_of_X, X_std = standard_(X_train, train=True)
X_test, _, _ = standard_(X_test, train=False, column_specified=None, mean_of_X=mean_of_X, X_std=X_std)


# 固定分割测试集
def data_train_split(X, Y, ratio=0.25):
    train_size = int(len(X) * (1 - ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# 把数据分成训练集和验证集
ratio = 0.1
X_train, Y_train, X_dev, Y_dev = data_train_split(X_train, Y_train, ratio=ratio)

print(X_train.shape)
train_size, dev_size, test_size, data_dim = X_train.shape[0], X_dev.shape[0], X_test.shape[0], X_train.shape[1]


# print('Size of training set: {}'.format(train_size))
# print('Size of development set: {}'.format(dev_size))
# print('Size of testing set: {}'.format(test_size))
# print('Dimension of data: {}'.format(data_dim))


# 生成行的编号，对行的编号进行打乱之后、即可获得一一对应的打乱后的两个数组
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# σ(z)函数
def Sig_moid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


# fw,b(x)=σ(w⋅x+b)，w为权重、b为偏差
def _f(X, w, b):
    return Sig_moid(np.matmul(X, w) + b)


# 预测值大于0.5时输出1，否则输出0
def yuce_(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)


# 计算预测准确率
def _accuracy(a, b):
    return 1 - np.mean(np.abs(a - b))


# 交叉熵损失函数计算cross entropy的值，第一个形参为fw,b(x^n)，第二个形参为y^n
def Jiaocha_Loss(f_w_b, y_n):
    return -np.dot(y_n, np.log(f_w_b)) - np.dot((1 - y_n), np.log(1 - f_w_b))


# 梯度,更新参数w、b
def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# 开始训练数据
# 初始化权重w和b，令它们都为0
w = np.zeros((data_dim,))
b = np.zeros((1,))

# 训练时的超参数（迭代次数，分批次大小，学习率）
max_iter, batch_size, learning_rate = 10, 8, 0.2

# 保存每个iteration的loss和accuracy，以便后续画图
train_loss, dev_loss, train_acc, dev_acc = [], [], [], []

# 用来更新学习率
step = 1

# 训练
for epoch in range(max_iter):
    # 每个epoch都会重新洗牌
    # X_train, Y_train = _shuffle(X_train, Y_train)

    # 分批次训练
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # 计算梯度值
        w_grad, b_grad = _gradient(X, Y, w, b)

        # 更新参数w、b
        w -= learning_rate / np.sqrt(step) * w_grad
        b -= learning_rate / np.sqrt(step) * b_grad
        step += 1
    # 参数总共更新了max_iter × （train_size/batch_size）次
    # 计算训练集的损失值和准确度

    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(Jiaocha_Loss(y_train_pred, Y_train) / train_size)

    # 计算验证集的损失值和准确度
    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(Jiaocha_Loss(y_dev_pred, Y_dev) / dev_size)

print(w, b)
print(Y_train_pred.shape[0])
print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))


def draw(a, b, c, d):
    plt.plot(a)
    plt.plot(b)
    plt.title(c)
    plt.legend(['train', 'dev'])
    plt.savefig(d)
    plt.show()


# 绘制Loss的曲线
draw(train_loss, dev_loss, 'Loss', 'loss.png')
# 绘制Accuracy的曲线
draw(train_acc, dev_acc, 'Accuracy', 'acc.png')
