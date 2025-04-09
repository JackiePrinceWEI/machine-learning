import numpy as np
import csv

np.random.seed(0)

# 把csv文件转换成numpy的数组
with open('X_train.csv') as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open('Y_train.csv') as f:
    next(f)
    rows = np.array(list(csv.reader(f))[1:], dtype=float)
    Y_train = rows
with open('X_test.csv') as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


# 数据标准化
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)  # 1e-8防止除零
    return X, X_mean, X_std


# 固定分割测试集
def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# 把数据分成训练集和验证集
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)
train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]


# 生成行的编号，对行的编号进行打乱之后、即可获得一一对应的打乱后的两个数组
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# σ(z)函数
def _sigmoid(z):
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


# fw,b(x)=σ(w⋅x+b)，w为权重、b为偏差
def _f(x, weight, bias):
    return _sigmoid(np.dot(x, weight) + bias)


# 利用round函数实现预测值大于0.5时输出1，否则输出0
def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)


# 计算预测准确率
def _accuracy(f_w_b, y):
    return 1 - np.mean(np.abs(f_w_b - y))


# 计算cross entropy的值，第一个形参为fw,b(x^n)，第二个形参为y^n
def _cross_entropy_loss(f_w_b, y):
    return -(np.dot(y.T, np.log(f_w_b)) + np.dot((1 - y).T, np.log(1 - f_w_b)))


# 梯度,更新参数w、b
def _gradient(X, Y, w, b):
    w1 = -X.T.dot(Y - _f(X, w, b).reshape(-1, 1))
    b1 = -np.sum(Y - _f(X, w, b).reshape(-1, 1))
    return w1, b1


# 开始训练数据
# 初始化权重w和b，令它们都为0
w = np.zeros((data_dim, 1))  # [0,0,0,...,0]
b = np.zeros((1,))  # [0]

# 训练时的超参数
max_iter = 272
batch_size = 128
learning_rate = 0.1

# 保存每个iteration的loss和accuracy，以便后续画图
train_loss, dev_loss, train_acc, dev_acc = [], [], [], []

# adagrad所需的累加和
adagrad_w, adagrad_b = 0, 0
# 防止adagrad除零
eps = 1e-8

step = 1

# 迭代训练
for epoch in range(max_iter):
    # 在每个epoch开始时，随机打散训练数据
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch训练
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)
        adagrad_w += w_grad ** 2
        adagrad_b += b_grad ** 2

        # 梯度下降法adagrad更新w和b
        w = w - learning_rate / (adagrad_w + eps) ** 0.5 * w_grad
        b = b - learning_rate / (adagrad_b + eps) ** 0.5 * b_grad

    # 计算训练集和验证集的loss和accuracy
    y_train_pred = _f(X_train, w, b).reshape(-1, 1)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    a = (_cross_entropy_loss(y_train_pred, Y_train) / train_size)[0, 0]
    train_loss.append(a)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    a = (_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)[0, 0]
    dev_loss.append(a)

# print('Training loss: {}'.format(train_loss[-1]))
# print('Development loss: {}'.format(dev_loss[-1]))
# print('Training accuracy: {}'.format(train_acc[-1]))
# print('Development accuracy: {}'.format(dev_acc[-1]))

# 绘制曲线
import matplotlib.pyplot as plt

# Loss曲线
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy曲线
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# 预测测试集标签
predictions = _predict(X_test, w, b)
# 保存到output_logistic.csv
with open('./output_{}.csv'.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# 输出最重要的特征和权重
# 对w的绝对值从大到小排序，输出对应的ID
ind = np.argsort(np.abs(w))[::-1]
with open('X_test.csv') as f:
    # 读入表头（特征名）
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
