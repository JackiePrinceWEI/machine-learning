import numpy as np
import sys
import csv
from numpy.linalg import inv


class data_manager():
    def __init__(self):
        self.data = {}

    def read(self, name, path):
        with open(path, newline='') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:], dtype=float)
            if name == 'X_train':
                # 使用正态分布的标准化处理方式，mean为平均值，std为方差
                self.mean = np.mean(rows, axis=0).reshape(1, -1)
                self.std = np.std(rows, axis=0).reshape(1, -1)
                self.theta = np.ones((rows.shape[1] + 1, 1), dtype=float)
                # 依次对每一行进行标准化：减去平均值后除以方差
                for i in range(rows.shape[0]):
                    # rows[i, :] =  ####please add code here
                    rows[i, :] = (rows[i, :] - self.mean) / self.std
            elif name == 'X_test':
                # 使用train里的均值和方差对test数据进行标准化
                for i in range(rows.shape[0]):
                    # rows[i, :] =  ####please add code here
                    rows[i, :] = (rows[i, :] - self.mean) / self.std
            # 将标准化后的数据存入data[name]中
            self.data[name] = rows

    def find_theta(self):
        # 将训练数据分为class_0_id、class_1_id两组
        class_0_id = []
        class_1_id = []
        # 循环遍历，训练集中小于5万的数据划入class_0_id类中，其他的划入class_1_id中
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)

        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id]
        # 分别计算均值mean_0和mean_1
        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)
        # 初始化协方差矩阵
        n = class_0.shape[1]
        cov_0 = np.zeros((n, n))
        cov_1 = np.zeros((n, n))
        N0, N1 = 0, 0
        # 对class_0_id和class_1_id分别计算协方差矩阵
        for i in range(class_0.shape[0]):
            # cov_0 +=  ####please add code here
            cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [class_0[i] - mean_0])
            N0 += 1
        for i in range(class_1.shape[0]):
            # cov_1 +=  ####please add code here
            cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [class_1[i] - mean_1])
            N1 += 1
        # cov =  ####please add code here
        cov_0 /= N0
        cov_1 /= N1
        # 求两个协方差的加权平均值，并将其作为共用的协方差值
        cov = ((cov_0 * N0) + (cov_1 * N1)) / (N0 + N1)
        # 求解w和b
        # self.w =  ####please add code here
        # self.b =  ####please add code here
        self.w = np.dot(inv(cov), (mean_0 - mean_1).T)
        self.b = (-1 / 2) * np.dot(np.dot(mean_0.T, inv(cov)), mean_0) + 0.5 * np.dot(np.dot(mean_1.T, inv(cov)),
                                                                                      mean_1) + np.log(float(N0) / N1)
        result = self.func(self.data['X_train'])  # func函数定义下面已经给出，就是sigmoid函数
        # 计算对训练集中数据的预测值
        answer = self.predict(result)
        # 计算模型预测的准确率：1-错误率（预测值与实际值绝对差的平均值）
        accuracy = 1 - np.mean(np.abs(self.data['Y_train'] - answer))
        print("Accuracy:", accuracy)

    # sigmoid函数
    def func(self, x):
        arr = np.empty([x.shape[0], 1], dtype=float)
        for i in range(x.shape[0]):
            z = x[i, :].dot(self.w) + self.b
            z *= (-1)
            arr[i][0] = 1 / (1 + np.exp(z))
        return np.clip(arr, 1e-8, 1 - (1e-8))

    # 经过func后，若输出结果大于0.5（年收入大于5万）则赋值为1，否则赋值为0
    def predict(self, x):
        ans = np.ones([x.shape[0], 1], dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                ans[i] = 0
        return ans

    # 将预测值写入output.csv文件
    def write_file(self, path):
        result = self.func(self.data['X_test'])
        answer = self.predict(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            for i in range(answer.shape[0]):
                writer.writerow([i + 1, answer[i][0]])


# 创建data_manager类
dm = data_manager()
# 读取文件并进行预处理
dm.read('X_train', 'X_train.csv')
dm.read('Y_train', 'Y_train.csv')
dm.read('X_test', 'X_test.csv')
# 参数计算
dm.find_theta()
# 将预测数据写入output.csv文件
dm.write_file('output.csv')
