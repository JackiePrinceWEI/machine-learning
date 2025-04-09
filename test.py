import csv
import pandas as pd
import numpy as np


def predict():
    data = pd.read_csv('test.csv', header=None)
    data = data.iloc[:, 2:]
    data = data.replace(['NR'], [0])  # 把值为NR的（大多集中在RAINFALL）替换为0.0
    data = data.to_numpy()

    x = np.empty([240, 162], dtype=float)
    i = 0
    while i < 240:
        x[i, :] = data[18 * i: 18 * (i + 1), :].reshape(1, -1)
        i += 1
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    i, j = 0, 0
    while i < len(x):
        while j < len(x[0]):
            if std[j] != 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]
            j += 1
        i += 1
    x = np.concatenate((np.ones([240, 1]), x), axis=1).astype(float)
    w = np.load('weight.npy')
    y = np.dot(x, w)  # 点乘
    return y


test_y = predict()
# 输出结果csv文件
with open('predict.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    k = ['id', 'PM2.5']  # 要求：输出的csv文件第一行必须是 id，value
    csv_writer.writerow(k)
    i = 0
    while i < 240:
        if int(test_y[i][0]) > 0:
            row = ['id_' + str(i), str(int(test_y[i][0]))]
        else:
            row = ['id_' + str(i), '0']  # 倘若是负值则输出0
        csv_writer.writerow(row)
        i += 1
