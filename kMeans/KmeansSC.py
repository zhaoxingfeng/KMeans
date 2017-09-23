# -*- coding: utf-8 -*-
"""
作者：zhaoxingfeng	日期：2017.9.23
功能：Python实现kMeans聚类算法，轮廓系数确定最佳k
版本：V1.0
"""
from __future__ import division
import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# 类中心在最小最大值范围内随机取值
def center_minmax(data, k):
    n, m = np.shape(data)
    cent_list = []
    for i in range(k):
        cent = []
        for j in range(m):
            mi, mx = min(data[:, j]), max(data[:, j])
            cent.append(mi + random.random() * (mx - mi))
        cent_list.append(cent)
    return cent_list

# 类中心在k个样本点中随机选取
def center_rand(data, k):
    return random.sample(data, k)

# 距离度量函数
def evaDist(arrA, arrB):
    return np.sqrt(np.sum(np.power((arrA - arrB), 2)))

def kmeans(data, k, max_iter):
    n, m = np.shape(data)
    center = center_minmax(data, k)
    # 用字典存储每个样本[所属类别，类中心，样本，距离]
    data_dict = {}
    # 用字典存储每一类所包含的样本
    class_dict = {}
    for i in range(n):
        data_dict[i] = [None, None, data[i], None]
    flag = True
    itr = 0
    while flag:
        print "iter", k, itr
        itr += 1
        flag = False
        data_dict_old = copy.deepcopy(data_dict)
        for i in range(k):
            class_dict[i] = []
        for i in range(n):
            dist_min = 'inf'
            for j in range(k):
                dist = np.sum(np.power(data[i] - center[j], 2)) ** 0.5
                if dist < dist_min:
                    dist_min = dist
                    data_dict[i][0] = j
                    data_dict[i][1] = center[j]
                    data_dict[i][-1] = dist
            class_dict[data_dict[i][0]].append(data[i])
            # 如果和上一次相比，有样本点距离类簇中心的距离发生变化，则继续迭代
            if data_dict[i][-1] != data_dict_old[i][-1]:
                flag = True
        for key, value in class_dict.iteritems():
            if value:
                data_class = np.vstack(value)
                center[key] = np.mean(data_class, axis=0)
        if itr > max_iter:
            break
    # 轮廓系数 s[i] = (b[i] - a[i]) / max(a[i], b[i])，对所有样本求轮廓系数均值，越接近1聚类效果越好
    s = 0
    for key, value in class_dict.iteritems():
        for val in value:
            a = 0
            for vl in value:
                a += evaDist(val, vl)
            a /= len(value)
            blst = []
            for key1, value1 in class_dict.iteritems():
                if key1 == key:
                    continue
                else:
                    b = 0
                    for val1 in value1:
                        b += evaDist(val, val1)
                    blst.append(b / len(value1))
            s += (min(blst) - a) / max(min(blst), a)
    return s / n, data_dict, center

def show(data, k, everyDict, center):
    mark = ['pr', 'ob', 'og', 'ok', '^r']
    for i in range(np.shape(data)[0]):
        plt.plot(data[i][0], data[i][1], mark[everyDict[i][0]], markersize=6)
    for lei in range(k):
        plt.plot(center[lei][0], center[lei][1], mark[lei], markersize=12)
    plt.show()

if __name__ == "__main__":
    # k=4时轮廓系数最大
    kdict = {}
    df = pd.read_csv('test.txt', header=None, sep='\t').values
    for k in range(2, 8):
        sk, data_dict, center = kmeans(df, k, 100)
        kdict[k] = [sk, data_dict, center]
    bestk = sorted(kdict.iteritems(), key=lambda x: x[1][0], reverse=True)[0][0]
    print 'best k and center:', bestk, kdict[bestk][-1]
    plt.plot(kdict.keys(), map(lambda x: x[0], kdict.values()))
    plt.show()
    show(df, bestk, kdict[bestk][1], kdict[bestk][-1])
