#coding=utf-8
"""
作者：zhaoxingfeng	日期：2016.12.17
功能：Python实现kMeans聚类算法
版本：V1.0
"""
from __future__ import division
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

def loadDataSet(filename):
    dataSet = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataSet.append([float(data) for data in lineArr[::]])
    return np.array(dataSet)

# 中心点在k个样本点中随机选取
def createCent(dataSet, k):
    center = random.sample(dataSet, k)
    return np.array(center)

# 计算聚类中心和数据点之间的距离
def evaDistan(vectA, vectB):
    return np.sqrt(np.sum(np.power((vectA - vectB), 2)))

def kMeans(dataSet, k, evaDistance, createCenter):
    numSamples = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterCenter = createCenter(dataSet, k)
    changeFlag = True
    while changeFlag:
        changeFlag = False
        for i in range(numSamples):
            minDist = float("inf"); minIndex = -1
            for j in range(k):
                distJI = evaDistance(clusterCenter[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                changeFlag = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 重新计算聚类中心
        for cent in range(k):
            sameCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            clusterCenter[cent, :] = np.mean(sameCluster, axis=0)
    return clusterCenter, clusterAssment.A

def show(dataSet, labelSet, k, clusterCenter):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(8, 6))
    mark = ['pr', 'ob', 'og', 'ok', '^r', 'dr', 'sr', 'dr', '<r', 'pg', 'pb', '^g']
    for i in range(np.shape(dataSet)[0]):
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[labelSet[i]], markersize=6)
    for lei in range(k):
        plt.plot(clusterCenter[lei][0], clusterCenter[lei][1], mark[lei], markersize=12)
    plt.xlabel(u'X', fontsize=18)
    plt.ylabel(u'Y', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(u"聚类结果", fontsize=18)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filename = 'test.txt'
    dataSet = loadDataSet(filename)
    k = 4
    clusterCenter, clusterAssment = kMeans(dataSet, k, evaDistance=evaDistan, createCenter=createCent)
    labelSet = [int(x[0]) for x in clusterAssment]
    show(dataSet, labelSet, k, clusterCenter)
