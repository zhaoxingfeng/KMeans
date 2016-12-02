#coding=utf-8
"""
作者：zhaoxingfeng	日期：2016.11.19
功能：Python实现kMeans聚类算法
版本：V2.0
"""
from __future__ import division
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl


class kMeans(object):
    # 读取数据
    def loadDataSet(self,filename):
        dataSet = []
        with open(filename) as fr:
            for line in fr.readlines():
                lineArr = line.strip().split('\t')
                dataSet.append([float(data) for data in lineArr[::]])
        return dataSet

    # 中心点在k个样本点中随机选取
    def createCenter(self,k,data):
        center1 = random.sample(data, k)
        center = []
        for i in xrange(k):
            center.append(np.array(center1[i]))
        return center

    # 中心点在最小最大值范围内随机取值
    def createCenter1(self,k,data):
        center = []
        for i in xrange(k):
            a = []
            for j in xrange(np.shape(data)[1]):
                minj, maxj = min(data[:,j]), max(data[:,j])
                a.append(minj + np.random.random() * (maxj - minj))
            center.append(np.array(a))
        return center

    # 计算聚类中心和数据点之间的距离
    def evaDist(self,arrA,arrB):
        distance = np.sqrt(np.sum(np.power((arrA - arrB),2)))
        return distance

    # 绘制散点图和聚类中心
    def show(self,data,k,everyDict,center):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        mark = ['pr', 'ob', 'og', 'ok', '^r', 'dr', 'sr', 'dr', '<r']
        for i in range(np.shape(data)[0]):
            plt.plot(data[i][0],data[i][1],mark[everyDict[i][0]], markersize = 6)
        for lei in range(k):
            plt.plot(center[lei][0],center[lei][1], mark[lei], markersize = 12)
        ax.set_xlabel(u'X',fontsize=18)
        ax.set_ylabel(u'Y',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(u"聚类结果",fontsize=18)
        plt.grid(True)
        plt.show()

    # filename为数据集文件地址；k为聚类数目；itermax为最大迭代次数
    def run(self,filename,k,itermax):
        data = np.array(self.loadDataSet(filename))
        num, dim = np.shape(data)[0], np.shape(data)[1]
        center = self.createCenter(k,data)
        # 用字典存储每个数据点所属类别，类中心，数据点，距离
        everyDict = {}      
        for p in xrange(num):
            everyDict[p] = [None,None,data[p],None]
        changeFlag = True
        # 用字典来存储每一类所包含的数据点
        classDict = {}  
        iter = 0
        while changeFlag:
            changeFlag = False
            # 存储上一次迭代的结果，和这一次迭代的结果进行比较，判断距离是否有变化
            everyDictago = copy.deepcopy(everyDict) 
            for kk in xrange(k):
                classDict[kk] = []
            for i in xrange(num):
                index, distmin = 0, float('inf')
                for j in xrange(k):
                    distIJ = self.evaDist(data[i],center[j])
                    if distIJ < distmin:
                        index = j; distmin = distIJ
                everyDict[i][0], everyDict[i][1], everyDict[i][-1] = index, center[index], distmin
                classDict[everyDict[i][0]].append(everyDict[i][2])
                # 如果和上一次相比距离有变化，则继续迭代
                if everyDict[i][-1] != everyDictago[i][-1]: 
                    changeFlag = True
            for item, value in classDict.iteritems():
                 # 将每一类包含的的数据点整合为一个数组，便于求平均值
                arrindex = np.vstack((classDict[item]))  
                center[item] = np.mean(arrindex, axis=0)
            iter += 1
            if iter > itermax:
                break
            print("iter " + str(iter) + ":  " + str(center))
        print("The final cluster center: " + str(center))
        self.show(data,k,everyDict,center)

if __name__ == "__main__":
    filename = 'test.txt'
    a = kMeans()
    a.run(filename,4,100)
