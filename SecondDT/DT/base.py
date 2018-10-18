# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
决策树简介：
    decision tree简称DT，通过某个特征将当前测试集一分为二，遍历特征直到训练集全部分类正确或者特征遍历完毕。
这里有个知识点如何判别训练集全部分类正确？本节用信息熵来表达：信息熵越高信息种类越多，信息熵=0表示只有一类信息-->分类正确。
现在算法转换成如何使得特征划分后全部信息熵都为0。
1、遍历n个特征，找出特征分割后信息熵之和为最小的特征W
2、用特征W将数据集分为D1、D2
3、分别在D1、D2数据集遍历n-1个特征(最开始特征W现在参与),找出特征分割后信息熵之和为最小的特征W11，W21
4、用特征W11，W21将数据集D1、D2分为D11、D12和D21、D22
5、继续第三步，递归直到信息熵=0 or 无特征可遍历
"""

import numpy as np
import math
from collections import Counter

class decisionTree ():

    def __init__(self):
        self.majorityLabel = []
        self.y = []
        self.mTree = {}
        print("being decisionTree")

    def majorityCnt(self,labels):
        """
        获取训练集中出现次数最多的分类
        :param labels: 
        :return:null
        """
        # Counter({'no lenses': 15, 'soft': 5, 'hard': 4})
        # no lenses
        self.majorityLabel = [label[0] for label in Counter(labels).most_common(1)]

    def fit(self,X, features):
        """
        训练DT模型
        :param X: 训练集的特征数组
        :param features: 训练集的标签数组
        :return: null
        """
        # 获取labels
        y = set([point[-1] for point in X])
        self.mTree = self.createTree(X, features)

    def predict(self,X):
        """
        输出测试点X，模型输出标签Y
        :param X: 测试点
        :return: Y 返回分类结果
        """
    def getTree(self):
        """
        
        :return: 
        """
        return self.m
    def createTree(self,X, features):
        """
        创建决策树
        :param X: 
        :param features: 
        :return: 决策树
        """
        # 最后一位是标签值
        labels = [point[-1] for point in X]
        # 若y 只剩1个值直接返回 = 训练集全部分类完毕
        if(labels.count(labels[0]) == len(labels)):
            return labels[0]
        # 特征遍历完毕,直接返回训练集中最多的种类
        if(len(X[0]) == 1):
            return self.majorityLabel[0]

        # 寻找bestFeature
        bestFeatIndex = self.getBestFeature(X)
        bestFeat = features[bestFeatIndex]
        mTree = {bestFeat:{}}
        # 删除当前bestFeat，后续不参与划分
        del(features[bestFeatIndex])
        # 去除重复
        uniqueFeaValue = set([example[bestFeatIndex] for example in X])
        # 根据特征划分数据集
        for value in uniqueFeaValue:
            tmpFeatures = features[:]
            # 这里为什么不直接用features[:]？后续不同数据会有不同的支路，剩下的features是不一样的，不能传址
            mTree[bestFeat][value] = self.createTree(self.spiltDataSet(X,bestFeatIndex,value),tmpFeatures)
        return mTree

    def getBestFeature(self,X):
        """
        本函数是决策树模型的核心
        在X数据集中找出信息熵最小的特征，并返回特征的列号
        在本节划分的依据是信息熵，下次会介绍基尼系数来构造决策树
        :param X: 数据集
        :return index: 信息熵最小的特征值列号
        """
        # 去除标签值
        featCounts = len(X[0]) - 1
        trainEntropy = self.calcShannonEnt(X)
        baseEntropy,bestFeatureIndex = 0.0,-1

        # 计算每个特征的信息熵，选出最小
        for i in range(featCounts):
            featValues = set([ example[i] for example in X])
            sumShannonEnt = 0
            for featValue in featValues:
                spiltData = self.spiltDataSet(X,i,featValue)
                prob = len(spiltData) / float(len(X))
                sumShannonEnt += prob * self.calcShannonEnt(spiltData)
            # 找出信息增益最大的特征，信息增益 = 原信息熵 - 划分后的信息熵；故信息增益越大表示划分后的信息熵越小
            infoGain = trainEntropy - sumShannonEnt
            if(infoGain > baseEntropy):
                baseEntropy = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex


    def spiltDataSet(self,X,bestFeatIndex,value):
        """
        在数据集X 中，找出features[bestFeatIndex] = value 的数据重新组合成新的数据集
        :param X: 
        :param bestFeatIndex: 特征列号 
        :param value: 特征值
        :return: features[bestFeatIndex] = value 的数据集
        """
        spiltData = [example[:bestFeatIndex] + example[bestFeatIndex + 1:] for example in X  if example[bestFeatIndex] == value ]
        # spiltData = [example[:bestFeatIndex]+example[bestFeatIndex+1:] for example in X for i,v in enumerate(example) if (i == bestFeatIndex) and (v == value) ]
        return spiltData

    def calcShannonEnt(self,X):
        """
        计算数据集X的信息熵
        :param X: 
        :return: 数据集信息熵
        """
        labelCounter = Counter(point[-1] for point in X)
        probs = [labelCnt[1]/len(X) for labelCnt in labelCounter.items()]
        # 信息熵计算公式：sum( -p(x) * log(p(x)) )
        shannonEnt = sum([ -prob * math.log(prob,2) for prob in probs ])
        return shannonEnt
