# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
朴素贝叶斯简介：
Bayesian：假设分类结果总计三种，计算测试点对于每个分类的概率，P1，P2,P3,将概率值最大的分类当作测试点的分类结果。
P(xy) = P(x|y)P(y) = P(y|x)P(x)
==> P(x|y) = P(y|x)P(x)/P(y)
Bayesian 算法都是基于上面的公式演化而来。
有特征值x1,x2,y1,y2,分类结果C1,C2；假设P(x1|C1) = P1,P(y1|C1) = P2,P(x1|C2) = P3,P(y1|C2) = P4,P(x2|C1) = P5,
P(y2|C1) = P6,P(x2|C2) = P7,P(y2|C2) = P8。则对于测试点(x1,y2)分类结果为C1 =>P(C1|x1,y2) = P1*P6*P(C1)，
分类结果为C2 => P(C2|x1,y2) = P3*P8*P(C2);若P1*P6*P(C1) >P3*P8*P(C2) 测试点(x1,y2) = C1,
P1*P6*P(C1) < P3*P8*P(C2)测试点(x1,y2) = C2。
上面只是一个简单的案例，但从这里我们可得知在Bayesian 中我们需要计算P(x1|C1)、P(x1|y2)、P(C1)，我们称P(C1) 为 先验概率，
P(x1|C1)、P(x1|y2) 为概率因子，P(C1|x1,y2)为后验概率 => 后验概率 = 先验概率 * 概率因子。我们首先得到先验概率P(C1)，
然后不断实验后得到概率因子P(x1|C1)、P(x1|y2)，即可得到后验概率既当前条件下C1的真实概率
朴素贝叶斯公式：P(y|W1,W2..Wn) = P(y)*P(W1|y)*P(W2|y)...*P(Wn|y)
注意：朴素贝叶斯公式成立的基础是特征W1,W2..Wn 相互独立
朴素贝叶斯分类器，它是一个分类模型，它的模型函数是朴素贝叶斯公式——贝叶斯定理在所有特征全部独立情况下的特例
"""

import numpy as np

class naiveBayes ():
    """
    朴素贝叶斯分类
    """

    def __init__(self):
        """
        :return: model
        """
        self.totalSum = 0
        self.intCount = 0
        # 特征
        self.featureList = []
        self.X = []
        self.y = []
        # 概率因子
        self.pSpamMatrix = []
        self.pHamMatrix = []
        # 先验条件
        self.pSpam = 0

    def fit(self,X,y,featureList,initSum = 2,intCount = 1):
        """
        naive bayes 训练
        :param X: 训练集特征数组
        :param y: 训练集标签数组
        :param featureList: 特征数组
        :param initSum: 初始化训练集总数
        :param intCount: 初始化训练集中特征数
        :return: model
        """
        self.totalSum = initSum
        self.intCount = intCount
        # 特征
        self.featureList = featureList
        self.X = X
        self.y = y

        # 数据转化成特征值数组
        dataMatrix = self.trainDataToMatrix()
        self.naiveBayesTrain(dataMatrix)
        print(self.pSpamMatrix)
        print(self.pHamMatrix)
        print(self.pSpam)



    def predict(self, X):
        """
        输出测试点X，模型输出标签Y
        :param X: 测试点
        :return: Y 返回分类结果
        """
        return  self.naiveBayesPredict(X)

    def trainDataToMatrix(self):
        """
        训练集数据转化成特征值数组
        :return: 特征值数组
        """
        dataMatrix = []
        for data in self.X:
            dataMatrix.append(self.dataToMatrix(data))

        return dataMatrix

    def dataToMatrix(self,data):
        """
        单个数据点转化成特征值数组
        :param data: 
        :return: 
        """
        matrix = np.zeros(len(self.featureList))
        for word in data:
            if word in self.featureList:
                # 单词出现计数1
                matrix[self.featureList.index(word)] = 1
        return matrix

    def naiveBayesTrain(self,matrix):
        """
        数据集通过naivebayes 训练后得到概率因子和先验条件
        :param matrix: 训练集数据
        :return: 
        """

        # 给初始值防止数值太小后续无法计算，也减少某类特征值带来的误差(若某类特征在ham 都没有出现，则该特征会对分类起决定性的作用，这显然不符合我们的预期)
        # 加一平滑法（就是把概率计算为：对应类别样本中该特征值出现次数 + 1 /对应类别样本总数）
        spamMatrix = np.zeros(len(self.featureList)) + self.intCount
        hamMatrix = np.zeros(len(self.featureList)) + self.intCount
        spamSum = self.totalSum
        hamSum = self.totalSum

        for i in range(len(matrix)):
            # spam 邮件
            if(self.y[i] == 1):
                # 统计word 在spam 邮件中出现次数
                spamMatrix += matrix[i]
                spamSum += 1
            # ham 邮件
            else:
                hamMatrix += matrix[i]
                hamSum += 1
        # 计算概率因子p(x|y) ; 为何要加上log 是为了后续计算后验概率时数值太小不利于计算(后续计算是连乘)
        self.pSpamMatrix = np.log(spamMatrix / spamSum)
        self.pHamMatrix = np.log(hamMatrix / spamSum)
        # spam 标签为1 ham 为0 ，spam 邮件个数即为数组值之和
        self.pSpam = np.sum(self.y) / len(self.y)

    def naiveBayesPredict(self,X):
        """
        naive bayes 返回预测值
        :param X: 测试数据
        :return: 
        """
        preHam = 0
        preSpam = 0
        matrix = self.dataToMatrix(X)
        # 数据中出现的特征值  概率因子相加;乘以0 = 0
        #  P(y | W1, W2..Wn) = P(y)*P(W1|y)*P(W2|y)...*P(Wn|y) = logP(y)+logP(W1|y)*logP(W2|y)... * logP(Wn|y)
        preHam = np.sum(matrix * self.pHamMatrix) + (1 - self.pSpam)
        preSpam = np.sum(matrix * self.pSpamMatrix) + self.pSpam

        if preSpam > preHam:
            return 1
        else:
            return 0