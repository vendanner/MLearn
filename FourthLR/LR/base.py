# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Logistic Regression(逻辑回归)简介：
逻辑回归是分类模型，它依据分类的概率得出最后的分类结果(与贝叶斯类似)
本节选择的回归模型1/1+e^(-z),z 是线性回归模型g(x) = w^T*x + b，这就是为什么会包含回归二字原因。
求解过程与线性回归类似，得到损失函数并求解极值；但不同的是损失函数的构造有差异，详细参看斯坦福2014机器学习-逻辑回归
求解极值的过程就是梯度上升(线性回归是梯度下降)，w 沿着导数的方向一步步调整
w = w + a * sum(g(x) - y)* x ; a(超参) = 每次调整的步长，步长越小越能精确得到极值但时间消耗越长
"""

import numpy as np
import math

class LogisticRegression():

    def __init__(self,alpha = 0.05,isRandom = False):
        """
        :param alpha: 调整的步长
        :param isRandom: 梯度上升随机样本标志
        """
        self.isRandom = isRandom
        self.alpha = alpha
        self.weights = []

    def fit(self,X,y,maxCycles=150):
        """
        训练模型
        :param X:训练集特征数组 
        :param y: 训练集标签数组
        :return: 无
        """
        if(self.isRandom):
            self.random_grad_ascent(np.array(X), np.array(y), maxCycles)
        else:
            # np.mat 才是矩阵
            self.grad_ascent(np.mat(X), np.mat(y),maxCycles)


    def predict(self,X):
        """
        预测测试集概率
        :param X: 测试集特征数组
        :return: 标签数组
        """
        # 概率大于0.5 为1，小于0.5为0
        return  [1 if predictVal[0] > 0.5 else 0 for predictVal in self.sigmoid(np.mat(X) * self.weights).tolist()]

    def grad_ascent(self,X, y,maxCycles):
        """
        梯度上升算法求解回归系数
        :param X:训练集特征数组 
        :param y: 训练集标签数组
        :param maxCycles: 梯度上升迭代次数
        :return: 
        """
        m,n = np.shape(X)
        # 初始化回归系数
        weights = np.ones((n,1))
        print(np.shape(X))
        print(np.shape(weights))
        for i in range(maxCycles):
            # 这行代码就是梯度上升的公式
            weights = weights + self.alpha * X.transpose() * (y.transpose() - self.sigmoid(X * weights))
        self.weights = weights
        print("grad_ascent",self.weights.shape)

    def random_grad_ascent(self,X, y,maxCycles):
        """
        1、遍历随机一个样本去更新weight而不是一次性选择所有样本
        2、alpha 不是固定值，逐渐减小;比采样固定alpha 收敛速度快
        :param X:训练集特征数组 
        :param y: 训练集标签数组
        :param maxCycles: 梯度上升迭代次数
        :return: 
        """
        m,n = np.shape(X)
        # 初始化回归系数
        weights = np.ones(n)
        print(weights)
        for j in range(maxCycles):
            dataIndex = list(range(m))
            # 遍历随机一个样本去更新weight
            for i in range(m):
                # alpha 不是固定值，逐渐减小
                alpha = 4/(1.0 + j +i) + 0.01
                randIndex = int(np.random.uniform(0,len(dataIndex)))
                # 这行代码就是梯度上升的公式
                weights = weights + alpha * X[dataIndex[randIndex]] * (y[dataIndex[randIndex]] - self.sigmoid(np.sum(X[dataIndex[randIndex]] * weights)))
                del(dataIndex[randIndex])

        self.weights = np.mat(weights).transpose()



    def sigmoid(self,z):
        """
        sigmoid 函数
        :param z: 
        :return: 概率值
        """
        return 1.0/(1+np.exp(-z))
