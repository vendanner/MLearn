#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
PCA 技术分析：PCA的数学原理(http://blog.codinglabs.org/articles/pca-tutorial.html)
    数据维度对于机器学习是灾难，维度越大算法花费的时间越久且模型越复杂越容易过拟合。
PCA (Principal Component Analysis)主成分分析是提取数据维度中重要维度的一种技术(在机器学习我们最要的是考究那些才是主要特征)。
PCA 将数据从n维降到k维(k<n)，这样可以降低计算的复杂度。但在降维时我们需要保证两点：
1、数据最大可分性：数据集映射到k维后，数据点不丢失，这需要保证映射到k维超平面后，数据分散既方差最大化
2、最近重构性:数据集映射到k维后，数据点的信息丢失最少，多维数据保证协方差为0(数据不相关)
    我们构造一个对角矩阵即可满足上述两个条件，斜对角是数据方差，其余为数据点之间的协方差
"""

import numpy as np

class PCA:
    """
    
    """
    def __init__(self):
        """
        
        """

    def pca(self,dataSet,topNfeat=9999999):
        """
        去除平均值
        计算协方差矩阵
        计算协方差矩阵的特征值和特征向量
        将特征值从小到大排序
        保留最上面的N个特征向量
        将数据转换到上述N个特征向量构建的新空间中
        :param dataSet: 
        :param topNfeat: 
        :return: 
        """
        dataMat = np.mat(dataSet)
        # 去除平均值
        meanVals = np.mean(dataMat,axis=0)
        meanRemoved = dataMat - meanVals
        # 协方差矩阵
        covMat = np.cov(meanRemoved,rowvar=0)
        # 协方差矩阵的特征值、特征向量
        eigVals,eigVects = np.linalg.eig(np.mat(covMat))
        # 特征值从小到大排序
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:-(topNfeat+1):-1]
        # 保留最上面的N个特征向量
        redEigVects = eigVects[:,eigValInd]
        # 将数据转换到上述N个特征向量构建的新空间中
        lowDataMat = meanRemoved * redEigVects
        # reconMat = k 维数据lowDataMat 在n 维空间的真实值
        reconMat = (lowDataMat * redEigVects.T) + meanVals
        return lowDataMat,reconMat