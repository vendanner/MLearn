#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
SVD 简介：https://redstonewill.com/1529/
    奇异值分解-SVD 是提取信息的强大工具。不同于PCA 单独的提取特征，在SVD 增加主题的想法。在整个数据集中可能包含
很多的主题，SVD 可以将这些主题抽取。假设SVD 在数据集中抽取出10个主题，对比发现前5个主题占比90%，那是否只可以用前5个主题
的数据来表示整个数据集呢？这里就简化数据集，剩下5个主题占比10%就是不重要的主题或噪声(数据压缩就是基于此原理)。
相同的概念：把整个数据集描述成一个矩阵S，矩阵S 可以由很多行列相同的矩阵s累加而成，矩阵s 就是主题。
    奇异值往往对应着矩阵中隐含的重要信息，且重要性和奇异值大小正相关。每个矩阵A都可以表示为一系列秩为1的“小矩阵”之和，
而奇异值则衡量了这些“小矩阵”对于A的权重；样本数据的特征重要性程度既可以用特征值来表征，也可以用奇异值来表征。
    SVD 常用的就是类似PCA 提取重要信息，奇异值 = (DATA * DATA.T)特征值的平方根。下面的代码就是这方面的应用

"""

import numpy as np
import numpy.linalg as la

class svd:
    def __init__(self):
        """
        
        """

    def ecludSim(self,inA,inB):
        """
        计算A、B 两点的距离来判断相似度(0,1)
        :param inA: 
        :param inB: 
        :return: 相似度
        """
        return 1.0/(1.0 + la.norm(inA - inB))

    def pearsSim(self,inA,inB):
        """
        计算A、B 两点的皮尔逊系数来判断相似度(-1,1)
        :param inA: 
        :param inB: 
        :return: 相似度
        """
        if(len(inA) < 3):
            return 1.0
        return 0.5 + 0.5 * np.corrcoef(inA,inB,rowvar=0)[0][1]

    def cosSim(self,inA,inB):
        """
        计算A、B 两点余弦值来判断相似度(-1,1)
        :param inA: 
        :param inB: 
        :return: 相似度
        """
        num = float(inA.T * inB)
        denom = la.norm(inA) * la.norm(inB)
        return 0.5 + 0.5 * (num/denom)

    def standEst(self,dataMat,user,simMeas,item):
        """
        计算当前物品item 综合评价
        :param dataMat: 
        :param user: 
        :param simMeas: 
        :param item: 
        :return: item 评价
        """
        n = np.shape(dataMat)[1]
        simTotal = 0.0
        ratSimTotal =0.0
        for j in range(n):
            userRating = dataMat[user,j]
            if userRating == 0:
                continue
            #  寻找一个用户评价过且与未评价商品有相似的商品
            overLap = np.nonzero(np.logical_and(dataMat[:,item],dataMat[:,j]))[0]
            if len(overLap) == 0 :
                # 相似度为0，不参与未评价商品的评价计算
                similarity = 0
            else:
                similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
                print('the %d and %d similarity is : %f'%(item, j, similarity))
                simTotal += similarity
                # important
                ratSimTotal += similarity * userRating
        if simTotal == 0.0:
            return 0
        else:
            return ratSimTotal / simTotal

    def svdEst(self,dataMat,user,simMeas,item):
        """
        svd简化后再计算当前物品item 综合评价，提高效率
        :param dataMat: 
        :param user: 
        :param simMeas: 
        :param item: 
        :return: item 评价
        """
        n = np.shape(dataMat)[1]
        simTotal = 0.0
        ratSimTotal =0.0
        U,Sigma,VT = la.svd(dataMat)
        Sig4 = np.mat(np.eye(4) * Sigma[:4])
        xformedItems = dataMat.T *U[:,:4] * Sig4.I
        # print("U",U)
        # print("Sigma",Sigma)
        # print("VT",VT)
        # print(Sig4.I)
        # print(dataMat)
        print(xformedItems)
        for j in range(n):
            userRating = dataMat[user,j]
            if userRating == 0 or j == item:
                continue
            similarity = simMeas(self,xformedItems[item,:].T,xformedItems[j,:].T)
            print('the %d and %d similarity is : %f'%(item, j, similarity))
            simTotal += similarity
            # important
            ratSimTotal += similarity * userRating
        if simTotal == 0.0:
            return 0
        else:
            return ratSimTotal / simTotal

    def recommend(self,dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
        """
        推荐前N个对于用户来说评价最高且未评价过的商品
        :param dataMat: 
        :param user: 
        :param N: 
        :param simMeas: 
        :param estMethod: 
        :return: 
        """
        unratedItem = np.nonzero(dataMat[user,:].A == 0)[1]
        if len(unratedItem) == 0:
            # 对于user 用户，所有商品都参与过评价，无法推荐
            return 'you rated everything'
        itemScores = []
        for item in unratedItem:
            # 返回未评价商品item 的评价
            estimatedScore = estMethod(dataMat,user,simMeas,item)
            itemScores.append((item,estimatedScore))
        # 前N 个评价最高的商品
        return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]