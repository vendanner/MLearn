# !/usr/bin/env python
# -*- coding:utf-8 -*-

""""
K-Means 算法简介：
    K-Means 是无监督算法既事先没有标注，聚类算法的一种。聚类和分类最大的区别在于，分类的目标是已知，而聚类是未知。
K-Means 可以将无序的数据按质心(每个类的中心点)分成几类数据。假设我们将数据集分成K类，则计算每个数据与K个质心的距离，
距离哪个质心最短则分成哪类。由此可见，与KNN 算法类似，距离的衡量成为K-Means 算法关键。
1、随机选择K个质心
2、遍历整个数据集，根据质心将数据分为K 类
3、计算K类数据的中心点重新设为质心
4、重复2-3步骤，所有数据点的聚类结果不再改变

    K-Means 缺陷：
非球形簇，或者多个簇之间尺寸和密度相差较大的情况，KMeans 就处理不好了
K-Means 是局部最优解，则对于初始的质心选择至关重要；二分K-Means 是最优解，本节代码基于此实现
需要指定超参K，K的取值需要进行缜密的计算且非常重要；谱聚类算法不需要超参K
"""

import numpy as np

class base:
    """
    
    """
    def __init__(self):
        """
        
        """

    def distEclud(self,vecA,vecB):
        """
        计算数据到质点的距离 = 标准差
        :param vecA: 
        :param vecB: 
        :return: 距离值
        """
        return np.sqrt(np.sum(np.power(vecA,vecB,2)))

    def randCent(self,dataSet,k):
        """
        在数据集范围内，随机产生k 个质点
        :param dataSet: 
        :param k: 
        :return: k 个质点,centroids
        """
        n = np.shape(dataSet)[1]
        centroids = np.mat(np.zeros((k,n)))
        for i in range(n):
            minJ = np.min(dataSet[:,i])
            rangeJ = float(np.max(dataSet[:,i]) - minJ)
            # np.random.rand(k,1) 在[0,1] 范围内生成 k 个随机值
            centroids[:,i] = minJ + rangeJ * np.random.rand(k,1)
        return centroids

    def kMeans(self,dataSet,k,distMeas = distEclud,createCent = randCent):
        """
        K-Means 算法
        :param dataSet: 
        :param k: 
        :param distMeans: 计算距离函数
        :param createCent: 质点初始化函数
        :return: 质点，数据集聚类结果
        """
        m = np.shape(dataSet)[0]
        centroids = createCent(dataSet,k)
        # 数据集聚类结果存储：聚类类别，到质点的距离
        clusterAssment = np.mat(np.zeros((m,2)))
        clusterChanged = True

        while(clusterChanged):
            clusterChanged = False
            # 给数据集中的每个点划分类别
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                # 寻找当前数据点的类别 = 距离最近的质点
                for j in range(k):
                    distJ = distMeas(centroids[j,:],dataSet[i,:])
                    if(distJ < minDist):
                        minDist = distJ
                        minIndex = j
                #  更新当前数据点的类别
                if(clusterAssment[i,0] != minIndex):
                    clusterAssment[i,:] = minIndex,minDist**2
                    # 有数据点的类别被更改
                    clusterChanged = True
            # 更新簇群的质点
            for cent in range(k):
                centroids[cent,:] = np.mean(dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]],axis=0)
        return centroids,clusterAssment

    def biKmeans(self,dataSet,k,distMeas = distEclud):
        """
        二分K-Means 算法，计算全局SSR 最优解
        将所有数据看作一个簇
        当簇数据小于k：
            遍历每一个簇：
                计算数据总误差
                在当前簇进行K-Means 聚类(k=2)
                计算当前簇二分之后的数据总误差
            选择使得数据总误差最小的簇进行划分
        :param dataSet: 
        :param k: 
        :param distMeas: 
        :return: 
        """
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m,2)))
        # 所有数据当作一个簇时的质点
        centroid0 = np.mean(dataSet,axis=0).tolist()[0]
        centList = [centroid0]
        # 所有数据当作一个簇时的数据聚类结果
        for i in range(m):
            clusterAssment[i,1] = distMeas(np.mat(centroid0),dataSet[i,:]) **2

        while(len(centList) < k):
            # 其实这里应该是计算数据总误差
            # lowestSSE = np.sum(clusterAssment[:,1])
            lowestSSE = np.inf
            for i in range(len(centList)):
                centroidMat,splitAssment = self.kMeans(dataSet[np.nonzero(clusterAssment[:,0].A == i)[0]],2,distMeas)
                sseSplit = np.sum(splitAssment[:,1])
                nosseSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])
                # 更新划分后的数据总误差
                if((sseSplit + nosseSplit) < lowestSSE):
                    lowestSSE = sseSplit + nosseSplit
                    # bestCentToSplit要被划分的簇群
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitAssment.copy()
            # 更新簇群标识，k-means 后簇群标识0-1，要改为bestCentToSplit，len(centList)
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
            print('the bestCentToSplit is: ', bestCentToSplit)
            print('the len of bestClustAss is: ', len(bestClustAss))
            # 更新bestCentToSplit 的质心
            centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
            # 添加len(centList) 簇群
            centList.append(bestNewCents[1,:].tolist()[0])
            # 更新未分割前bestCentToSplit 群那些数据
            clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
        return np.mat(centList),clusterAssment