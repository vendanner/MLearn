#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
svm 算法简介：
    svm 算法旨在众多线性模型中找到最优的模型(健壮性好 - 容错高)。但不同于线性模型的求解过程是找最小误差，
svm 求support vector 的alpha。svm 的涉及到东西挺多，详细参考：https://blog.csdn.net/v_JULY_v/article/details/7624837。
    smo 是求svm 中alpha 的一种方法。
"""

import random
import numpy as np

class plattsmo():
    def __init__(self,C,toler):
        """
        
        :param C: 
        :param toler: 容忍误差
        """
        self.C = C
        self.toler = toler
        self.w = []
        self.b = 0
        self.alphas = []
        self.eCache = []
        self.ws = []
        self.K = []

    def fit(self,data,label,maxIter,kTup=('lin',0)):
        """
        
        :param data: 
        :param label: 
        :param maxIter: 
        :param kTup: 
        :return: 
        """
        self.smoPla(data,label,maxIter,kTup)
        self.calcWs(data,label)
    def predict(self,data):
        """
        
        :param data: 
        :return: 
        """
        return np.mat(data) * self.ws + self.b

    def smoPla(self, data, label, maxIter, kTup=('lin', 0)):
        """
        把simpleSMO 拆成 smoPla、interL 两个函数

        :param dataMat: 
        :param labelMat: 
        :param maxIter: 
        :param kTup: 
        :return: 
        """
        iter = 0
        dataMat = np.mat(data)
        labelMat = np.mat(label).T
        m, n = np.shape(dataMat)
        # 初始化变量
        self.alphas = np.mat(np.zeros((m, 1)))
        self.eCache = np.mat(np.zeros((m, 2)))
        self.K = np.mat(np.zeros((m, m)))
        for i in range(m):
            self.K[:, i] = self.kernelTrans(dataMat, dataMat[i, :], kTup)

        entireSet = True
        alphaPairsChange = 0
        # 迭代次数；遍历所有数据都没有调整；满足任意条件则停止
        while ((iter < maxIter) and ((alphaPairsChange > 0) or (entireSet))):
            # 第一个数据选择
            alphaPairsChange = 0
            if entireSet:
                # 与simpleSMO 相同，遍历所有数据
                for i in range(m):
                    alphaPairsChange += self.interL(i, dataMat, labelMat)
                    # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChange))
                iter += 1
            else:
                # 遍历support vector 数据
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChange += self.interL(i, dataMat, labelMat)
                    # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChange))
                iter += 1

            if entireSet:
                entireSet = False
            elif (alphaPairsChange == 0):
                entireSet = True
        print("iteration number: %d" % iter)

    def calcEk(self,dataMat,labelMat,k):
        """
        计算第k 个数据的误差
        :param dataMat: 
        :param labelMat: 
        :param k: 
        :return: 误差
        """
        # fxk = float(np.multiply(self.alphas,labelMat).T*(dataMat*dataMat[k,:].T)) + self.b
        # 增加核函数;将X * Xi.T替换为核函数
        fxk = float(np.multiply(self.alphas, labelMat).T * self.K[:,k]) + self.b
        Ek = fxk - float(labelMat[k])
        return Ek

    def selectJ(self,i,dataMat,labelMat,Ei):
        """
        选择另一个数据
        :param i: 
        :param dataMat: 
        :param labelMat: 
        :param Ei: 
        :return: 另一个数据序列号和误差
        """
        maxK = -1
        maxDeleteE = 0
        Ej = 0
        self.eCache[i] = [1,Ei]
        # 有计算误差E 的数据点
        validEcacheList = np.nonzero(self.eCache[:,0].A)[0]
        if len(validEcacheList) > 1 :
            # 选择误差相差最大的
            for k in validEcacheList:
                if k == i :
                    continue
                Ek = self.calcEk(dataMat,labelMat,k)
                deltaE = abs(Ek - Ei)
                if(deltaE > maxDeleteE):
                    maxDeleteE = deltaE
                    maxK = k
                    Ej = Ek
            return maxK,Ej
        else:
            # 随机选
            m = np.shape(dataMat)[0]
            j = i
            while j == i:
                # 序列号应该为int
                j = random.randint(0,m)
            Ej = self.calcEk(dataMat,labelMat,j)

        return j,Ej
    def updateEk(self,dataMat,labelMat,k):
        """
        
        :param dataMat: 
        :param labelMat: 
        :param k: 
        :return: 
        """
        Ek = self.calcEk(dataMat,labelMat,k)
        # 1 表示eCache 有效
        self.eCache[k]= [1,Ek]

    def clipAlpha(self,aj,H,L):
        """
        alpha 约束：L < alpha < H
        :param aj: 
        :param H: 
        :param L: 
        :return: 
        """
        if(aj > H):
            aj = H
        if(aj < L):
            aj = L
        return aj

    def kernelTrans(self,X,A,kTup):
        """
        核变换;后续计算将内积X * Xi.T替换为核函数
        :param X: 数据集
        :param A: 单个数据
        :param kTup: 核函数信息
        :return: 
        """
        m,n = np.shape(X)
        K = np.mat(np.zeros((m,1)))
        if(kTup[0] == 'lin'):
            # 线性核
            K = X*A.T
        elif (kTup[0] == 'rbf'):
            # 高斯核
            for i in range(m):
                deltaRow = X[i,:] - A
                K[i] = deltaRow * deltaRow.T
            K = np.exp(K/(-1*kTup[1]**2))
        else:
            raise NameError('Houston We Have a Problem -- That Kernel is not recognized')

        return K

    def interL(self,i,dataMat,labelMat):
        """
        内循环，选择第二个数据一起更新alpha
        相比simpleSMO：
            1、选择第二个数据机制
            2、增加E cache
        :param i: 
        :param dataMat: 
        :param labelMat: 
        :return: 
        """
        # i 数据点的预测值
        Ei = self.calcEk(dataMat,labelMat,i)
        # 该数据不满足KKT，需要调整该数据的alpha
        # 分类错误点，alpha > C；原本alpha=C
        # 分类正确点但不是support vector， alpha>0 ；原本alpha=0
        # support vector，0 < alpha< C
        if (((labelMat[i] * Ei < -self.toler) and (self.alphas[i] < self.C)) or ((labelMat[i] * Ei > self.toler) and (self.alphas[i] > 0))):
            j,Ej= self.selectJ(i,dataMat,labelMat,Ei)

            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            # 保证0 < alpha < C
            # 下面if 操作就是smo 更新alpha 的核心，具体原理看https://blog.csdn.net/v_JULY_v/article/details/7624837
            if (labelMat[j] != labelMat[i]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = max(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                print("L = H")
                return 0
            # eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j,
            #                                                                                                 :].T
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j]
            if (eta >= 0):
                print("eta >= 0")
                return 0

            # 更新alpha
            self.alphas[j] -= labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
            self.updateEk(dataMat,labelMat,j)

            # alpha 更新幅度小则不更新
            if (abs(self.alphas[j] - alphaJold) < 0.00001):
                # print("j not moving enough")
                return 0
            self.alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - self.alphas[j])
            self.updateEk(dataMat,labelMat,i)

            # 更新b
            b1 = self.b - Ei - labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i,i] - labelMat[j] * (
                self.alphas[j] - alphaJold) * self.K[i,j]
            b2 = self.b - Ej - labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i,j] - labelMat[j] * (
                self.alphas[j] - alphaJold) * self.K[j,j]
            # 选support vector 的b
            # 若2个数据都不是support vector 平均值作为b
            if ((0 < self.alphas[i]) and (self.C > self.alphas[i])):
                b = b1
            elif ((0 < self.alphas[j]) and (self.C > self.alphas[j])):
                b = b2
            else:
                b = (b1 + b2) / 2

            return 1
        return 0



    def calcWs(self,data,label):
        """
        计算ws
        :param data: 
        :param label: 
        :return: 
        """
        dataMat = np.mat(data)
        labelMat = np.mat(label).T
        m, n = np.shape(dataMat)
        self.ws = np.mat(np.zeros((n,1)))
        for i in range(m):
            self.ws += np.multiply(self.alphas[i]*labelMat[i],dataMat[i,:].T)



class simplesmo():
    """
    简单smo，简化了某些步骤，导致运行效率低
    1、遍历每个数据量而不是support vector
    2、第二个数据是随机选择而不是选择距离最大
    """
    def __init__(self):
        print("smo init")


    def selectJrand(self,i,m):
        """
        在m 个数据随机选择一个不同于i 的数据
        :param i: 
        :param m: 
        :return: 
        """
        j = i
        while(j == i):
            j = random.uniform(0,m)
        return j



    def simpleSmo(self,data,label,C,toler,maxIter):
        """
        简单smo 
        迭代次数maxIter
            遍历每一个数据向量
                如果该数据可以被优化(不满足KKT)
                    随机选择另一个数据
                    同时优化这2个数据
                    如果数据不能被优化，退出当前循环
            如果所有数据都不能被优化，退出当前迭代进入下个迭代(数据连续不能被优化迭代次数>maxIter,表示找到最优alpha，return)
        :param data: 
        :param label: 
        :param C: 
        :param toler: 
        :param maxIter: 
        :return: 
        """
        dataMat = np.mat(data)
        labelMat = np.mat(label).T
        m,n = np.shape(dataMat)
        # b 初始化为0
        b = 0
        # 每个数据集的alpha 都初始化为0
        alphas = np.mat(np.zeros((m,1)))
        iter = 0
        while(iter < maxIter):
            alphaPairsChange = 0
            # 遍历每个数据集，更改alpha
            for i in range(m):
                # i 数据点的预测值
                fxi = float((np.multiply(alphas,labelMat).T) * (dataMat*dataMat[i,:].T))+b
                Ei = fxi - float(labelMat[i])
                # 该数据不满足KKT，需要调整该数据的alpha
                # 分类错误点，alpha > C；原本alpha=C
                # 分类正确点但不是support vector， alpha>0 ；原本alpha=0
                # support vector，0 < alpha< C
                if( ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0))):
                    j = self.selectJrand(i,m)
                    fxj = float((np.multiply(alphas, labelMat).T) * (dataMat * dataMat[j, :].T)) + b
                    Ej = fxj - float(labelMat[j])

                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    # 保证0 < alpha < C
                    # 下面if 操作就是smo 更新alpha 的核心，具体原理看https://blog.csdn.net/v_JULY_v/article/details/7624837
                    if(labelMat[j] != labelMat[i]):
                        L = max(0,alphas[j]-alphas[i])
                        H = min(C,C + alphas[j] - alphas[i])
                    else:
                        L = max(0,alphas[j]+alphas[i]-C)
                        H = max(C,alphas[j] + alphas[i])
                    if L == H:
                        print("L = H")
                        continue
                    eta = 2.0*dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                    if (eta >= 0):
                        print("eta >= 0")
                        continue

                    # 更新alpha
                    alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                    alphas[j] = self.clipAlpha(alphas[j],H,L)
                    # alpha 更新幅度小则不更新
                    if(abs(alphas[j] - alphaJold) < 0.00001):
                        print("j not moving enough")
                        continue
                    alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                    # 更新b
                    b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMat[i,:]*dataMat[j,:].T
                    b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMat[j,:]*dataMat[j,:].T
                    # 选support vector 的b
                    # 若2个数据都不是support vector 平均值作为b
                    if((0 < alphas[i]) and (C > alphas[i])):
                        b = b1
                    elif((0 < alphas[j]) and (C > alphas[j])):
                        b = b2
                    else:
                        b = (b1+b2)/2

                    alphaPairsChange += 1
                    print("iter : %d,i : %d,pairs changed %d",iter,i,alphaPairsChange)
            if(alphaPairsChange == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d",iter)
        return b,alphas



