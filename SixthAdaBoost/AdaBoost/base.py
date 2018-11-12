#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

"""
AdaBoost 算法简介：
    AdaBoost 是脱胎于boosting，属于集成算法。何为集成算法：模型的输出是由多个模型共同作用结果，典型的有bagging 和boosting；
在bagging 中对于最后输出每个模型的权重相同(少数服从多数)，每个模型的生成是并行；在boosting 中对于最后输出每个模型的权重不同，
后一个模型的训练是基于前一个模型的输出，主要关注前个模型输出错误的数据，故boosting 里每个模型的生成是串行。
而AdaBoost 是在boosting 上进一步改进。
    初始化所有数据权重相等，首次分类器输出结果后，对分类错误数据增加权重供下次分类训练；循环操作得到多个分类器，
根据每个分类器的错误率设置对应分类器的权重，然后在测试集上预测 = 每个分类器 * 权重 之和
"""

class adaBoost():
    """
    """
    def __init__(self,numIt = 40):
        """
         :param numIt: 迭代次数既分类器个数；超参
        """
        self.numIt = numIt
        self.weekClassArr = []

    def fit(self,xArr,yArr):
        """
        每次迭代：
            利用buildStump 函数找到最佳单层决策树
            将最佳单层决策树加到决策数组
            计算当前最佳决策树alpha
            计算新数据的权重向量D
            更新累积类别估计值
            如果错误率为0，退出循环
        :param xArr: 
        :param yArr: 
        :return: 
        """
        m = np.shape(xArr)[0]
        D = np.mat(np.ones((m,1))/m)
        agg_class_est = np.mat(np.zeros((m, 1)))
        for i in range(self.numIt):
            print("D:", D.T)
            bestStump,error,classEst = self.buildStump(xArr,yArr,D)
            # 分类器权重；np.max(error,1e-16))为了防止error 太小，发生除0错误
            alpha =float(0.5 * np.log( (1.0 - error)/max(error,1e-16) ) )
            bestStump['alpha'] = alpha
            self.weekClassArr.append(bestStump)
            # 重点来了，更新数据权重,分类错误数据权重增加
            # multiply是对应项相乘
            # 类似于异或，相同0，不同为1，为下面e 的冥次方准备
            expon = np.multiply(-1*alpha* np.mat(yArr).T,classEst)
            # 分类正确e**(-1),分类错误e**(1)，这样导致错误数据权重高，在计算分类器错误率时作用，看buildStump
            D = np.multiply(D,np.exp(expon))
            D = D/D.sum()
            # agg_class_est 其实是最终adaBoost 预测的值
            agg_class_est += alpha*classEst
            # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
            # agg_class_est中数值是浮点数，np.sign修正；np.multiply其实没必要，直接判断"!=" 和为0就是分类正确
            agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(yArr).T,np.ones((m,1)))
            error_rate = agg_errors.sum() / m
            if(error_rate == 0.0):
                # 如果错误率为0，退出循环
                break
        return agg_class_est

    def predict(self,xArr):
        """
        预测
        :param xArr: 
        :return: aggClassEst 预测值
        """
        xMat = np.mat(xArr)
        aggClassEst = np.mat(np.zeros((np.shape(xMat)[0],1)))
        for i in range(len(self.weekClassArr)):
            # 调用弱分类器预测
            week_est = self.stumpClassify(xMat,self.weekClassArr[i]['dim'],self.weekClassArr[i]['thresh'],self.weekClassArr[i]['ineq'])
            aggClassEst += self.weekClassArr[i]['alpha'] * week_est
            print(aggClassEst.T)
        # 同理aggClassEst 是浮点数，还不是分类结果
        return np.sign(aggClassEst)

    def buildStump(self,xArr,yArr,D):
        """
        构建单层最佳决策树，可以替换为其他分类器
        遍历特征
            当前特征分割值逐步递增
                计算当前特征分割值下分类器的错误率
            
        :param xArr: 训练数据特征
        :param yArr: 训练数据标签
        :param D: 训练数据权重
        :return: 最佳决策树
        """
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        m,n = np.shape(xMat)
        # 分割线步进数
        steps = 10.0
        minError = np.inf
        # 最佳决策树相关信息
        bestStump = {}
        bestCast = np.mat(np.zeros((m, 1)))
        # 遍历每个特征选出最佳分割线的特征
        for i in range(n):
            rangeMin = xMat[:,i].min()
            rangeMax = xMat[:,i].max()

            # 求出当前特征范围后，计算具体步进
            step = (rangeMax - rangeMin)/steps
            # 为何要加-1，1；考虑到数据集都是同一类别的情况，阈值需要小于特征最小值或者大于特征最大值
            for j in range(-1,int(steps)+1):
                # 为何分大于和小于，考虑到数据集都是同一类别的情况
                for inequal in ('lt','gt'):
                    # 当前特征分割的阈值，steps 次慢慢计算
                    threshVal = (rangeMin + float(j)*step)
                    predictVals = self.stumpClassify(xMat,i,threshVal,inequal)
                    # 计算错误率
                    errMat = np.mat(np.ones((m,1)))
                    # 分类正确数据 = 0
                    errMat[predictVals == yMat] = 0
                    weightedErr = D.T * errMat
                    if(weightedErr < minError):
                        # 更新最佳决策树
                        minError = weightedErr
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
                        bestCast = predictVals.copy()
        return bestStump,minError,bestCast


    def stumpClassify(self,xMat,indexFeat,threshVal,threshIneq):
        """
        与threshVal 比大小后，根据threshIneq 条件确定类别
        :param xMat: 训练数据集
        :param indexFeat: 特征index
        :param threshVal: 特征阈值
        :param threshIneq: 条件
        :return: 数据集类别数组
        """
        # 初始化所有数据类别相同为1
        retArr = np.ones((np.shape(xMat)[0],1))
        if(threshIneq == 'lt'):
            retArr[xMat[:,indexFeat] <= threshVal] = -1.0
        else:
            retArr[xMat[:, indexFeat] > threshVal] = -1.0
        return retArr