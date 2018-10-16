#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
识别手写数字
1、手写数字的图形文件转化成matrix
2、将test matrix 代入KNN模型 输出test 的分类结果:0-9
"""


import os
import numpy as np
import time

from . import KNN

def fileToMatrix(fileName):
    """
    文件映射到一维数组;32*32
    :param fileName: 文件名
    :return: matrix：一维数组
    """
    matrix = np.zeros((1,1024))
    f = open(fileName,'r')
    for i in range(32):
        str = f.readline()
        for j in range(32):
            matrix[0,32*i + j] =int( str[j])

    return matrix


def test():
    # 数据文件转为矩阵
    trainingDir = os.listdir("input/2.KNN/trainingDigits")
    trainingSize = len(trainingDir)
    trainingLabel =[]
    # 图形像素点是32*32 = 1024
    trainingMatrix = np.zeros((trainingSize,1024))
    for i in range(trainingSize):
        fileStr = trainingDir[i]
        trainingLabel.append(int(fileStr.split('_')[0]))
        trainingMatrix[i] = fileToMatrix("input/2.KNN/trainingDigits/%s"%fileStr)

    testDir = os.listdir("input/2.KNN/testDigits")
    testSize = len(testDir)
    testLabel = []
    # 图形像素点是32*32 = 1024
    testMatrix = np.zeros((testSize, 1024))
    for i in range(testSize):
        fileStr = testDir[i]
        testLabel.append(int(fileStr.split('_')[0]))
        testMatrix[i] = fileToMatrix("input/2.KNN/testDigits/%s" % fileStr)


    # 预测
    start = time.time()
    errNo = 0
    for i in range(testSize):
        # k 是超参
        label = KNN.knnModel(testMatrix[i],trainingMatrix,trainingLabel,5,False)
        errNo += (label != testLabel[i])
    # 对于本案例来说，计算距离dist 欧几里方式比异或多3倍时间
    print("running time = ",time.time() - start)

    # 输出错误率
    print("the total error rate is: %f" % (errNo / testSize))
    print(errNo)
