# !/usr/bin/env python
# -*- coding:utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
from numpy import *

from . import KNN

def fileToMatrix(fileName):
    """
    将数据转化成矩阵便于后续计算
    :param fileName: 数据文件名
    :return: matrix: 特征值的矩阵
            labels：测试集对应的分类结果
    """
    dataSet = open(fileName,'r')
    lines = len(dataSet.readlines())
    # 三个特征值，矩阵为lines * 3
    matrix = zeros((lines,3))
    labels = []
    index = 0
    #  important;之前readlines后，dataSet没缓存了需要重新读文件
    dataSet = open(fileName, 'r')
    for line in dataSet.readlines():
        strLine = line.strip()
        strLines = strLine.split('\t')
        # 特征值
        matrix[index] = strLines[0:3]
        labels.append(int(strLines[-1]))
        index += 1
    return matrix,labels


def datingClass(isAutoNorm):
    """
    特征没有做归一化处理导致影响模型准确率
    :param isAutoNorm: 特征是否做归一化
    :return: 
    """
    # 载入数据
    datingMat,datingLabels = fileToMatrix("input/2.KNN/datingTestSet2.txt")
    print(datingMat,datingLabels)
    # 特征值归一化处理
    if(isAutoNorm):
        datingMat = KNN.autoNorm(datingMat)
    print(datingMat)

    # 测试样本 = numTest,训练样本  = numTest:allNum
    allNum = datingMat.shape[0]
    numTest = int(allNum*0.1)
    errNo = 0
    for i in range(numTest):
        # 当前k = 7
        label = KNN.knnModel(datingMat[i],datingMat[numTest:allNum],datingLabels[numTest:allNum],7)
        print("the classifier came back with: %d, the real answer is: %d" % (label, datingLabels[i]))
        errNo += (label != datingLabels[i])
    print("the total error rate is: %f" % (errNo / numTest))
    print(errNo)

def test():

    # 特征没有做归一化处理导致影响模型准确率
    # 测试发现有无进行归一化处理，错误率相差4-5倍
    datingClass(False)
