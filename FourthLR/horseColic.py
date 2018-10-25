#! /usr/bin/env python
# -*- coding:utf-8 -*-

import re
import numpy as np

from . import LR


def createData(fliename):
    """
    读取文件数据生成数组
    :param fliename: 文件名
    :return: matrix,labelMatrix:特征、标签数组
    """
    matrix = []
    labelMatrix = []
    f = open(fliename,'r')
    for line in f.readlines():
        currLine = line.strip().split('\t')
        if(len(currLine) < 2):
            continue
        # 0 - 20 特征值
        matrix.append([ float(currLine[i]) for i in range(21) ])
        labelMatrix.append(float(currLine[21]))
    return matrix,labelMatrix

def test():
    """
    病马存活问题
    :return: 
    """
    trainDataMatrix,trainLabelMatrix = createData("input/5.Logistic/HorseColicTraining.txt")
    testDataMatrix, testLabelMatrix = createData("input/5.Logistic/HorseColicTest.txt")

    # LR 模型可以设定 alpha
    model = LR.LogisticRegression(isRandom = True)
    # model = LR.LogisticRegression(alpha=0.01)
    # 设定迭代次数
    model.fit(trainDataMatrix,trainLabelMatrix,1500)
    predictLabels = model.predict(testDataMatrix)

    err = 0.0
    for i in range(len(testLabelMatrix)):
        if(predictLabels[i] != testLabelMatrix[i]):
            err += 1
    print(err,len(testLabelMatrix))
    print("error rate: %f"%(err/len(testLabelMatrix)))