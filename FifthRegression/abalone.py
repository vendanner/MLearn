# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

from . import  Regression

def createData(fileName):
    """
    读取文件数据
    :param fileName: 
    :return: dataMat,labelMat
    """
    dataMat = []
    labelMat = []
    f = open(fileName,'r')
    for line in f.readlines():
        strLine = line.strip().split('\t')
        dataMat.append([ (float)(strLine[i]) for i in range(8)])
        labelMat.append((float)(strLine[8]))
    return np.array(dataMat),np.array(labelMat)

def test():
    print("abalone test")
    dataMat,labelMat = createData("input/8.Regression/abalone.txt")
    # 测试集占比1%
    testIndx = (int)(len(labelMat) * 0.99)
    trainData = dataMat[0:testIndx]
    trainLabel = labelMat[0:testIndx]
    testData = dataMat[testIndx:len(labelMat)]
    testLabel = labelMat[testIndx:len(labelMat)]

    # 选取不同k 生产不同权重，对模型影响
    # lwlr1 = Regression.lwlr(0.1)
    # lwlr2 = Regression.lwlr(1)
    # lwlr3 = Regression.lwlr(10)
    # # 训练集预测
    # trainy1 =[]
    # trainy2 = []
    # trainy3 = []
    # for i in range(testIndx):
    #     trainy1.append(lwlr1.predict(trainData,trainLabel,trainData[i]))
    #     trainy2.append(lwlr2.predict(trainData, trainLabel, trainData[i]))
    #     trainy3.append(lwlr3.predict(trainData, trainLabel, trainData[i]))
    #
    # # 测试集预测
    # testy1 = []
    # testy2 = []
    # testy3 = []
    # for i in range(len(labelMat) - testIndx):
    #     testy1.append(lwlr1.predict(trainData, trainLabel, testData[i]))
    #     testy2.append(lwlr2.predict(trainData, trainLabel, testData[i]))
    #     testy3.append(lwlr3.predict(trainData, trainLabel, testData[i]))
    # print(type(trainLabel))
    # print(type(trainy1))
    # print("train data k = 0.1 rssError :",((trainLabel - trainy1)**2).sum())
    # print("train data k = 1 rssError :", ((trainLabel - trainy1) ** 2).sum())
    # print("train data k = 10 rssError :", ((trainLabel - trainy1) ** 2).sum())
    #
    # print("test data k = 0.1 rssError :",((testLabel - testy1)**2).sum())
    # print("test data k = 1 rssError :", ((testLabel - testy2) ** 2).sum())
    # print("test data k = 10 rssError :", ((testLabel - testy3) ** 2).sum())

    testStage = []
    stageMode = Regression.stageWise()
    stageMode.fit(stageMode.regularize(dataMat),(np.mat(labelMat).T - np.mean(np.mat(labelMat).T,0)),200)
    for i in range(len(labelMat)):
        testStage.append(stageMode.predict(dataMat[i]))
    print(" rssError :", ((testStage - labelMat) ** 2).sum())

