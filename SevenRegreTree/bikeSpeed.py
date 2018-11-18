# !/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
from . import regressionTree


def createData(fileName):
    """

    :param fileName: 
    :return: 
    """
    xArr = []
    with open(fileName, 'r') as f:
        numFeat = len(f.readline().split('\t'))

    with open(fileName, 'r') as f:
        for line in f.readlines():
            lineArr = []
            lineList = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(lineList[i]))
            xArr.append(lineArr)
    return xArr

def test():
    """"
    自行车速度与人智力预测
    """

    trainData = np.mat(createData("input/9.RegTrees/bikeSpeedVsIq_train.txt"))
    testData = np.mat(createData("input/9.RegTrees/bikeSpeedVsIq_test.txt"))
    regTree = regressionTree.tree(False,ops=(1,20))
    regTree.fit(trainData)
    print(regTree.tree)
    testPreRegData = np.mat(np.zeros((len(testData),1)))
    for i in range(len(testData)):
        testPreRegData[i,0] = (regTree.predict(testData[:,0][i]))

    # 回归树的R2系数分析
    print("回归树的回归系数: ",np.corrcoef(testPreRegData,testData[:,1],rowvar=0)[0,1])

    modeTree = regressionTree.tree(True, ops=(1, 20))
    modeTree.fit(trainData)
    print(modeTree.tree)
    testPreModeData = np.mat(np.zeros((len(testData), 1)))
    for i in range(len(testData)):
        testPreModeData[i, 0] = (modeTree.predict(testData[:, 0][i]))

    # 模型树的R2系数分析
    print("模型树的回归系数: ", np.corrcoef(testPreModeData, testData[:, 1], rowvar=0)[0, 1])

    ws,X,Y = modeTree.linerSolve(trainData)
    testPreLinerData = np.mat(np.zeros((len(testData), 1)))
    print(ws)
    for i in range(len(testData)):
        testPreLinerData[i,0] = testData[i,0]*ws[1,0] + ws[0,0]

    print("线性模型的回归系数: ", np.corrcoef(testPreLinerData, testData[:, 1], rowvar=0)[0, 1])

    """
    模型树优于回归树优于线性回归
    """
