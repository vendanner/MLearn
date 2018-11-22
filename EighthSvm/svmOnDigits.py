#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import numpy as np

from . import svm

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


def loadImage(dirName):
    """
    
    :param dirName: 目录
    :return: 
    """
    hwLabels = []
    fileList = os.listdir(dirName)
    m = len(fileList)
    hwData = np.zeros((m,1024))
    for i in range(m):
        fileStr = fileList[i].split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        # 分2 类?
        if(classNumStr == 9):
            hwLabels.append(-1)
        else:
            hwLabels.append(1)

        hwData[i,:] = fileToMatrix("%s/%s"%(dirName,fileList[i]))

    return hwData,hwLabels

def test():
    """
    svm 分类模型识别手写数字
    :return: 
    """
    trainData,trainLabel = loadImage("input/6.SVM/trainingDigits")
    testData, testLabel = loadImage("input/6.SVM/testDigits")
    trainDataMat = np.mat(trainData)
    trainLabelMat = np.mat(trainLabel).T
    testDataMat = np.mat(testData)
    testLabelMat = np.mat(testLabel).T

    kernelList = []
    kernelList.append(('rbf',0.1))
    kernelList.append(('rbf', 5))
    kernelList.append(('rbf',10))
    kernelList.append(('rbf', 50))
    kernelList.append(('rbf', 100))
    kernelList.append(('lin',0))
    for j in range(len(kernelList)):
        # 不同高斯核参数与线性核错误率比对
        print("*****************************************************************")
        print("%s,%f:"%(kernelList[j][0],kernelList[j][1]))
        mode = svm.plattsmo(200,0.0001)
        mode.fit(trainData,trainLabel,1000,kernelList[j])

        svInd = np.nonzero(mode.alphas > 0)[0]
        sVS = trainDataMat[svInd]
        labelSv = trainLabelMat[svInd]
        print("there are %d Support Vectors" % np.shape(sVS)[0])
        m,n = np.shape(trainDataMat)
        errCount = 0
        for i in range(m):
            # 用alpha 求值 y
            kernelEval = mode.kernelTrans(sVS,trainDataMat[i,:],kernelList[j])
            predict = kernelEval.T * np.multiply(labelSv,mode.alphas[svInd]) + mode.b
            if(np.sign(predict) != np.sign(trainLabel[i])):
                errCount += 1
        print("the training error rate is: %f" % (float(errCount) / m))


        errCount = 0
        m, n = np.shape(testDataMat)
        for i in range(m):
            kernelEval = mode.kernelTrans(sVS,testDataMat[i,:],kernelList[j])
            predict = kernelEval.T * np.multiply(labelSv,mode.alphas[svInd]) + mode.b
            if(np.sign(predict) != np.sign(testLabel[i])):
                errCount += 1
        print("the test error rate is: %f" % (float(errCount) / m))