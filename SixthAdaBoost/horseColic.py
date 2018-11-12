#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from . import AdaBoost

def createData(fileName):
    """
    
    :param fileName: 
    :return: 
    """
    xArr = []
    yArr = []
    with open(fileName, 'r') as f:
        numFeat = len(f.readline().split('\t'))

    with open(fileName, 'r') as f:
        for line in f.readlines():
            lineArr = []
            lineList = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(lineList[i]))
            xArr.append(lineArr)
            yArr.append(float(lineList[-1]))
    return xArr,yArr

def plot_roc(predStrengths, classLabels):  #predStrengths 分类器的预测强度
    # 如何理解机器学习和统计中的AUC？ - 刘站奇的回答 - 知乎 https://www.zhihu.com/question/39840928/answer/146205830
    cur = (1.0,1.0) #cursor 绘制光标的位置
    ySum = 0.0 #variable to calculate AUC  计算AUC的值
    numPosClas = sum(np.array(classLabels)==1.0) #通过数组过滤方式计算正例的数目，并将该值赋值给numPosClas
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)  #计算步长
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse 获取排好序的索引
    fig = plt.figure()  #构建画笔，并在所有排序值上进行循环
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    # 先从排名最低的样例开始，对比所有排名更低样例判为负，对比所有排名更高的样例判为正，此情况相当于所有数据都判为正(1,1)，真阳率假阳率都为1
    # 接着继续从次低样例对比，如果样例属于正例，对真阳率修改反之对假阳率修改；
    # 不断对比的过程(分类阈值变化)生成一些列的roc，就组成roc 曲线；for 循环中一次就生成1个roc 值
    # 真阳率 = 真实值为正的数据集中预测也为正的概率(预测为错的时候减)；假阳率 = 真实值为负的数据集中预测为正的概率(预测为正确的时候减)
    # 对照代码解释第二句话，想得比较久；下面for 循环其实就是从次低样例开始了，按第一句话意思，最低样例应该判为负，
    # 但现在最低样例真实值为正，表示真阳率要降低(为真的判断错了)，减步进
    # 但若最低样例真实值为负，表示假阳率要降低(因为负它真的预测为负)，减步进

    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum* xStep)

def test():
    """
    本节用AdaBoost 预测第四节的马问题，会发现在AdaBoost 错误率更低
    :return: 
    """
    trainData,trainLabel = createData("input/7.AdaBoost/horseColicTraining2.txt")
    testData, testLabel = createData("input/7.AdaBoost/horseColicTest2.txt")

    mode = AdaBoost.adaBoost(40)
    agg_class_est = mode.fit(trainData,trainLabel)

    # 预测
    predict_label = mode.predict(testData)
    err = np.mat(np.ones((len(testLabel),1)))
    err_count = err[predict_label != np.mat(testLabel).T].sum()
    print("err count = ",err_count)
    # 10个弱分类器下，adaBoost 错误率23% 少于LR 35%
    print("err rate = ",err_count/len(err))

    """
    在以往的模型判断准则中，我们都是用错误率来衡量模型性能。但在不同情况下，衡量模型的标志是不一样的，
    正确率 = P（预测为正，真实也为正的个数）/（P（预测为正，真实也为正的个数） + P（预测为正，真实为负的个数）），负的预测成正
    召回率 = P（预测为正，真实也为正的个数）/（P（预测为正，真实也为正的个数） + P（预测为负，真实为正的个数）），正的预测成负
    接下来介绍ROC 模型性能评估
    """
    plot_roc(agg_class_est.T, trainLabel)

