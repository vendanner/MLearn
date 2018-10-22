# !/usr/bin/env python
# -*- coding:utf-8 -*-


import re
import random

from . import Bayesian

def textParse(strFile):
    """
    将文件内容转化成字符串数组
    :param strFile: 文件内容
    :return: 字符串数组
    """
    strList = re.split(r'\W+',strFile)
    if(len(strList) == 0):
        print(strList)
    # word 标准：包含2个英文字母
    return [word.lower() for word in strList if len(word) > 2]

def createData(fileDir):
    """
    读取文件
    :return: 
    """
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1,26):
        try:
            words = textParse(open(fileDir.format(i)).read())
        except:
            words = textParse(open(fileDir.format(i), encoding='Windows 1252').read())
        # list.append 添加新list
        doc_list.append(words)
        # list.extend 在list最后增加其他list值组成新的list 来替代旧的list
        full_text.extend(words)
    return doc_list,full_text

def test():
    """
    垃圾邮件分类器
    :return: 
    """
    doc_list = []
    class_list = []
    full_text = []
    spamDocs,spamWords = createData('input/4.NaiveBayes/email/spam/{}.txt')
    doc_list.extend(spamDocs)
    full_text.extend(spamWords)
    # 垃圾邮件 标记1
    [class_list.append(1) for i in range(0,len(spamDocs))]
    hamDocs, hamWords = createData('input/4.NaiveBayes/email/ham/{}.txt')
    doc_list.extend(hamDocs)
    full_text.extend(hamWords)
    # 垃圾邮件 标记0
    [class_list.append(0) for i in range(0, len(spamDocs))]

    # 选择训练和测试样本
    testSet = [int(i) for i in random.sample(range(50),10)]
    trainSet =list(set(range(50)) - set(testSet))
    # 生成训练集、测试集
    trainDataMatrix = []
    trainLabelMatrix = []
    testDataMatrix = []
    testLabelMatrix = []

    for i in trainSet:
        trainDataMatrix.append(doc_list[i])
        trainLabelMatrix.append(class_list[i])
    for i in testSet:
        testDataMatrix.append(doc_list[i])
        testLabelMatrix.append(class_list[i])

    # 训练模型
    model = Bayesian.naiveBayes()
    model.fit(trainDataMatrix,trainLabelMatrix,list(set(full_text)))

    # 预测模型
    err = 0
    for i in range(len(testDataMatrix)):
        label = model.predict(testDataMatrix[i])
        print("testLabelMatrix = %f,predict label = %f"%(testLabelMatrix[i],label))
        err +=(label != testLabelMatrix[i])
    print("error count = %d"%err)

