# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
算法简介：
    KNN 是k-NearestNeighbor简称，主要用于分类：在离测试点最近的K个训练点里，那个分类结果占多数该测试点就属于哪类。
- 最近：以距离来衡量，而距离为2个向量的距离，根据特征的维度可以用欧几里得(L-2)、曼哈顿距离(L-1)、闵可夫斯基距离(多维)
        在计算向量距离时根据实际情况对特征值做特殊处理：1、特征权重都相同时，将所有特征值都压缩在[0,1]之间以免某个特征值
    过大干扰距离生产影响算法准确率；2、当特征权重有较大差异时，在计算特征值时需要将该 特征值 * 权重
- K：最近的训练点个数，K的选择很讲究它会影响准确率，相当于超参；K选择很小很小容易过拟合而K选择趋于无穷大则无意义
"""


import numpy as np
from collections import Counter

def autoNorm(sourceMatrix):
    """
    特征归一化处理
    :param sourceMatrix: 待处理矩阵
    :return: targetMatrix：归一化后的矩阵
    """
    minVal = sourceMatrix.min(0)
    maxVal = sourceMatrix.max(0)
    rangs = maxVal - minVal
    targetMatrix = (sourceMatrix - minVal)/rangs
    return targetMatrix


def knnModel(point,matrix,labels,k,isXOR = False):
    """
    KNN模型：
        传入point ，训练数据，k；输出分类
    :param point: 待分类的点
    :param matrix: 特征值矩阵
    :param labels: 与特征值想对应的标签矩阵
    :param k: 近邻个数
    :param isXOR: 特征值只有0 or 1 ，用异或来计算距离可减少时间提高效率
    :return: label:类别
    """
    # 1、欧式定理计算
    # axis＝0表示按列相加，axis＝1表示按照行的方向相加表示point到每个训练点的距离
    if(isXOR):
        dist = np.sum(point != matrix, axis=1)
    else:
        dist = np.sum((point - matrix)**2,axis=1)**0.5

    # 2、dist 排序并取出前k的分类结果
    kLabels = [labels[index] for index in dist.argsort()[0:k]]

    # 3、计算前k的分类结果中最多的分类
    # kLabels = [3, 3, 3, 3, 1]
    # Counter({3: 4, 1: 1});4个3，1个1
    count = Counter(kLabels)
    # most_common 出现频率最高的n 个字符
    label = count.most_common(1)[0][0]
    return label

if __name__ == "__main__":
    print("")
