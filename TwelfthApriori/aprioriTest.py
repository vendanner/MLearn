#!/usr/bin/env python
# -*- coding:utf-8 -*-




from . import Apriori

# 加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def test():
    """
    
    :return: 
    """
    # apriori = Apriori.apriori()
    # apriori.fit(loadDataSet())

    # 项目案例
    # 发现毒蘑菇的相似特性
    # 得到全集的数据
    dataSet = [line.split() for line in open("input/11.Apriori/mushroom.dat").readlines()]
    apriori = Apriori.apriori()
    apriori.fit(dataSet, minSupport=0.3)
    L = apriori.L
    supportData = apriori.supportData
    # # 2表示毒蘑菇，1表示可食用的蘑菇
    # # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[2]:
        if item.intersection('2'):
            print(item)
