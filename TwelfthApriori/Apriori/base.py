#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Apriori 简介：
    Apriori 是在大数据集中寻找关系的任务，关联分析的一种；有两种表现形式：频繁项 - 经常出现一块物品集合；关联规则 - 两种物品之间存在很强的关系。
频繁项由支持度和可信度来定义：支持度-该项集占整个数据集的比例；可信度- 针对某条关联规则定义，某天关联规则支持度/商品原本的支持度。
Apriori 就是能在庞大的数据中高效得找到关联的一种算法。
    Apriori 原理：某个项集是频繁，则它的所有子集也是频繁；相反某个项集非频繁，则它的所有超集也是非频繁 - 这可以减少很多不必要的计算。

1、找到频繁项
2、找出频繁项内部之间的关联规则
    
"""
class apriori:
    """
    
    """
    def __init__(self):
        """
        
        """
        self.L = []
        self.supportData = {}

    def fit(self,dataSet,minSupport = 0.5,minConf=0.7):
        """
        
        :param dataSet: 
        :param minSupport: 
        :return: 
        """
        self.apriori(dataSet,minSupport)
        self.generateRules(self.L,self.supportData,minConf)


    def createC1(self,dataSet):
        """
        构建出在dataSet 中单个item 的数据集
        :param dataSet: 数据集
        :return: 单个item 的数据集
        """
        C1 = []
        for transaction in dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        # frozenset 不可变
        return map(frozenset,C1)

    def scanD(self,D,CK,minSupport):
        """
        计算数据集CK 在数据集D中，满足minSupport 的数据集
        :param D: 数据集
        :param CK: 要计算在D中频繁项的数据集(候选项)
        :param minSupport: 最小支持度
        :return: 满足最小支持度的数据集,CK 数据集中所有频繁项
        """
        ssCnt = {}
        # python3 遍历一遍map 后,map 变为空;这里先转为list
        CKList = list(CK)
        DList = list(D)
        for tid in DList:
            for can in CKList:
                if can.issubset(tid):
                    if can not in ssCnt:
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1
        numItems = float(len(DList))
        retList = []
        supportData = {}
        # 注意这里迭代的是key，values = ssCnt.valus,同时迭代ssCnt.items()
        for key in ssCnt:
            support = ssCnt[key]/numItems
            if support >= minSupport:
                retList.insert(0,key)
            # 存储所有项的频繁
            supportData[key] = support
        return retList,supportData

    def aprioriGen(self,Lk,k):
        """
        在数据集Lk中，选取数据前k 个相同的数据合并
        :param Lk: 
        :param k: 
        :return: 
        """
        retList = []
        lenLk = len(Lk)
        for i in range (lenLk):
            for j in range(i+1,lenLk):
                # 这里为何是k-2？
                # 起始这里没必要搞k-2，只是为了和调用函数的apriori 参数k 匹配而已
                # 直接理解，第一次比较0个数字，第二次比较1个数字，依次类推
                L1 = list(Lk[i])[:k-2]
                L2 = list(Lk[j])[:k-2]
                L1.sort()
                L2.sort()
                if(L1 == L2):
                    retList.append(Lk[i]|Lk[j])
        return retList

    def apriori(self,dataSet,minSupport = 0.5):
        """
        找出数据集中 所有 支持度 >minSupport 的频繁项
        :param dataSet: 
        :param minSupport: 
        :return: L = 频繁项；supportData = 数据集所有项的支持度
        """
        C1 = self.createC1(dataSet)
        # 不考虑商品出现的次数
        D = dataSet
        L1,supportData = self.scanD(D,C1,minSupport)
        L = [L1]
        k = 2
        # 不断筛选出包含更多物品的频繁项
        while len(L[k-2]) > 0:
            # 找出满足合并要求的数据项
            Ck = self.aprioriGen(L[k-2],k)
            Lk,supK = self.scanD(D,Ck,minSupport)
            # 在 supportData 插入supK
            supportData.update(supK)
            L.append(Lk)

            k += 1

        self.L = L
        self.supportData = supportData
        return L,supportData

    """
    关联规则函数
    """
    def calcConf(self,freqSet,H,supportData,br1,minConf=0.7):
        """
        计算 (freqSet - conseq) -> freqSet 可信度>minConf 的关联规则
        :param freqSet: 
        :param H: 
        :param supportData: 
        :param br1: 
        :param minConf: 
        :return: 满足可信度的后件
        """
        prunedH = []
        for conseq in H:
            conf = supportData[freqSet]/supportData[freqSet - conseq]
            if conf >= minConf :
                print(freqSet - conseq, '-->', conseq, 'conf:', conf)
                br1.append((freqSet - conseq,conseq,conf))
                prunedH.append(conseq)
        return prunedH

    def rulesFromConseq(self,freqSet,H,supportData,br1,minConf=0.7):
        """
        找出freqSet 中可信度 > minConf 的所有规则
        :param freqSet: 
        :param H: 
        :param supportData: 
        :param br1: 
        :param minConf: 
        :return: 
        """
        m = len(H[0])
        if len(freqSet) > (m + 1):
            Hmp1 = self.aprioriGen(H,m+1)
            Hmp1 = self.calcConf(freqSet,Hmp1,supportData,br1,minConf)
            # 必须大于1 不然后续H 没办法合并
            if(len(Hmp1) > 1):
                self.rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

    def generateRules(self,L,supportData,minConf=0.7):
        """
        找出频繁项L 内所有可信度 > 0.7 关联规则
        :param L: 
        :param supportData: 
        :param minConf: 
        :return: 
        """
        bigRuleList = []
        # 只有一件商品的频繁项不参与关联计算
        for i in range(1,len(L)):
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                    self.rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
                #  只有2件商品的情况，偏离即可
                else:
                    self.calcConf(freqSet,H1,supportData,bigRuleList,minConf)

        return bigRuleList