#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
FP-growth 算法简介：
    在上一节Apriori，我们基于Aprior 原理设计了算法查找频繁项；虽然减少一定的计算量(根据原理我们不需要对每一项都计算出现次数)，
但总体对数据集的扫描大约为n-1次(n 为数据集中商品种类)。应用到大数据集上，这仍然是很大的计算量；本节的FP-growth 只对数据集扫描两次即可
构建出全部的频繁项。
    FP-growth ：
    1、构建FP 树：树的分支就是数据集的每项数据(每项数据不一定都是从根到叶，可能是根到某个节点就停止)
        1.1、对单个商品出现的次数扫描
        1.2、基于第一次扫描得到的频繁项，再对数据集扫描；根据频繁项出现的次数和关联(此关联是指频繁项出现在同个订单中)，直接将数据映射到FP树
    2、挖掘频繁项：以单个itme 为例
        2.1、以item 为叶在FP 树抽取出条件基 - 一个条件基就是包含item 购买清单
        2.2、以上面抽取出的条件基为基础，再次构造FP1 树(此时，频繁项集中应加上item)
        2.3、以item1 为叶在FP1 树抽取出条件基
        2.4、以上面抽取出的条件基为基础，再次构造FP2 树(此时，频繁项集中应加上item，item1)
        2.5、继续迭代直到无法构造FP 树
    
    FP-growth 优越性
    1、扫描次数少
    2、相比与Apriori，多次扫描数据集来生成频繁项，FP-growth 是不断构建FP-growth 来筛选出频繁项
"""

class fpGrowth:
    """
    
    """
    def __init__(self):
        """
        
        """
        self.headerTable = {}
        self.Tree = self.treeNode("Null Set",1,None)
        self.freqItems = []

    def fit(self, dataSet, minSup=1):
        self.Tree,self.headerTable = self.createTree(dataSet,minSup)
        self.Tree.disp()
        self.mineTree(self.Tree,self.headerTable,3,set([]),self.freqItems)

    def updateHeader(self,nodeToTest,targetNode):
        """
        相同item 链表的更新;为了后续关联查找使用
        :param nodeToTest: 
        :param targetNode: 
        :return: 
        """
        while (nodeToTest.nodeLink is not None):
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode

    def updateTree(self,items,inTree,headerTable,count):
        """
        更新树,从根节点一直往下更新
        :param items: 
        :param inTree: 
        :param headerTable: 
        :param count: 
        :return: 
        """
        # 从树的根节点开始更新
        if items[0] in inTree.children:
            inTree.children[items[0]].inc(count)
        else:
            # 如果当前节点的子节点没有item[0]，则新建节点
            inTree.children[items[0]] = self.treeNode(items[0],count,inTree)

            # 相同item 之间的关联
            if headerTable[items[0]][1] == None:
                headerTable[items[0]][1] = inTree.children[items[0]]
            else:
                self.updateHeader(headerTable[items[0]][1],inTree.children[items[0]])

        if len(items) > 1:
            # 继续往下更新树，知道items 全部更新完毕
            self.updateTree(items[1:],inTree.children[items[0]],headerTable,count)

    def createTree(self,dataSet,minSup=1):
        """
        构建FP  树
        :param dataSet: 
        :param minSup: 
        :return: 
        """
        headerTable = {}
        # 第一次扫描统计单个item 次数
        for trans in dataSet:
            for item in trans:
                # 统计item 次数
                headerTable[item] = headerTable.get(item,0) + dataSet[trans]
        # keys 会在迭代的时候删除，这里先一次性去除全部key
        for k in list(headerTable.keys()):
            if headerTable[k] < minSup:
                # 删除不满足的item
                del (headerTable[k])
        freqItem = set(headerTable.keys())
        if(len(freqItem) == 0) :
            # 没有频繁项直接返回None
            return  None,None

        # FP 树 头
        for k in headerTable:
            # None 表示为相同item  链表
            headerTable[k] = [headerTable[k],None]
        retTree = self.treeNode("Null Set",1,None)

        # 第二次扫描，映射成FP 树
        for trans,count in dataSet.items():
            localD = {}
            for item in trans:
                if item in freqItem:
                    # 记录item 出现次数
                    localD[item] = headerTable[item][0]

            if(len(localD) > 0):
                # 每项中依据item 出现次数，将item 从高到低排序;FP 树的叶子一定是出现次数item 最少
                orderItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
                self.updateTree(orderItems,retTree,headerTable,count)
        return retTree,headerTable

    """ 查找条件基 """
    def ascendTree(self,leafNode,prefixPath):
        """
        当前节点往上找到根节点
        :param leafNode: 
        :param prefixPath: 
        :return: 
        """
        if leafNode.parent != None:
            prefixPath.append(leafNode.name)
            self.ascendTree(leafNode.parent,prefixPath)

    def findPrefixPath(self,basePat,treeNode):
        """
        找到 basePat item 所有的条件基；
        :param basePat: 
        :param treeNode:  Header 中 basePat 节点，存储着相同item 的FP 树节点位置
        :return: 
        """
        condPats = {}
        while treeNode != None:
            prefixPath = []
            self.ascendTree(treeNode,prefixPath)
            if len(prefixPath) >0:
                # 1 开始去除本身
                condPats[frozenset(prefixPath[1:])] = treeNode.count
            treeNode = treeNode.nodeLink
        return condPats

    def mineTree(self,inTree, headerTable, minSup, preFix, freqItemList):
        """mineTree(创建条件FP树)

        Args:
            inTree       myFPtree
            headerTable  满足minSup {所有的元素+(value, treeNode)}
            minSup       最小支持项集
            preFix       preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
            freqItemList 用来存储频繁子项的列表
        """
        # 通过value进行从小到大的排序， 得到频繁项集的key
        # 最小支持项集的key的list集合
        # 从叶开始往根追溯
        bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
        print('-----', sorted(headerTable.items(), key=lambda p: p[1][0]))
        print('bigL=', bigL)
        # 循环遍历 最频繁项集的key，从小到大的递归寻找对应的频繁项集
        for basePat in bigL:
            # preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
            newFreqSet = preFix.copy()
            # 加上preFix，组成多种商品的频繁项
            newFreqSet.add(basePat)
            print('newFreqSet=', newFreqSet, preFix)

            freqItemList.append(newFreqSet)
            print('freqItemList=', freqItemList)
            condPattBases = self.findPrefixPath(basePat, headerTable[basePat][1])
            print('condPattBases=', basePat, condPattBases)

            # 构建FP-tree
            myCondTree, myHead = self.createTree(condPattBases, minSup)
            print('myHead=', myHead)
            # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
            if myHead is not None:
                myCondTree.disp(1)
                print('\n\n\n')
                # 递归 myHead 找出频繁项集
                self.mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
            print('\n\n\n')

    class treeNode:
        """
        FP 树
        """
        def __init__(self,nameValue,numOccur,parentNode):
            self.name = nameValue
            self.count = numOccur
            self.parent = parentNode
            self.nodeLink = None
            self.children = {}

        def inc(self,numOccur):
            """
            增加次数
            :param numOccur: 
            :return: 
            """
            self.count += numOccur

        def disp(self,ind=1):
            """
            调试使用
            :param ind: 
            :return: 
            """
            for child in self.children.values():
                print('  ' * ind, self.name, ' ', self.count)
                child.disp(ind + 1)