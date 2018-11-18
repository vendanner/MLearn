# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CART 回归算法简介：
    在第二节我们介绍了树回归，我们是利用ID3生成树；但这种方式只适合离散数据(连续数据也只能转化为离散)。
本节CART 算法构建决策树是基于二元切分发是可以处理连续数据(大于等于 or 小于)
    ID3中划分的标准是信息增益，而在CART中是Gini系数：1 - ( P(当前类别) 平方和)；假设数据集有C1和C2，C1个数1，C2个数为5，
Gini 系数 = 1 - P(C1)**2 -P(C2)**2 = 1- (1/6)**2 - (5/6)**2 = 0.278；显然 Gini index 越小越好(与信息熵类似);以上是离散数据集的划分标准。
本节讨论的是连续数据，划分标准为总方差 = 各个数据方差之和
"""

import numpy as np

class tree():
    """
    
    """


    """
    回归树生成相关函数
    """
    def regLeaf(self,dataSet):
        """
        创建叶节点
        :param dataSet: 
        :return: 均值
        """
        return np.mean(dataSet[:,-1])

    def regErr(self,dataSet):
        """
        总方差计算函数
        :param dataSet: 
        :return: 总方差
        """
        return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

    def regTreeEval(self,mode,inData):
        return float(mode)

    """
    模型数的叶节点生成，区别于上面回归数的regLeaf、regErr 均值，方差计算
    """
    def linerSolve(self,dataSet):
        """
        求线性模型的w
        :param dataSet: 数据集
        :return: 
        """
        m,n = np.shape(dataSet)
        X = np.mat(np.ones((m,n)))
        y = np.mat(np.ones((m,1)))
        # 1:n ? 线性模型中常数的w0 = 1
        X[:,1:n] = dataSet[:,0:n-1]
        y = dataSet[:,-1]
        xTx = X.T* X
        if(np.linalg.det(xTx) == 0):
            # 不可逆;抛异常
            raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
        ws = xTx.I *(X.T *y)
        return ws,X,y

    def linerLeaf(self,dataSet):
        """
        叶节点 = 线性模型的w
        :param dataSet: 
        :return: 
        """
        ws,X,y = self.linerSolve(dataSet)
        return ws
    def linerErr(self,dataSet):
        """
        注意这里的误差计算
        在regErr 中，是求数据的总房差
        linerErr，求模型误差
        :param dataSet: 
        :return: 
        """
        ws, X, y = self.linerSolve(dataSet)
        yHat = X * ws
        return np.sum(np.power((y - yHat),2))

    def modeTreeEval(self,mode,inData):
        """
        模型数预测，yHat = X.T * w
        :param mode: 
        :param inData: 
        :return: 
        """
        n = np.shape(inData)[1]
        X = np.mat(np.ones((1,n+1)))
        X[:,1:n+1] = inData
        return float(X * mode)

    def splitDataSet(self,dataSet,indexFea,valFea):
        """
        数据集划分为2部分
        :param dataSet: 
        :param indexFea: 
        :param valFea: 
        :return: 
        """
        # np.nonzero 返回数组中非0值的 维度，[0] = 第几个数据
        mat0 = dataSet[np.nonzero(dataSet[:,indexFea] <= valFea)[0],:]
        mat1 = dataSet[np.nonzero(dataSet[:, indexFea] > valFea)[0], :]
        return mat0,mat1


    def chooseBestSplit(self, dataSet, leafType=regLeaf, errType=regErr,ops=(1,4)):
        """ 
        选择最好的分割特征以及特征值；决策树生成完毕则返回None
        对每个特征：
            对每个特征值：(容易过拟合)
                将数据分层2份
                计算切分误差
                找到最小误差的切分值
        返回最佳切分的特征和特征值
        :param dataSet: 
        :param leafType: 计算均值
        :param errType: 计算总方差
        :param ops: 
        :return: indexFea，valFea
        """
        # 容许误差
        tolS = ops[0]
        # 切分的最少样本数(预剪枝),当小于tolN 表示不再继续切分直接返回；防止切割的太细
        tolN = ops[1]
        # 全都都是同个类别
        if(len(set(dataSet[:,-1].T.tolist()[0])) == 1):
            return None,leafType(dataSet)

        m,n = np.shape(dataSet)
        S = errType(dataSet)
        bestS = np.inf;bestIndex= 0;bestVal = 0
        for i in range(n-1):
            for splitVal in set(dataSet[:, i].T.tolist()[0]):
                mat0,mat1 = self.splitDataSet(dataSet,i,splitVal)
                # 不在切分
                if((np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN)):
                    continue
                # 计算切分后的误差
                newS = errType(mat0) + errType(mat1)
                if(newS < bestS):
                    bestS = newS
                    bestIndex = i
                    bestVal = splitVal
        # 假设找到最好划分线后生成的误差与原本数据集的误差之间差值 < 容许误差，则表示分割无效直接返回
        if((S - bestS) < tolS):
            return None,leafType(dataSet)
        mat0, mat1 = self.splitDataSet(dataSet, bestIndex, bestVal)
        # 如果切分的某个数据集大小小于用户定义也直接退出
        if ((np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN)):
            return None, leafType(dataSet)
        return bestIndex,bestVal


    """
    下面的代码是后剪枝
    基于已有的数据切分测试数据集
        如果存在任一子集是一棵树(分得太细，可提前设置参数参考tolN)，则在该子集递归剪枝过程
        计算当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并(为何会出现合并后误差还变小？现在是测试数据)
    从上到下遍历，最开始参与误差计算的是最下层的叶
    """
    def isTree(self,obj):
        return (type(obj).__name__ == 'dict')

    def getMean(self,tree):
        """
        计算树的平均值
        从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
        对 tree 进行塌陷处理，即返回树平均值。
        :param tree: 
        :return: 
        """
        if(self.isTree(tree['right'])):
            tree['right'] = self.getMean(tree['right'])
        if(self.isTree(tree['left'])):
            tree['left'] = self.getMean(tree['left'])
        # 递归求当前树的均值(regLeaf 函数定义的)
        return (tree['right'] + tree['left'])/2.0

    def prune(self,tree,testData):
        """
        满足条件的树进行剪枝
        :param tree: 待剪枝的数
        :param testData: 测试数据
        :return: 
        """
        if(np.shape(testData)[0] == 0):
            # 如果测试集为空，直接返回tree 均值
            return self.getMean(tree)
        if((self.isTree(tree['right'])) or self.isTree(tree['left'])):
            lSet,rSet = self.splitDataSet(testData,tree['spInd'],tree['spVal'])

        # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if self.isTree(tree['left']):
            tree['left'] = self.prune(tree['left'], lSet)
        # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if self.isTree(tree['right']):
            tree['right'] = self.prune(tree['right'], rSet)
        # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

        if( (not self.isTree(tree['right'])) and (not self.isTree(tree['left']))):
            lSet, rSet = self.splitDataSet(testData, tree['spInd'], tree['spVal'])
            # 叶点没合并时，划分后的测试数据误差
            errorNoMerge = np.sum(np.power((lSet[:,-1] - tree['left']),2) + np.power((rSet[:,-1] - tree['right']),2))
            # 叶点合并后，当前测试数据集的误差
            errorMerge = np.sum(np.power((testData[:,-1] - self.getMean(tree)),2))
            # 假设合并后误差小则进行合并
            if(errorMerge < errorNoMerge):
                print("merge")
                return self.getMean(tree)
            else:
                return tree
        else:
            # 不会发生
            return tree


    def createTree(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        """
        创建树，如果是回归树，叶节点是常数；如果是模型数，叶节点是线性模型
        :param dataSet: 
        :param leafType: 
        :param errType: 
        :param ops: 
        :return: 
        """
        feat,val = self.chooseBestSplit(dataSet,leafType,errType,ops)
        if feat is None:
            return val

        retTree = {}
        # 保存当前切割参数
        retTree['spInd'] = feat
        retTree['spVal'] = val
        lSet,rSet = self.splitDataSet(dataSet,feat,val)
        # 递归继续切割直到树创建完毕
        retTree['left'] = self.createTree(lSet,leafType,errType,ops)
        retTree['right'] = self.createTree(rSet, leafType, errType, ops)
        return retTree

    def treeForeCast(self, tree, inData, modeEval=regTreeEval):
        """

        :param tree: 模型数
        :param inData: 测试数据
        :param modeEval: 分类模型
        :return: 
        """
        if not self.isTree(tree):
            return modeEval(tree, inData)

        if (inData[tree['spInd']] <= tree['spVal']):
            if (self.isTree(tree['left'])):
                return self.treeForeCast(tree['left'], inData, modeEval)
            else:
                return modeEval(tree['left'], inData)
        else:
            if (self.isTree(tree['right'])):
                return self.treeForeCast(tree['right'], inData, modeEval)
            else:
                return modeEval(tree['right'], inData)


    def __init__(self,isMode=False,ops=(1, 4)):
        """"
        """
        self.ops = ops
        self.isMode = isMode
        self.tree = {}

    def fit(self,dataSet):
        """
        
        :param dataSet: 包含特征值和标签
        :return: 
        """
        if(self.isMode):
            self.tree = self.createTree(dataSet,self.linerLeaf,self.linerErr,self.ops)
        else:
            self.tree = self.createTree(dataSet, self.regLeaf, self.regErr, self.ops)

    def pruneTree(self, tree, testData):
        """
        后剪枝
        :param tree: 
        :param testData: 
        :return: 
        """
        self.tree = self.prune(self.tree,testData)

    def predict(self,inData):
        """
        
        :param inData: 单个测试数据
        :return: 预测值
        """
        if(self.isMode):
            return self.treeForeCast(self.tree,inData,self.modeTreeEval)
        else:
            return self.treeForeCast(self.tree,inData,self.regTreeEval)