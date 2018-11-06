# !/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np

class standRegres():
    def __init__(self):
        print("standRegres init")
        self.w = []

    def fit(self,trainData,trainLabel):
        """
        公式求解回归系数
        :param trainData: 
        :param trainLabel: 
        :return: 
        """
        xMat = np.mat(trainData)
        """
        为何这里要转置？假设训练集有m 个数据，trainLabel = m * 1，而实际情况是 1 * m
        X1 * w = y1
        X2 * w = y2
        ...
        Xm * w = ym
        """
        yMat = np.mat(trainLabel).T
        if(np.linalg.det(xMat.T * xMat) == 0):
            # xMat.T * xMat 不可逆，无法进行计算
            print("This matrix is singular, cannot do inverse")
            return
        # 普通线性回归公式
        self.w =(xMat.T * xMat).I * xMat.T * yMat

    def predict(self,X):
        """
        预测
        :param X: 
        :return: y
        """
        return X * self.w

    def rssError(self,y,yHat):
        """
        计算误差
        :param y: 真实值
        :param yHat: 预测值
        :return: 
        """
        return ((y - yHat)**2).sum()

class lwlr():
    """
    局部加权线性回归算法简介：
    局部加权线性回归是在普通线性回归的基础上做了优化。普通线性回归在预测时对于每个训练点的权重相同都为1，而局部加权遵循
    近邻原则：与测试点相邻的训练点权重高，而与测试点较远的训练点权重低甚至可以为0 。那么对于局部加权来说，如何选择权重算法
    对模型准确率很重要。本例用高斯核来计算训练点的权重W。
    普通线性回归公式：w = (XT*X)^(-1) * XT * y
    局部加权线性回归增加权重：w = (XT*WX)^(-1) * XTW * y
    关于普通线性回归公式是利用方程组来求解，详细自行查阅相关资料
    很显然在局部加权算法中很容易过拟合，因为近邻数据权重很大，导致完美的拟合噪声(由此可见机器学习中去除噪声很重要，但实际情况
    要完全去除噪声是伪命题)
    """
    def __init__(self,k):
        """
        本例用高斯核来计算权重，k 相同KNN 算法中K值，相当于超参
        k 取值决定预测时训练集数据参与比例，k 越小参与比例越低
        :param k: 
        """
        print("lwlr init")
        self.k = k
        self.isInverse = False
        self.weights = []

    def predict(self,trainData,trainLabel,X):
        """
        局部加权预测,不必事先进行训练，因为在预测时需要训练数据的参与(计算权重),这是局部加权的缺点
        :param trainData: 
        :param trainLabel: 
        :param X: 
        :return: y
        """
        xMat = np.mat(trainData)
        yMat = np.mat(trainLabel).T
        # 高斯核求权重
        # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
        # shape(a)[0] a数组第一维的长度；训练集数量的方阵 m * m
        self.weights = np.mat(np.eye(((np.shape(trainLabel)[0]))))
        for i in range(len(trainLabel)):
            diffMat = X - xMat[i,:]
            self.weights[i,i] = np.exp((diffMat * diffMat.T) / (-2.0*(self.k ** 2)))
        if (np.linalg.det(xMat.T *(self.weights * xMat)) == 0):
            # xMat.T * xMat 不可逆，无法进行计算
            print("This matrix is singular, cannot do inverse")
            return
        else:
            self.isInverse = True
        return (float)(X * ((xMat.T *(self.weights * xMat)).I * xMat.T * self.weights * yMat))

    def rssError(self,y,yHat):
        """
        计算误差
        :param y: 真实值
        :param yHat: 预测值
        :return: 
        """
        return ((y - yHat)**2).sum()


class ridgeRegres():
    """
    岭回归算法：w = (XT*X + λI)^(-1) * XT * y,I 是特征的单位矩阵，λ 是超参，类似与局部加权的k
    当训练集数据小于特征时，xMat.T * xMat无法进行逆运算，上面的2个算法都无法计算。岭回归是在(xMat.T * xMat) 加上
惩罚项后使其可逆。
    岭回归和后面的Lasso 都是缩减法，在普通线性回归算法中加入惩罚项，减小不重要特征的权重。不同之处在岭回归是L2，
Lasso 是L1 。 
    注意：在使用岭回归算法时，数据需要做标准差标准化（ (特征 - 均值)/方差 ==> 聚集在0附近，方差为1）
    """
    def __init__(self,lam = 0.2):
        print("ridgeRegres init")
        self.lam = lam
        self.w = []

    def regularize(self,mat):
        """
        fit 前数据标准化
        :param mat: 
        :return: 归一化的数据
        """
        means = np.mean(mat,0)
        if(np.shape(mat)[1] < 2):
           #  除方差
           mat = (mat - means) / np.var(mat,0)
        else:
            mat = mat - means
        return mat

    def fit(self,trainData,trainLabel):
        """
        不同与局部加权，预测时不需要训练集的参与
        :param trainData: 
        :param trainLabel: 
        :return: 
        """
        xMat = np.mat(trainData)
        yMat = np.mat(trainLabel).T
        # 岭回归惩罚项，特征的单位矩阵
        ridge = np.eye(np.shape(xMat)[1]) * self.lam
        if(np.linalg.det(xMat.T * xMat + ridge) == 0):
            print("This matrix is singular, cannot do inverse")
            return
        # 岭回归公式
        self.w =(xMat.T * xMat + ridge).I * xMat.T * yMat

    def predict(self,X):
        """
        预测
        :param X: 
        :return: y
        """
        return X * self.w


class stageWise():
    """
    
    """
    def __init__(self,eps = 0.01):
        """
        
        :param eps: 步长
        """
        print("stageWise init")
        self.eps = eps
        self.w = []

    def regularize(self,mat):
        """
        fit 前数据标准化
        :param mat: 
        :return: 归一化的数据
        """
        means = np.mean(mat,0)
        mat = (mat - means) / np.var(mat, 0)
        return np.mat(mat).A

    def fit(self,trainData,trainLabel,cnt):
        """
        
        :param trainData: 
        :param trainLabel: 
        :return: 
        """
        print(type(trainLabel))
        n = np.shape(trainData)[1]
        returnMat = np.zeros((cnt,n))
        self.w = np.zeros((n,1))
        wMax = self.w.copy()
        lowestError = np.inf
        # 特征逐个调整,一次循环调整一个特征；看看是否与LR 的算法很相似
        for i in range(cnt):
            print(self.w.T)

            for j in range(n):
                # 特征增大 or 减少，选方差小的特征值
                for sign in [-1,1]:
                    wTest = self.w.copy()
                    wTest[j] += sign * self.eps
                    error = self.rssError((trainData *  np.mat(wTest)).A,trainLabel.A)
                    if(error < lowestError ):
                        lowestError = error
                        wMax = wTest
            self.w = wMax.copy()
            returnMat[i,:] = self.w.T
        return returnMat

    def predict(self,X):
        """
        预测
        :param X: 
        :return: y
        """
        return np.mat(X) * self.w

    def rssError(self, y, yHat):
        """
        计算误差
        :param y: 真实值
        :param yHat: 预测值
        :return: 
        """
        return ((y - yHat) ** 2).sum()