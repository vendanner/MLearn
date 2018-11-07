# !/usr/bin/env python
# -*- coding:utf-8 -*-

from bs4 import BeautifulSoup
import numpy as np
import random

from . import  Regression

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    print(inFile)
    # 打开并读取HTML文件
    with open(inFile,'r',encoding='utf-8') as fr:
        soup = BeautifulSoup(fr.read(),features="lxml")
        i=1
        # 根据HTML页面结构进行解析
        currentRow = soup.findAll('table', r="%d" % i)
        while(len(currentRow)!=0):
            currentRow = soup.findAll('table', r="%d" % i)
            title = currentRow[0].findAll('a')[1].text
            lwrTitle = title.lower()
            # 查找是否有全新标签
            if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
                newFlag = 1.0
            else:
                newFlag = 0.0
            # 查找是否已经标志出售，我们只收集已出售的数据
            soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
            if len(soldUnicde)==0:
                print ("item #%d did not sell" % i)
            else:
                # 解析页面获取当前价格
                soldPrice = currentRow[0].findAll('td')[4]
                priceStr = soldPrice.text
                priceStr = priceStr.replace('$','') #strips out $
                priceStr = priceStr.replace(',','') #strips out ,
                if len(soldPrice)>1:
                    priceStr = priceStr.replace('Free shipping', '')
                sellingPrice = float(priceStr)
                # 去掉不完整的套装价格
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
            i += 1
            currentRow = soup.findAll('table', r="%d" % i)

def createData():
    """
    获取乐高数据,Google 被墙直接读取本地文件
    :return: 
    """
    retX = []
    retY = []
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10196.html', 2009, 3263, 249.99)
    return retX,retY

def crossValidation(xArr,yArr,num = 10):
    """
    交叉验证岭回归
    :param trainData: 
    :param trainLabel: 
    :return: 
    """
    m = len(yArr)
    indexList = np.array(range(m))
    # 10 交叉验证 - 30 次不同lambda 岭回归误差记录
    errorMat = np.zeros((num,30))
    for i in range(num):
        trainX = [];trainY =[]
        testX = [];testY = []
        # 打乱数据集，主要年份 ==> 训练、测试数据集的分布
        random.shuffle(indexList)
        for j in range(m):
            # 训练:测试 = 9:1
            if(j < m*0.9):
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 30 个不同lambda 的岭回归
        wMat = np.zeros((30,np.shape(xArr)[1]))
        for z in range(30):
            ridge =  Regression.ridgeRegres(np.exp(z - 10))
            ridge.fit(ridge.regularize(np.array(trainX)),(np.mat(trainY).T - np.mean(np.mat(trainY).T,0)))
            wMat[z,:] = ridge.w.T

            # 计算测试集每次岭回归的误差
            ridgeModeLabel = []
            for s in range(len(testY)):
                ridgeModeLabel.append(ridge.predict(testX[s]))

            errorMat[i,z] = ridge.rssError(np.array(ridgeModeLabel),np.array(testY))
    # 找到 10 交叉验证 - 30 次不同lambda 岭回归 中误差值最小
    # 求30 次不同lambda 岭回归(被10 次交叉验证) 的均值
    meanErrors = np.mean(errorMat,0)
    # 求30 次不同lambda 岭回归的最小值
    minError = float(np.min(meanErrors))
    # np.nonzero(meanErrors == minError) 找到最小值的 index
    bestw = wMat[np.nonzero(meanErrors == minError)]
    print(bestw)
    # 数据还原求方差与standRed 比较
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestw/varX
    # 输出构建的模型
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat,0))




def test():
    """
    乐高玩具预测
    :return: 
    """
    trainData,trainLabel = createData()
    m,n = np.shape(trainData)
    trainStandData = np.ones((m,n+1))
    trainStandData[:,1:5] = trainData
    print("**********************************************")
    # 线性回归
    stand = Regression.standRegres()
    stand.fit(trainStandData,trainLabel)
    print("standRegres w = ",stand.w)

    standModeLabel = []
    for i in range(len(trainLabel)):
        standModeLabel.append(stand.predict(trainStandData[i]))
    print("standRegres rssError = ",stand.rssError(np.array(standModeLabel),np.array(trainLabel)))

    # 交叉验证
    # 不要加常量w0,否则无法进行数据标准化
    crossValidation(trainData,trainLabel,10)

