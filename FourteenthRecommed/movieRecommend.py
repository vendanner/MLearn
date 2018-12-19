#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    本节讨论的是基于用户的电影推荐系统。采集用户观影打分信息，分成训练和测试数据集；在训练训练集中计算每个用户的相似度，
取出测试集用户，根据用户的相似度 * 电影的评分 得到测试用户对当前电影的评分，根据电影评分给测试用户推荐电影。
    此推荐系统是只基于用户，没有考虑冷启动(当前用户是新用户该如何推荐)。
"""

import random
import numpy as np
import math

def createData(pivot=0.7):
    """
    
    :param pivot: 训练、测试数据集比例
    :return: 
    """
    trainSet = {}
    testSet = {}
    with open("input/16.RecommenderSystems/ml-1m/ratings.dat",'r') as f:
        for line in f.readlines():
            user, movie, rating, timestamp = line.split('::')
            intuser = int(user)
            if(intuser > 2000):
                break
            # print("user,%s;movie,%s;rating,%s"%(user, movie, rating))
            if(random.random() < pivot):
                trainSet.setdefault(user,{})
                trainSet[user][movie]=int (rating)
            else:
                testSet.setdefault(user,{})
                testSet[user][movie] =int (rating)
    return trainSet,testSet

def calcSimilarUser(trainSet):
    """
    
    :param trainSet: 
    :return: 用户相似度矩阵
    """
    m = len(trainSet)
    # 用户相似度矩阵
    simMat = np.zeros((m,m))
    userMoive = dict()

    for user,moives in trainSet.items():
        for simUser, simMoives in trainSet.items():
            # 自身不参与相似度计算
            if(user == simUser):
                continue
            for moive,rat in moives.items():
                for simMoive, simrat in simMoives.items():
                    if(moive == simMoive):
                        intuser = int(user) -1
                        intSimuser = int(simUser)-1
                        simMat[intuser][intSimuser] += 1
    # 计算用户相似度
    for i in range(m):
        for j in range(m):
            # 余弦相似度
            simMat[i][j] = simMat[i][j]/math.sqrt(len(trainSet[str(i+1)])*len(trainSet[str(j+1)]))
    # print(simMat)
    return simMat

def recommend(trainSet,testSet,simMat,K=20,N=10):
    """
    
    :param trainSet: 
    :param testSet: 
    :param simMat: 
    :param K: 相似用户前K个
    :param N: 推荐N部电影
    :return: 
    """
    # 推荐成功的电影数；推荐的电影在对应用户的测试集中
    hit = 0
    # 总推荐数
    recCount = 0
    # 测试集的电影数
    testCount = 0
    for user,moives in testSet.items():
        # 已经看过的电影
        watchMoives = trainSet[user]
        # 相似度从高到低排序
        indexMat =  np.argsort(-(simMat[int(user)-1]))[0:K]
        # 推荐的电影
        simMoives = {}
        for i in indexMat:
            # 相似用户过的电影
            for simoive,rat in trainSet[str(i+1)].items():
                if simoive in watchMoives:
                    continue
                simMoives.setdefault(simoive,0)
                # 用户相似度 * 电影评分
                simMoives[simoive] += rat * simMat[int(user)-1][i]
        # print(simMoives)
        # 评分最高的电影
        rank = sorted(simMoives.items(), key=lambda item:item[1], reverse=True)[0:N]
        # print(rank)

        for moive ,rat in rank:
            if moive in moives:
                # 推荐成功
                hit += 1

        testCount += len(moives)
        recCount += N
    precision = hit/(1.0*recCount)
    recall = hit/(1.0*testCount)
    print("hit=%.4f \tprecision=%.4f \t recall=%.4f"%(hit,precision,recall))



def recommendMovies():
    """
    
    :return: 
    """
    # 创建训练、测试数据集
    trainSet,testSet = createData()

    # 训练集计算用户相似度
    simMat = calcSimilarUser(trainSet)
    # 给测试集用户推荐电影
    recommend(trainSet,testSet,simMat)
def test():
    """
    
    :return: 
    """
    print("movie recommend\n")
    recommendMovies()