#!/usr/bin/env python
# -*- coding:utf-8 -*-


from . import FPGrowth

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def testSimpData():
    """
    
    :return: 
    """
    retDict = {}
    dataSet = loadSimpDat()
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    print(retDict)
    fp = FPGrowth.fpGrowth()
    fp.fit(retDict)
    # fp.Tree.disp()

def test():
    """
    
    :return: 
    """
    testSimpData()