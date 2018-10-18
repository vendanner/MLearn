# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
{'tearRate': 
	{'normal': 
		{'astigmatic': 
			{'yes': 
				{'prescript': 
					{'myope': 'hard', 
					'hyper':
						{'age': 
							{'pre': 'no lenses', 'presbyopic': 'no lenses', 'young': 'hard'}
						}
					}
				}, 
			'no': 
				{'age': 
					{'pre': 'soft', 
					'presbyopic': 
						{'prescript': 
							{'myope': 'no lenses', 'hyper': 'soft'}
						}, 
					'young': 'soft'}
				}
			}
		}, 
	'reduced': 'no lenses'}
}
"""
from . import DT
from .decisionTreePlot import createPlot
import matplotlib.pyplot as plt



def createData(fileName):
    """
    读取数据文件返回可用的数组
    :param fileName: 数据文件地址
    :return: matrix 数组
    """
    f = open(fileName,'r')
    # strip() 必须要有，可以去掉换行符
    trainData = [line.strip().split('\t')  for line in f.readlines()]
    return trainData

def test():
    """
    眼镜分类
    """
    trainData = createData("input/3.DecisionTree/lenses.txt")
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    dt = DT.decisionTree()
    dt.fit(trainData,features)
    print(dt.mTree)
    createPlot(dt.mTree)
    # dt.predict()


