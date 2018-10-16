#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
源码都是copy https://github.com/apachecn/AiLearning/tree/dev/src/py3.x/ml
"""
import FirstKnn
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 设置pd 输出显示
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    pd.set_option('display.width', 1000)

    # 约会
    # FirstKnn.dating.test()
    # 数字识别
    FirstKnn.handWritingClass.test()

    plt.show()