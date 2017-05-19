# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:58:55 2017

@author: Chin-Wei
"""

import os
import numpy as np
path = r'./active_learning_BHN'
fnames = os.listdir(path)

import matplotlib.pyplot as plt


legends = []

for fname in fnames:
    if 'all' in fname:
        legends.append(fname[12:-23].replace('__',' '))
        data = np.load(os.path.join(path,fname))
        plt.plot(np.arange(data.shape[0])*10,data.mean(1))
#        np.arange(data.shape[0])*10,

plt.legend(legends,loc=4)
plt.xlabel('Number of acquired images')
plt.ylabel('Accuracy (%)')



