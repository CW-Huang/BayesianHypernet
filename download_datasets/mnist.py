# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:23:14 2017

@author: Chin-Wei
"""

import urllib


url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
fn = r'./data/mnist.pkl.gz'
urllib.urlretrieve(url,fn)



