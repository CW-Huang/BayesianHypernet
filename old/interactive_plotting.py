#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.ion()

n=0
while n < 100:
    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        #plt.pause(0.15)
    n += 1

    plt.pause(0.25)
