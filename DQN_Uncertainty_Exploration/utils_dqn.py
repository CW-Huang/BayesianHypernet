import os
import numpy as np


def get_last_folder_id(folder_path):
    t = 0
    for fn in os.listdir('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments'):
        t = max(t, int(fn))
    return t


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma