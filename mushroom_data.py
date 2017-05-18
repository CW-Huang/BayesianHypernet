#!/usr/bin/env python
# modified from https://github.com/hlin117/data-science/blob/master/classification/mushroom.py
import pandas as pd
import numpy as np

"""The mushroom data was obtained through the UCI website. The task
is to classify whether a mushroom is edible or not."""

columns = ["edible", "cap-shape", "cap-surface", "cap-color", "bruises?",
        "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
        ]
dataset = pd.read_csv("data/mushroom.data", names=columns, index_col=None)
YX = np.array(pd.get_dummies(dataset[columns]))

Y = YX[:, 0:1] # is_edible
X = YX[:, 2:] # edible gets 2 dummies
