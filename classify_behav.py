from pickle import load as pload

import numpy as np


def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


fp = './data/body_part_loc_df.pkl'

data = load_pickle(fp)


def rotate_coordinate(xy=tuple(), angle=float()):
    if xy.__len__() < 1 or not isinstance(xy[0], float):
        raise ValueError

    x = xy[0] * np.cos(angle) + xy[1] * np.sin(angle)
    y = - xy[0] * np.sin(angle) + xy[1] * np.cos(angle)
    return x, y

def discretize_df(df, bin_duration)
    pass  # todo

def extract_classification_automated():
    pass  # use kmeans or tSNE, then clustering + apply classification to data later

def apply_categories(df, bin_duration):
    # bin data
    bin_duration = .2  # msec
    discretize_df(df, bin_duration)

    # make binary matrix for conditions

    # add label based on bin binary labels

    pass
