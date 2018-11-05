# performs clustering on behvoural metrics over time frames

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from ggplot import *
#import swat
#import matplotlib.pyplot as plt
#import seaborn as sns
#from time import time

#5 frames
t_sne_test_df = pd.DataFrame({"frame":np.array([1,2,3,4,5]),
                            "angle": np.array([0.55, 0.60, 1.2, 0.8, 1.4]),
                            "relative_distance":np.array([2,3,7,2,2])})


def cluster(dataframe):
    """
    Function to take in a dataframe and generate a T-SNE
    :return: T-SNE
    """


    tsne_model = TSNE(n_components=2,perplexity = 5)
    tsne_results = tsne_model.fit_transform(dataframe)
    tsne_plot = ggplot(tsne_results)
    print(tsne_results)

cluster(t_sne_test_df)


def viz_cluster():
    pass


