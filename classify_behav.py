from pickle import load as pload
from matplotlib import pyplot as plt

def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


def discretize_df(df, bin_width):
    pass  # todo


def main(param):
    # load distances, angles
    distances = load_pickle(param['f_dist'])
    angles = load_pickle(param['f_angle'])

    # angle difference vector: F-M
    angles_diff = angles['female'] - angles['male']

    # discretize
    angles_diff_disc = discretize_df(angles_diff.as_matrix(), param)
    distances_disc = discretize_df(distances, param)

    plt.plot(angles_diff.as_matrix())
    plt.show()

    # classify by criteria

    # tSNE, clustering

    # save both results -> do visualization from output file
    return distances, angles


if __name__ == '__main__':
    f_dist = './data/distance_df.pkl'
    f_angle = './data/angles_df.pkl'
    f_out = './data/classification.pkl'

    param = {'f_dist': f_dist,
             'f_angle': f_angle,
             'f_out': f_out,
             'bin_width': .2,  # sec
             'sampling_rate': 30,  # Hz
             'thresh_dist': 30,  # px
             'thresh_angle': 45}  # degree rad

    distances, angles = main(param)
