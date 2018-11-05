from pickle import load as pload


def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


def discretize_df(df, bin_width):
    pass  # todo


def main(param):
    # load distances, angles
    distances = load_pickle(param['f_dist'])
    angles = load_pickle(param['f_angle'])

    # discretize
    distances_disc = discretize_df(distances, param['bin_width'])

    # angle difference vector: F-M

    # classify by criteria

    # tSNE, clustering

    # save both results -> do visualization from output file


if __name__ == '__main__':
    f_dist = './data/distance_df.pkl'
    f_angle = './data/angles_df.pkl'
    f_out = './data/classification.pkl'

    bin_width = .2  # sec

    param = {'f_dist': f_dist,
             'f_angle': f_angle,
             'f_out': f_out,
             'bin_width': bin_width}

    main(param)
