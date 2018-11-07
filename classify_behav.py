from pickle import dump as pdump
from pickle import load as pload

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


def save_pickle(filepath, objects, mode='w'):
    with open(filepath, mode) as f:
        for o in objects:
            pdump(o, f)


def discretize_df(data_vec, bin_width):
    """
    Discretizes the given data vector into bins of size bin_width.
    :param data_vec: vector of data samples
    :param bin_width: duration of bins, in number of samples
    :return:
    """

    # assert data_vec is np.ndarray, 'data_vec is not a nunpy array'

    nr_bins = round(len(data_vec) / bin_width)
    bins = np.linspace(0, len(data_vec), nr_bins + 1, True).astype(np.int)
    bin_counts = np.diff(bins)

    discrete = np.add.reduceat(data_vec, bins[:-1]) / bin_counts

    return discrete, bins


def discretize_vars(angles_diff, distances, param):
    bins = []
    # discretize
    bin_width = round(param['bin_width'] * param['sampling_rate'])
    angles_diff_disc, _ = discretize_df(angles_diff, bin_width)
    distances_col = list(distances.columns.values)
    # for c in distances_col: print(c)
    distances_discrete = pd.DataFrame(columns=distances_col)
    for dcol in distances_col:
        distances_discrete[dcol], bins = discretize_df(distances[dcol], bin_width)
    return angles_diff_disc, distances_discrete, bins


def calculate_velocity(x, y, sampling_rate):
    dist_x = np.ediff1d(x, to_end=np.array([np.nan]))
    dist_y = np.ediff1d(y, to_end=np.array([np.nan]))
    dist = np.sqrt(dist_x + dist_y)
    velocity = dist * sampling_rate
    return velocity


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    Source: https://stackoverflow.com/a/4495197/3751373
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition, n=1, axis=0)
    idx, _ = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right. -JK
    # LB this copy to increment is horrible but I get
    # ValueError: output array is read-only without it

    mutable_idx = np.array(idx)
    mutable_idx += 1
    idx = mutable_idx

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def part_to_part(angle_diff, nose_distance, angle_threshold_tuple=None, dist_threshold=None, duration=None):
    """
    Determines whether there is part_to_part contact in each frame, where part is any body part of the mice
    :param angle_diff: angle difference of the centre of each mice (?), numpy array
    :param nose_distance: nose-to-nose distance difference, numpy array
    :param angle_threshold_tuple: tuple of (lower, upper) boundary for angle difference
    :param dist_threshold:
    :param duration: minimum duration (in frames)
    :return nose_to_nose_score: score (currently binary) of whether there is nose-to-nose contact
    """

    assert angle_diff is np.ndarray, 'angle_diff is not a numpy array'
    assert nose_distance is np.ndarray, 'nose_distance is not a numpy array'

    part_to_part_score_vec = np.zeros(len(angle_diff))
    contact = np.intersect1d(
        np.where(angle_threshold_tuple[0] <= angle_diff <= angle_threshold_tuple[1]),
        np.where(nose_distance < dist_threshold)
    )

    part_to_part_score_vec[contact] = 1

    # TO DO: work on duration condition
    # TO DO: -> durations determined by discretization, so here calculations in # frames

    thresh_bool = .5
    part_to_part_intervals = []
    for start, stop in contiguous_regions(part_to_part_score_vec > thresh_bool):
        if (stop - start > duration):
            part_to_part_intervals.append([start, stop])

    part_to_part_intervals = np.array(part_to_part_intervals)

    return part_to_part_score_vec, part_to_part_intervals


def debug_plot(dataframe, interval=None):
    dval = list(dataframe.columns.values)
    for dcol in dval:
        currDF = dataframe[dcol].as_matrix()

        if not interval:
            interval = [0, currDF.__len__() - 1]

        plt.plot(currDF[interval[0]:interval[1]])

    plt.legend(dval, bbox_to_anchor=(1.1, 1.05))
    plt.show()


def main(param):
    # load distances, angles
    distances = load_pickle(param['f_dist'])
    angles = load_pickle(param['f_angle'])

    # angle difference vector: F-M
    angles_diff = angles['female'] - angles['male']
    angles_diff = angles_diff.as_matrix()

    angles_diff_disc, distances_discrete, bins = discretize_vars(angles_diff, distances, param)
    assert angles_diff_disc.shape[0] == distances_discrete.shape[0], 'bin count mismatch'

    if param['debug']:
        debug_plot(distances_discrete)

    # classify by criteria: for each behavioral gesture
    classif_scores = pd.DataFrame()
    classif_intervals = pd.DataFrame()

    for motif, constraints in param['motifs'].items():
        if not constraints:
            continue
        print(motif, constraints)
        # classify
        classif_scores[motif], classif_intervals[motif] = \
            part_to_part(angles_diff_disc, distances_discrete[constraints['distance_name']],
                         angle_threshold_tuple=constraints['angle_range'],
                         dist_threshold=constraints['distance_thresh'],
                         duration=param['motif_duration_min'])

    save_pickle(param['f_out'], (classif_scores, classif_intervals), 'w')

    # tSNE, clustering

    # save both results -> do visualization from output file

    print('debug')


if __name__ == '__main__':
    f_dist = './data/distance_df.pkl'
    f_angle = './data/angles_df.pkl'
    f_out = './data/classification.pkl'

    distance_thresh = 30  # px : general threshold for contact behaviours
    velocity_thresh = 5  # px/sec : general threshold for moving vs steady behaviours

    # motifs of behavioral poses: [distance name, distance threshold (px), angle diff range threshold]
    motifs = {'nose2body': {'distance_name': 'male_nose_to_female_tail', 'distance_thresh': distance_thresh,
                                'angle_range': ()},  # M 2 F comparisons
              'nose2nose': {'distance_name': 'male_nose_to_female_nose', 'distance_thresh': distance_thresh,
                            'angle_range': (-135, 135), 'velocity_range': (0, velocity_thresh)},
              'nose2genitals': {'distance_name': 'male_nose_to_female_tail', 'distance_thresh': distance_thresh,
                                'angle_range': ()},
              'above': {'distance_name': 'XX', 'distance_thresh': distance_thresh,
                                'angle_range': ()},
              'following': {'distance_name': 'XX', 'distance_thresh': distance_thresh,
                                'angle_range': ()},
              'standTogether': {'distance_name': 'XX', 'distance_thresh': distance_thresh,
                                'angle_range': ()},
              'standAlond': {'distance_name': 'XX', 'distance_thresh': distance_thresh,
                                'angle_range': ()},
              'walkAlone': {'distance_name': 'XX', 'distance_thresh': distance_thresh,
                                'angle_range': ()}}

    param = {'debug': True,
             'f_dist': f_dist,
             'f_angle': f_angle,
             'f_out': f_out,
             'bin_width': .2,  # sec
             'sampling_rate': 30,  # Hz
             'thresh_dist': 30,  # px
             'thresh_angle': 45,
             'motifs': motifs,
             'motif_duration_min': .5 * 30}  # degree rad

    main(param)
