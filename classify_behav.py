from pickle import load as pload
from matplotlib import pyplot as plt
import numpy as np

def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


def discretize_df(data_vec, bin_width):
    """
    Discretizes the given data vector into bins of size bin_width.
    :param data_vec: vector of data samples
    :param bin_width: duration of bins, in number of samples
    :return:
    """

    assert data_vec is np.ndarray, 'data_vec is not a nunpy array'

    nr_bins = round(len(data_vec) / bin_width)
    bins = np.linspace(0, len(data_vec), nr_bins + 1, True).astype(np.int)
    bin_counts = np.diff(bins)

    discrete = np.add.reduceat(data_vec, bins[:-1]) / bin_counts

    return discrete, bins

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition,n=1, axis=0)
    idx, _ = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right. -JK
    # LB this copy to increment is horrible but I get
    # ValueError: output array is read-only without it

    mutable_idx = np.array(idx)
    mutable_idx +=  1
    idx = mutable_idx

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


def part_to_part(angle_diff, nose_distance, angle_threshold = None, dist_threshold = None, duration = None):
    """
    Determines whether there is part_to_part contact in each frame, where part is any body part of the mice
    :param angle_diff: angle difference of the centre of each mice (?), numpy array
    :param nose_distance: nose-to-nose distance difference, numpy array
    :param angle_threshold:
    :param dist_threshold:
    :param duration: minimum duration (in frames)
    :return nose_to_nose_score: score (currently binary) of whether there is nose-to-nose contact
    """

    assert angle_diff is np.ndarray, 'angle_diff is not a numpy array'
    assert nose_distance is np.ndarray, 'nose_distance is not a numpy array'

    part_to_part_score_vec = np.zeros(len(angle_diff))
    contact = np.intersect1d(
        np.where(angle_diff < angle_threshold),
        np.where(nose_distance < dist_threshold)
    )

    part_to_part_score_vec[contact] = 1

    # TODO: work on duration condition
    # TODO: -> durations determined by discretization, so here calculations in # frames

    thresh_bool = .5
    part_to_part_intervals = []
    for start, stop in contiguous_regions(part_to_part_score_vec > thresh_bool):
        if (stop - start > duration):
            part_to_part_intervals.append([start, stop])

    part_to_part_intervals = np.array(part_to_part_intervals)

    return part_to_part_score_vec, part_to_part_intervals



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
