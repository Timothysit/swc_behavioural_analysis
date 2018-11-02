import numpy as np
import pandas as pd
import pickle
import itertools


def find_distance(body_part_1, body_part_2, method='euclidean'):
    """
    Computes the distances between any two body parts, given x, y coordinates
    INPUT
    body_part_1  | [x, y] array of the position of body_parts (can have time dimension)
    method       | method for calculating distance between two body parts
    """
    if method == 'euclidean':
        dist = np.linalg.norm(body_part_1 - body_part_2, axis = 1)
    else:
        print('No valid find_distance method specified')
    return dist

def get_coord(data, body_part_name):
    """
    Extract both the x and y coordinate from data and return it as a joined array
    :param data:
    :param body_part_name:
    :return:
    """
    x_pos = data[body_part_name + '_x']
    y_pos = data[body_part_name + '_y']
    coord = np.array([x_pos, y_pos])
    return coord.T  # TODO: find more efficient way then doing a transpose

def visualise_distance():
    # TODO: visualse distance between all pairs over time
    return None


def main():
    """
	Computes distance over all body parts
	"""
    # load data
    file_name = 'data/body_part_loc_df.pkl'
    with open(file_name, 'rb') as f:
        coord_data = pickle.load(f)

    # TODO: preview dataframe

    body_parts = ['male_left_ear', 'male_right_ear', 'male_nose', 'male_tail',
                 'female_left_ear', 'female_right_ear', 'female_nose', 'female_tail']

    # iterate through each pair of body parts
    body_part_distance = dict()

    for part_1, part_2 in itertools.combinations(body_parts, 2):
        part_1_coord = get_coord(coord_data, part_1)
        part_2_coord = get_coord(coord_data, part_2)
        dist = find_distance(part_1_coord, part_2_coord, method='euclidean')

        body_part_distance[part_1 + '_to_' + part_2] = dist

    # save distances in pd dataframe
    distance_df = pd.DataFrame.from_dict(body_part_distance)

    # preview df
    print(distance_df)

    with open('data/distance_df' + '.pkl', 'wb') as f:
        pickle.dump(distance_df, f)




if __name__ == '__main__':
    main()
