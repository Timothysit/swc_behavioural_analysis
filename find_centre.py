# find centre of mice
import pickle
import numpy as np
import pandas as pd

def find_centre(nose_loc, tail_loc):
    """
    find centre of mice, current method uses the centre point between head and tail
    :param nose_loc: N x 2 array, where N is the number of frames, columns represent x and y coordinates
    :param tail_loc:
    :return:
    """
    centre_loc = (nose_loc + tail_loc) / 2
    centre_x = centre_loc[:, 0]
    centre_y = centre_loc[:, 1]

    return centre_x, centre_y


def main():
    loc_file = 'data/body_part_loc_df.pkl'
    with open(loc_file, 'rb') as f:
        coord_df = pickle.load(f)

    male_nose_loc = np.column_stack((coord_df['male_nose_x'],coord_df['male_nose_y']))
    male_tail_loc = np.column_stack((coord_df['male_tail_x'], coord_df['male_tail_y']))

    female_nose_loc = np.column_stack((coord_df['female_nose_x'],coord_df['female_nose_y']))
    female_tail_loc = np.column_stack((coord_df['female_tail_x'], coord_df['female_tail_y']))

    centre_loc = dict()
    centre_loc['male_centre_x'], centre_loc['male_centre_y'] = find_centre(male_nose_loc, male_tail_loc)
    centre_loc['female_centre_x'], centre_loc['female_centre_y'] = find_centre(female_nose_loc, female_tail_loc)

    # convert to pandas dataframe
    centre_df = pd.DataFrame.from_dict(centre_loc)

    with open('centre_loc_df' + '.pkl', 'wb') as f:
        pickle.dump(centre_df, f)


if __name__ == '__main__':
    main()
