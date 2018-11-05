import numpy as np
import pandas as pd
import pickle

#nose_x = [30, 45]
#tail_x = [20, 15]
#nose_y = [25, 50]
#tail_y = [20, 40]

#test_df = pd.DataFrame({"female_nose_x":nose_x,
                       # "female_tail_x":tail_x,
                       # "female_nose_y":nose_y,
                       # 'female_tail_y':tail_y})

def calculate_angle (coord_df, female=True, male=True):
    """
    Function to calculate the angle of the mouse in relation to the space.
    The angle is calculated from the vector from the nose coordinate and tail coordinate

    :param coordinates for nose an tail in each their own numpy array
    :return pd.Dataframe containing the angle of male and female mice
    """
    column_names = ['female', 'male']
    zero_df = np.zeros(shape=(len(coord_df),len(column_names)))
    angles_df = pd.DataFrame(zero_df, columns=column_names)

    #should update the function to use the average between the two ears as opposed to the tail coordinate

    if female == True:
        delta_x = coord_df['female_nose_x']-coord_df['female_tail_x']
        delta_y = coord_df['female_nose_y']-coord_df['female_tail_y']

        angles_df['female'] = np.arctan (delta_y/delta_x)

    if male == True:
        delta_x = coord_df['male_nose_x'] - coord_df['male_tail_x']
        delta_y = coord_df['male_nose_y'] - coord_df['male_tail_y']

        angles_df['male'] = np.arctan(delta_y / delta_x)

    # save as pickle file in data folder
    with open("data/angles_df" + ".pkl", "wb") as f:
        pickle.dump(angles_df, f)

    return angles_df

if __name__ == '__main__':

    coord_df_filename = "data/body_part_loc_df.pkl"
    with open(coord_df_filename, "rb") as f:
        coord_df = pickle.load(f)
    calculate_angle(coord_df, female =True, male=True)


