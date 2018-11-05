
def calculate_angle (nose_x, nose_y, tail_x, tail_y):
    """
    Function to calculate the angle of the mouse in relation to the space.
    The angle is calculated from the vector from the nose coordinate and tail coordinate

    :param coordinates for nose an tail in each their own numpy array
    :return Angle of animal
    """
n_x = nose_x
n_y = nose_y
t_x = tail_x
t_y = tail_y

nose_tail_coor_df = pd.DataFrame({"nose_x": n_x[:-1],
                                  "direction": direction,
                                  "time": time})


delta_x = n_x - t_x
delta_y = n_y - t_y

angles_df = numpy.arctan(delta_y/delta_x)