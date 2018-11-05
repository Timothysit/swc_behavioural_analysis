import numpy as np
import pandas as pd

def calculate_angle (coord_df, female=TRUE):
    """
    Function to calculate the angle of the mouse in relation to the space.
    The angle is calculated from the vector from the nose coordinate and tail coordinate

    :param coordinates for nose an tail in each their own numpy array
    :return Angle of animal
    """
 length = len(coord_df['female_nose_x'])
 size=np.zeros(length)
 angles_df = pd.DataFrame({"female":
                          , "male":})

### load male data points
### load female data points
### calculate nose-tail-distance for x and y
### write angles column 1 (= male)

if female == TRUE:
    delta_x = coord_df['female_nose_x']-coord_df['female_tail_x']
    delta_y = coord_df['female_nose_y']-coord_df['female_tail_y']

    angle = np.arctan (delta_y/delta_x)

if female == FALSE:


delta_x = n_x - t_x
delta_y = n_y - t_y

angles_df = numpy.arctan(delta_y/delta_x)