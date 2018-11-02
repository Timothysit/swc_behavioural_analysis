from scipy.signal import lfilter
import pandas as pd
import numpy as np
import pickle

def main():
    md_interaction, scorer = load_data()
    body_part_loc = pre_process(md_interaction, scorer, likelihoodThreshold=0.98, outputType='dataframe')

    print(body_part_loc)

    # save data
    with open('body_part_loc_df' + '.pkl', 'wb') as f:
        pickle.dump(body_part_loc, f)

def load_data():
    # load data and do the pre-processing
    mf_interaction = pd.read_hdf('data/18_10_29_mf_interaction_leftDeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000.h5')
    md_interaction = mf_interaction.T
    #Copy and paste the name of the scorer from the dataframe above 
    # (also find out how to get the infor directly from the dataframe..)
    scorer = 'DeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000'
    return md_interaction, scorer

def pre_process(data, scorer, likelihoodThreshold = 0.98, outputType = 'dataframe'):
    """
    :param data: camera frame data to analyse
    :param scorer:
    :param likelihoodThreshold: the threshold for accepting a position estimate, otherwise, convert it to 0
    :param outputType: output format, either dataframe or dictionary
    :return:
    """
    bodyParts = ['male_left_ear', 'male_right_ear', 'male_nose', 'male_tail',
                 'female_left_ear', 'female_right_ear', 'female_nose', 'female_tail']

    part_loc = dict()

    for part in bodyParts:
        # run the clean up and interpolation procedure
        # also need to save it into a sensible data frame

        # this will set all values where DLC gave a predicted location at less than a specified confidence interval to 0
        body_part_0s_x, body_part_0s_y = get_x_y_data_cleanup(data, scorer, part, likelihoodThreshold)

        # TODO: check if data loaded successfully
        assert body_part_0s_x is not None, 'get_x_y_data_cleanup not loading anything'

        # this will interpolate linearly over all co-ordinates set to 0 in the previous function '0scleanup'
        start_value_cleanup(body_part_0s_x)
        start_value_cleanup(body_part_0s_y)
        body_part_interpolated_x = interp_0_coords(body_part_0s_x)
        body_part_interpolated_y = interp_0_coords(body_part_0s_y)

        assert body_part_interpolated_x is not None, 'interp_0_coords returning None'

        # filtering
        body_part_interpolated_lfilt_x = smooth_data(body_part_interpolated_x, n = 20, method = 'lfilt')
        body_part_interpolated_lfilt_y = smooth_data(body_part_interpolated_y, n = 20, method='lfilt')

        assert body_part_interpolated_lfilt_x is not None, 'smooth_data returning None'

        part_loc[part + '_x'] =  body_part_interpolated_lfilt_x
        part_loc[part + '_y'] = body_part_interpolated_lfilt_y

    # check whether to output dictionary or dataframe
    if outputType == 'dict':
        pass
    elif outputType == 'dataframe':
        part_loc = pd.DataFrame.from_dict(part_loc)
    else:
        print('No valid datatype specified')
    return part_loc

def smooth_data(data, n = 20, method = 'lfilt'):
    """
    Performs smoothing on data (numpy vector time series?)
    :param data:
    :param n:
    :param method:
    :return:
    """
    if method == 'lfilt':
        nom = [1.0 / n] * n
        denom = 1
        return lfilter(nom,denom,data)
    else:
        print('No valid smoothing method specified')

########## some of Daniel's function for pre-processing data ########################################

def get_x_y_data(data, scorer, bodypart):
    # get x_y_data
    print('bodypart is: ', bodypart)
    bodypart_data = (data.loc[(scorer, bodypart)])

    bodypart_data_x = bodypart_data.loc[('x')]
    bodypart_data_y = bodypart_data.loc[('y')]

    return bodypart_data_x, bodypart_data_y


def get_x_y_data_cleanup(data, scorer, bodypart, likelihood):
    # sets any value below a particular point to value 0 in x and y, this 0 value can then be used by a later
    # interpolation algorithm

    bodypart_data = (data.loc[(scorer, bodypart)])

    x_coords = []
    y_coords = []

    for index in bodypart_data:
        if bodypart_data.loc['likelihood'][index] > likelihood:
            x_coords.append(bodypart_data.loc['x'][index])
            y_coords.append(bodypart_data.loc['y'][index])
        else:
            x_coords.append(0)
            y_coords.append(0)

    return x_coords, y_coords

def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        if value > 0:
            start_value = value
            start_index = index
            break

    for x in range(start_index):
        coords[x] = start_value


def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
                        print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
    print('function exiting')
    return(coords_list)

if __name__ == '__main__':
    main()