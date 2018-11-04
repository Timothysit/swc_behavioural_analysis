import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import matplotlib.animation as animation

def visualise_distance(distance_df):
    """
    :param distance_df: dataframe containing distance between each pair of body parts
    :return:
    """

    male_distances = ['male_left_ear_to_male_right_ear', 'male_nose_to_male_tail']
    female_distances = ['female_left_ear_to_female_right_ear', 'female_nose_to_female_tail']
    male_female_distances = ['male_nose_to_female_tail', 'male_nose_to_female_nose']

    plt.figure()
    # within male distances
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(distance_df['male_left_ear_to_male_right_ear'], color = 'cornflowerblue')
    ax1.plot(distance_df['male_nose_to_male_tail'], color = 'royalblue')
    sns.despine(top = True, right = True)
    plt.xlabel('Camera frame number')
    plt.ylabel('Distance (pixels)')
    plt.legend(['Left ear - Right ear', 'Nose - Tail'], frameon = False)
    ax1.set_title('Male')

    # within female distances
    ax2 = plt.subplot(1, 2, 2, sharey = ax1)
    ax2.plot(distance_df['female_left_ear_to_female_right_ear'], color = 'salmon')
    ax2.plot(distance_df['female_nose_to_female_tail'], color = 'tomato')
    sns.despine(top=True, right=True)
    plt.xlabel('Camera frame number')
    # plt.ylabel('Distance (pixels)')
    plt.legend(['Left ear - Right ear', 'Nose - Tail'], frameon = False)
    ax2.set_title('Female')

    plt.show()

    # male_female distance
    plt.figure()
    ax3 = plt.subplot(1, 1, 1)
    vert_offset = 700 # the two lines overlap too much
    ax3.plot(distance_df[male_female_distances[1]] + vert_offset, color = 'mediumpurple')
    ax3.plot(distance_df[male_female_distances[0]], color = 'rebeccapurple')
    plt.xlabel('Camera frame number')
    plt.ylabel('Distance')
    plt.legend(['Male nose - Female nose', 'Male nose - Female Tail'], frameon = False,
               loc = 'upper left')
    sns.despine(top=True, right=True, left = True)


    ymin = 0
    ymax = 2100
    ax3.set_ylim([ymin, ymax])

    # xmin = 0
    # xmax =
    # ax3.set_xlim([xmin, xmax])

    # manual scale bar (and hide y axis except title)
    ax3.axes.get_yaxis().set_ticks([])
    plt.axvline(x = 0, ymin = 0.75, ymax =  0.85, color = 'black',
                label = '200 Pixels')
    ax3.text(x = 0 + 100, y = ymax * 0.79, s = '200 pixels', color = 'black')

    # highlight time regions of interest
    # genital_sniff = np.array([[3600, 4200], [0, 0]])
    # plt.axvspan(genital_sniff[0, 0], genital_sniff[0, 1], color = 'red', alpha = 0.5)

    plt.show()

    return 0

def plot_location(coord_df, body_part):
    # TODO: Add start point and end point for plotting
    """
    Plot 2D location of mice
    :param coord_df: dataframe containing coordinates (x, y) of each body part
    :param body_part: which body part to plot
    :return:
    """
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for part in body_part:
        x_coord = coord_df[part + '_x']
        y_coord = coord_df[part + '_y']
        ax1.plot(x_coord, y_coord)

    sns.despine(top = True, right = True, left = True, bottom = True)
    plt.legend(body_part, frameon = False)
    plt.show()


def animate_location(coord_df, body_part, export = False):
    # TODO: allow dealing with two body parts
    # TODO: plot distance between two body parts alongside
    """
    Plot animated 2D location of mice
    :return:
    """
    x_coord = coord_df[body_part + '_x']
    y_coord = coord_df[body_part + '_y']

    fig, ax = plt.subplots(figsize = (5, 3))
    ax.set(xlim = (min(x_coord), max(x_coord)),
           ylim = (min(y_coord), max(y_coord))
           ) # set limits to avoid rescaling during animation

    sns.despine(bottom = True, top = True, left = True, right = True)

    x_y_coord = np.array([x_coord, y_coord])

    def update(i, fig, scat):
        scat.set_offsets((x_y_coord[:, i]))
        ax.set_title('Frame ' + str(i))
        # add legend
        ax.legend(['Male nose'], frameon = False)
        return scat

    x = [x_coord[0]]
    y = [y_coord[0]]
    scat = plt.scatter(x, y, c = x)
    anim = animation.FuncAnimation(fig, update, fargs=(fig, scat),
                                   frames=len(x_coord), interval=100) # what does interval do?


    # plt.draw()
    plt.show()

    if export is True:
        # TODO: export animation
        pass

def animate_location_2part(coord_df, body_part, export = False):
    # TODO: generalise this to any number of body parts
    # TODO: plot distance between two body parts alongside
    """
    Plot animated 2D location of mice
    :return:
    """
    x_coord_1 = coord_df[body_part[0] + '_x']
    y_coord_1 = coord_df[body_part[0] + '_y']

    x_coord_2 = coord_df[body_part[1] + '_x']
    y_coord_2 = coord_df[body_part[1] + '_y']

    fig, ax = plt.subplots(figsize = (5, 3))
    # TODO: This is not strictly true, need to set to limit of both parts
    ax.set(xlim = (min(x_coord_1), max(x_coord_1)),
           ylim = (min(y_coord_1), max(y_coord_1))
           ) # set limits to avoid rescaling during animation

    sns.despine(bottom = True, top = True, left = True, right = True)

    x_y_coord_1 = np.array([x_coord_1, y_coord_1])
    x_y_coord_2 = np.array([x_coord_2, y_coord_2])

    def update(i, fig, scat_1, scat_2):
        scat_1.set_offsets((x_y_coord_1[:, i]))
        scat_2.set_offsets((x_y_coord_2[:, i]))
        ax.set_title('Frame ' + str(i))
        # add legend
        ax.legend(['Male nose', 'Female nose'], frameon = False)
        return scat_1, scat_2

    x_1 = x_coord_1[0]
    y_1 = y_coord_1[0]

    x_2 =  x_coord_2[0]
    y_2 =  y_coord_2[0]

    scat_1 = plt.scatter(x_1, y_1, c = 'blue')
    scat_2 = plt.scatter(x_2, y_2, c = 'red')

    anim = animation.FuncAnimation(fig, update, fargs=(fig, scat_1, scat_2),
                                   frames=len(x_coord_1), interval=100) # what does interval do?


    # plt.draw()
    plt.show()

    if export is True:
        # TODO: export animation
        pass

def animate_location_distance():
    # animate both location and distance
    pass

def animate_distance():
    # animate distance
    pass



def main(plot_distance = True, plot_coord = True, plot_animation = True):
    # load data
    if plot_distance is True:
        file_name = 'data/distance_df.pkl'
        with open(file_name, 'rb') as f:
            distance_df = pickle.load(f)

        # visualise distance between body parts
        visualise_distance(distance_df)

    coord_file_name = 'data/body_part_loc_df.pkl'
    with open(coord_file_name, 'rb') as f:
        coord_df = pickle.load(f)

    # specify body parts to plot / animate
    body_part = ['male_nose', 'female_nose']

    if plot_coord is True:
        plot_location(coord_df, body_part)

    if plot_animation is True:
        # body_part = 'male_nose'
        # animate_location(coord_df, body_part)
        body_part = ['male_nose', 'female_nose']
        animate_location_2part(coord_df, body_part, export = False)


if __name__ == '__main__':
    main(plot_distance = False, plot_coord = False, plot_animation = True)