import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
import load_data # for the filtering functions
from scipy import fftpack
import cv2 # for dealing with video
from bokeh.palettes import all_palettes
import skvideo.io # for saving video

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

def compare_filters(unfiltered_df, body_part, fs = 30):
    """
    Plots signal before and after filtering, and also compare between filters
    :param unfiltered_df:
    :param body_part:
    :return:
    """
    # TODO: compare filters (both frequency and time domain)
    body_part_x = unfiltered_df[body_part + '_x']
    body_part_y = unfiltered_df[body_part + '_y']

    frame = np.arange(len(body_part_x))
    time = frame / fs

    body_part_x_l_filterd = load_data.smooth_data(body_part_x, n = 20, method = 'lfilt')
    body_part_x_savgol_filtered = load_data.smooth_data(body_part_x, method = 'savgol',
                                                         savgol_window=51, savgol_degree=3)
    # usual savgol_window = 51, and savegol_degree = 3

    # time domain comparison of filters
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(time, body_part_x) # change alpha values
    ax1.plot(time, body_part_x_l_filterd)
    ax1.plot(time, body_part_x_savgol_filtered)
    ax1.legend(['No filter', 'Linear filter', 'Savgol filter'], frameon = False)
    plt.title(['Male nose x position'])
    sns.despine(top = True, right = True)
    plt.show()

    # frequency domain comparison of filters
    plt.figure()
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(fftpack.fft(body_part_x)) # change alpha values
    ax2.plot(fftpack.fft(body_part_x_l_filterd))
    ax2.plot(fftpack.fft(body_part_x_savgol_filtered))
    ax2.legend(['No filter', 'Linear filter', 'Savgol filter'], frameon = False)
    plt.title(['Male nose x position'])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    sns.despine(top = True, right = True)
    plt.show()

    # frequency domain comparison attempt 2
    plt.figure()
    ax2 = plt.subplot(1, 1, 1)
    freq, power = plot_freq_spectrum(body_part_x)
    ax2.plot(freq, power) # change alpha values

    freq, power = plot_freq_spectrum(body_part_x_l_filterd)
    ax2.plot(freq, power)

    freq, power = plot_freq_spectrum(body_part_x_savgol_filtered)
    ax2.plot(freq, power)

    ax2.legend(['No filter', 'Linear filter', 'Savgol filter'], frameon = False)
    plt.title(['Male nose x position'])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    sns.despine(top = True, right = True)
    plt.show()




def plot_freq_spectrum(time_series, fs = 30, plot = False):
    """
    see:
    https://ericstrong.org/fast-fourier-transforms-in-python/
    https://plot.ly/matplotlib/fft/
    :param time_series:
    :param fs: sampling rate
    :return:
    """
    # TODO: Make plot of frequency domain for coordinate position

    n = len(time_series)  # length of signal
    # Nyquist sampling limit
    T = 1/fs
    x = np.linspace(0.0, 1.0/(2.0 * T), int(n/2)) # not too sure what 1/2T does...

    filterd_signal = fftpack.fft(time_series)
    y = 2/n * np.abs(time_series[0:np.int(n/2)]) # positive freqs only, and normalisation by dividing by N

    if plot:
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
    else:
        return x, y

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_bgr(rgbcolor):
    rgbcolor = rgbcolor
    b = rgbcolor[2]
    g = rgbcolor[1]
    r = rgbcolor[0]
    bgr = (b, g, r)
    return(bgr)


def get_color_scheme(palette, no_colors):
    """
    Define color palette and amount of colors from bokeh.all_palettes and convert this to bgr
    :param palette:
    :param no_colors:
    :return:
    """
    colors_hex = all_palettes[palette][no_colors]  # can change which color palette to use
    colors_rgb = []
    colors_bgr = []

    for color in colors_hex:  # change from hex to color rgb
        colors_rgb.append(hex_to_rgb(color))
    for color in colors_rgb:  # change from rgb to bgr
        colors_bgr.append(rgb_to_bgr(color))
    colors = colors_bgr
    return colors


def color_behav(dataarray, array_index, behaviour, color_index, colors, position_index, frame, x_pos=1080, change = True):
    """
    Make string of the eight behaviours colored if they are present in the numpy array corresponding to the current frame
    Make string default color if not
    :param dataarray: numpyarray
    :param array_index: int
    :param behaviour: string
    :param color_index: int
    :param default_color: bgr tuple
    :param position_index: int
    :param frame:
    :param x_pos:
    :return:
    """
    # parameters for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    positions = [150, 250, 350, 450, 550, 650, 750, 850]
    fontScale = 2.3
    default_color = (100, 100, 100)  # default color if no
    lineType = 2
    thickness = 6

    if change == True:
        if dataarray[array_index] == 1.0:
            fontColor = colors[color_index]
            frame = cv2.putText(frame, behaviour, (x_pos, positions[position_index]), font, fontScale, fontColor, thickness,
                                lineType)

        elif dataarray[array_index] == 0.0:

            fontColor = default_color
            frame = cv2.putText(frame, behaviour, (x_pos, positions[position_index]), font, fontScale, fontColor, thickness,
                                lineType)

    else:
        fontColor = default_color
        frame = cv2.putText(frame, behaviour, (x_pos, positions[position_index]), font, fontScale, fontColor, thickness,
                            lineType)

def color_cluster(current_cluster, cluster_array, cluster_array_i, colors, color_index, frame, position=(50, 940), change=False):
        # text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3
        lineType = 2
        thickness = 7

        frame = cv2.putText(frame, current_cluster, position, font, fontScale, colors[color_index], thickness, lineType)

def get_video_frames(video_file_path, coord_df, add_behaviour = True, add_clusters = True):
    """
    Obtains video frames (and do some pre-processing to make things more manageable, such as gray-scaling)
    :param video_file_path:
    :return:
    """
    cap = cv2.VideoCapture(video_file_path)

    # starting parameters
    frame_num = 0
    numpy_i = 0  # for indexing through our behaviour array
    cluster_i = 0  # for counting up the frames until cluster change
    cluster_array_i = 0  # for indexing throught he cluster array
    x_pos = 1400  # position of text on x axis should be same for all behaviour titles
    colors = get_color_scheme("Set1", 8)  # chose color scheme behaviour
    colors2 = get_color_scheme("GnBu",4)  # get color scheme for clusters

    #  load in behaviour
    if add_behaviour == True:
        behav_file_path = 'data/classification.pkl'
        with open(behav_file_path, 'rb') as f:
            classification_df = pickle.load(f)

    nose2body = classification_df["nose2body"]
    nose2nose = classification_df["nose2nose"]
    nose2genitals = classification_df["nose2genitals"]
    above = classification_df["above"]
    following = classification_df["following"]
    standtogether = classification_df["standTogether"]
    standAlone = classification_df["standAlone"]
    walkAlone = classification_df["walkAlone"]

    # load in clusters
    if add_clusters == True:
        cluster_file_path = 'data/cluster_result.pkl'
        with open(cluster_file_path, 'rb') as f:
            cluster_df = pickle.load(f)

    # initiate video writer (for saving the video)
    writer = skvideo.io.FFmpegWriter("mouse_behaviour.mp4")

    # Plot video
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        horizontal_offset = 320
        vertical_offset = -30

        # set location of body parts
        male_nose = np.array([coord_df['male_nose_x'] - horizontal_offset, coord_df['male_nose_y']+vertical_offset])
        female_nose = np.array([coord_df['female_nose_x'] - horizontal_offset, coord_df['female_nose_y']+vertical_offset])

        male_tail = np.array([coord_df['male_tail_x'] - horizontal_offset, coord_df['male_tail_y']+vertical_offset])
        female_tail = np.array([coord_df['female_tail_x'] - horizontal_offset, coord_df['female_tail_y']+vertical_offset])

        male_left_ear = np.array([coord_df['male_left_ear_x'] - horizontal_offset, coord_df['male_left_ear_y']+vertical_offset])
        male_right_ear = np.array([coord_df['male_right_ear_x'] - horizontal_offset, coord_df['male_right_ear_y']+vertical_offset])

        female_left_ear = np.array([coord_df['female_left_ear_x'] - horizontal_offset, coord_df['female_left_ear_y']+vertical_offset])
        female_right_ear = np.array([coord_df['female_right_ear_x'] - horizontal_offset, coord_df['female_right_ear_y']+vertical_offset])


        # draw circle of body parts
        frame = cv2.circle(frame, center= tuple(male_nose[:, frame_num].astype(int)), radius = 20,
                           color = (255,255,255), thickness = -1)

        frame = cv2.circle(frame, center= tuple(male_tail[:, frame_num].astype(int)), radius = 20,
                           color = (255,255,255), thickness = -1)

        frame = cv2.circle(frame, center=tuple(male_left_ear[:, frame_num].astype(int)), radius=20,
                           color=(255,255,255), thickness=-1)

        frame = cv2.circle(frame, center=tuple(male_right_ear[:, frame_num].astype(int)), radius=20,
                           color=(255,255,255), thickness=-1)


        frame = cv2.circle(frame, center=tuple(female_nose[:, frame_num].astype(int)), radius=20,
                           color=(0, 0, 0), thickness=-1)

        frame = cv2.circle(frame, center=tuple(female_tail[:, frame_num].astype(int)), radius=20,
                           color=(0, 0, 0), thickness=-1)

        frame = cv2.circle(frame, center=tuple(female_left_ear[:, frame_num].astype(int)), radius=20,
                           color=(0, 0, 0), thickness=-1)

        frame = cv2.circle(frame, center=tuple(female_right_ear[:, frame_num].astype(int)), radius=20,
                           color=(0, 0, 0), thickness=-1)


        # draw lines connecting body parts (nose-tail, nose-ears)

        frame = cv2.line(frame, tuple(male_nose[:, frame_num].astype(int)),
                         tuple(male_tail[:, frame_num].astype(int)),
                         color=(255,255,255), thickness = 1)

        frame = cv2.line(frame, tuple(male_nose[:, frame_num].astype(int)),
                         tuple(male_left_ear[:, frame_num].astype(int)),
                         color=(255,255,255), thickness = 1)

        frame = cv2.line(frame, tuple(male_nose[:, frame_num].astype(int)),
                         tuple(male_right_ear[:, frame_num].astype(int)),
                         color=(255,255,255), thickness = 1)

        frame = cv2.line(frame, tuple(female_nose[:, frame_num].astype(int)),
                         tuple(female_tail[:, frame_num].astype(int)),
                         color=(0,0,0), thickness=1)

        frame = cv2.line(frame, tuple(female_nose[:, frame_num].astype(int)),
                         tuple(female_left_ear[:, frame_num].astype(int)),
                         color=(0, 0, 0), thickness=1)

        frame = cv2.line(frame, tuple(female_nose[:, frame_num].astype(int)),
                         tuple(female_right_ear[:, frame_num].astype(int)),
                         color=(0, 0, 0), thickness=1)

        # Insert text describing behaviour:
        if frame_num % 3 == 0: #  change to next behaviour
            numpy_i += 1

        color_behav(nose2body, numpy_i, "Nose2Body", 0, colors, 0, frame, x_pos)
        color_behav(nose2nose, numpy_i, "Nose2Nose", 1, colors, 1, frame, x_pos)
        color_behav(nose2genitals, numpy_i, "Nose2Genitals", 2, colors, 2, frame, x_pos)
        color_behav(above, numpy_i, "Above", 3, colors, 3, frame, x_pos)
        color_behav(following, numpy_i, "Following", 4, colors, 4, frame, x_pos)
        color_behav(standtogether, numpy_i, "StandTogether", 5, colors, 5, frame, x_pos)
        color_behav(standAlone, numpy_i, "StandAlone", 6, colors, 6, frame, x_pos)
        color_behav(walkAlone, numpy_i, "WalkAlone", 7, colors, 7, frame, x_pos)


        # Define current cluster
        if cluster_df[cluster_array_i] == 0:
            current_cluster = "Cluster1"
            color_index = 0
        if cluster_df[cluster_array_i] == 1:
            current_cluster = "Cluster2"
            color_index = 1
        if cluster_df[cluster_array_i] == 2:
            current_cluster = "Cluster3"
            color_index = 2
        if cluster_df[cluster_array_i] == 3:
            current_cluster = "Cluster4"
            color_index = 3

        if cluster_i == 30:  # 30 frames per second go to next spot of cluster array
            cluster_i = 0 #  reset the frame counting
            color_cluster(current_cluster, cluster_df,cluster_array_i, colors2, color_index, frame, change = True)
            cluster_array_i += 1

        else:
            cluster_i += 1
            color_cluster(current_cluster, cluster_df, cluster_array_i, colors2, color_index, frame, change=False)


        frame_num = frame_num + 1

        # save video
        writer.writeFrame(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    writer.close()

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))

    # return frames

def viz_ethogram(ethogram_vec):
    """
    visualises ethogram
    :param ethogram_vec:
    :return:
    """
    pass





def main(plot_distance = False, plot_coord = False, plot_animation = False, plot_filtering = False,
         plot_video = False, plot_ethogram = False):
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

    if plot_filtering is True:
        unfiltered_coord_file_name = 'data/body_part_loc_unfiltered_df.pkl'
        with open(unfiltered_coord_file_name, 'rb') as f:
            body_part_unfiltered_df = pickle.load(f)

        compare_filters(body_part_unfiltered_df, body_part = 'male_nose')

    if plot_video is True:
        video_path = 'data/18_10_29_mf_interaction_right.avi'
        frames = get_video_frames(video_path, coord_df)
        # play the video (checking)

    if plot_ethogram is True:
        ethogram_file = ''




if __name__ == '__main__':
    main(plot_distance = False, plot_coord = False, plot_animation = True,
         plot_filtering = False, plot_video = True)