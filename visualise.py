import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np

def visualise_distance(distance_df):

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




    return None

def main():
    # load data
    file_name = 'data/distance_df.pkl'
    with open(file_name, 'rb') as f:
        distance_df = pickle.load(f)

    # visualise distance between body parts
    visualise_distance(distance_df)

if __name__ == '__main__':
    main()