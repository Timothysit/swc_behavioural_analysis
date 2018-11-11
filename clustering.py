# performs clustering on behvoural metrics over time frames

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
# import ggplot
from scipy.signal import resample

#import swat
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# clustering algorithms
from sklearn.cluster import KMeans

from classify_behav import discretize_df
from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.palettes import all_palettes
from bokeh.io import export_png

# video cutting
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from itertools import groupby

# feature analysis
from sklearn import preprocessing

# Dynamic time warping clustering
from tslearn.clustering import TimeSeriesKMeans

# Loading bar (for long for loops)
from tqdm import tqdm



# categorical heatmap
import catheat
import plotly.plotly as py # see: https://plot.ly/python/colorscales/#custom-discretized-heatmap-colorscale
from bokeh.plotting import figure

def bin_dataframe(data_vec, timeBin = 1, fs = 30):
    """
    Bin dataframes
    :param dataframe:
    :param timeBin: time bin to reduce the dataframe (seconds)
    :return:
    """

    frame_bin = fs / timeBin  # number of frames per bin
    new_sample_num = int(len(data_vec) / frame_bin)

    down_sampled_data = resample(data_vec, new_sample_num)

    return down_sampled_data




def run_tsne(dataframe):
    """
    Function to take in a dataframe and generate a T-SNE
    :return: T-SNE
    """
    tsne_model = TSNE(n_components=2, perplexity = 50, random_state = 10)
    tsne_results = tsne_model.fit_transform(dataframe)
    # tsne_plot = ggplot(tsne_results)
    # print(tsne_results)

    return tsne_results

def cluster(dim_reduced_data, method = 'kmeans', n_clusters = 4, compute_distance = False):
    """

    :param dim_reduced_data:
    :param method: method for performing clustering,
    if tslearn, you need to feed in the raw data, instead of dim_reduced_data
    :param n_clusters:
    :param compute_distance:
    :return:
    """
    if method == 'kmeans':
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(dim_reduced_data)
        clustered_labels = kmeans.predict(dim_reduced_data)
        return clustered_labels
    if compute_distance:
        """
        Find centroid, the compute and the distance of each point from the cluster
        """
        pass
    if method == 'tslearn':
        # Note that this takes the raw data instead of dim_reduced_data
        coord_df = dim_reduced_data

        # TODO: Pre-processing, normalise or scale data

        # Reshape data to be numTimeSeries x length x numVariable
        # a not-so-elegant approach to this (bin each variable), then layer then one behind another
        time_bin_width = int(30)
        if coord_df.shape[0] % time_bin_width != 0:
            trim_length = (coord_df.shape[0] // time_bin_width) * time_bin_width
            coord_df = coord_df[:trim_length]
        num_sample = int(coord_df.shape[0] / time_bin_width)
        num_features = int(coord_df.shape[1])
        multi_var_ts_3D = np.zeros((num_sample, time_bin_width, num_features))
        feature_idx = 0
        for feature in coord_df:
            multi_var_ts_3D[:, :, feature_idx] = coord_df[feature].values.reshape(-1, time_bin_width)
            feature_idx + 1
        # run dynamic time warping and clustering

        dtw_cluster_model = TimeSeriesKMeans(n_clusters = n_clusters, metric = 'dtw')
        dtw_cluster_model = dtw_cluster_model.fit(multi_var_ts_3D)
        dtw_cluster_labels = dtw_cluster_model.predict(multi_var_ts_3D)
        dtw_cluster_inertia = dtw_cluster_model.inertia_


        return dtw_cluster_labels, dtw_cluster_inertia



def viz_cluster(dim_reduced_df, clustered_labels = None):
    """
    Plots clustering / tSNE results
    :return:
    """
    if clustered_labels is None:
        plt.figure()
        ax1 = plt.scatter(dim_reduced_df[:, 0], dim_reduced_df[:, 1])
        plt.show()
    else:
        plt.figure()
        ax1 = plt.scatter(dim_reduced_df[:, 0], dim_reduced_df[:, 1],
                          c = clustered_labels, cmap = 'viridis')
        sns.despine(top = True, right = True,
                    left = True, bottom = True)
        plt.title('t-SNE and k-means clustering')
        plt.show()

def extract_cluster_time(clusterd_labels, time_bin):
    """
    Extract the time frame and their cluster identity
    :param time_bin: duration of time each label represent
    :param clusterd_labels:
    :return:
    """
    pass


def find_optimal_k(data, method = 'dtw_k_means'):
    """
    Find the optimal number of clusters for k-means clustering
    :return:
    """
    num_cluster_list = np.arange(3, 10) # list of clusters to try out

    if method == 'dtw_k_means':
        inertia_store = list()
        for num_cluster in tqdm(num_cluster_list):
            _, dtw_cluster_inertia = cluster(data, method = 'tslearn', n_clusters = num_cluster)
            inertia_store.append(dtw_cluster_inertia)

    plt.figure()
    plt.plot(num_cluster_list, inertia_store)
    sns.despine(top = True, right = True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()





def viz_cluster_ethogram(clusterd_labels, method = 'sns'):
    if method == 'sns':
        plt.figure()
        sns.heatmap([clusterd_labels])
        plt.show()
    elif method == 'bokeh':
        # define colormap

        label_df = dict()
        label_df['Behaviour'] = clusterd_labels
        label_df['Time'] = np.arange(0, len(clusterd_labels))
        label_df['Mice'] = np.repeat(1, len(clusterd_labels))

        label_df = pd.DataFrame.from_dict(label_df)
        source = ColumnDataSource(label_df)

        # colors = ["#EF5350", "#9C27B0", "#42A5F5", "#66BB6A"]
        colors = all_palettes['Viridis'][4]
        mapper = LinearColorMapper(palette=colors, low=label_df.Behaviour.min(), high=label_df.Behaviour.max())

        p = figure(plot_width=600, plot_height=150, title="Ethogram",
                   toolbar_location=None, tools="", x_axis_location="below",
                   x_range = (min(label_df['Time']), max(label_df['Time'])), y_range = (0.75, 1.25))

        # actual 'heatmap'
        p.rect(x="Time", y = "Mice", width=1, height=1, source=source,
               line_color=None, fill_color=transform('Behaviour', mapper))

        # color bar
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             # formatter=PrintfTickFormatter(format="%d%%")
                             title = 'Cluster')
        # see: https://bokeh.pydata.org/en/latest/docs/reference/models/annotations.html for colorbar settings
        # TODO: separate colorbar categories by some space
        # TODO: export figures

        # hide y-axis
        p.yaxis.visible = False

        # add x-axis title
        p.xaxis.axis_label = 'Time (s)'

        p.add_layout(color_bar, 'right')

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "5pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0

        export_png(p, filename="clustering_heatmap.png") # still have dpi issues

        show(p)


def extract_cluster_video(cluster_labels, cut_video = False, video_path = None, cut_and_join_video = False):
    """
    Extract part of video corresponding to particular clusters
    Also computes the longest duration of each label
    :param cluster_labels: vector giving the label for each time frame
    :return cluster_max_time: dataframe containing the max start and end time for each label
    """
    num_unique_labels = len(np.unique(cluster_labels))

    cluster_max_time = dict()
    cluster_max_time['Cluster'] = np.arange(0, num_unique_labels)

    groups = groupby(cluster_labels)
    label_seq = list()
    duration_seq = list()

    start_time_list = list()
    end_time_list = list()

    for label, duration in groups:
        label_seq.append(label)
        duration_seq.append(sum(1 for _ in duration))

    for label in np.unique(cluster_labels):
        duration_subset = np.array(duration_seq)[label_seq == label]
        duration_max = max(duration_subset) # find maximum continuation of a label
        # currently assumes only one max
        # back-track to find when this continuation started (w/ respect to the entire video
        duration_max_idx = np.intersect1d(np.where(np.array(label_seq) == label)[0],
                                          np.where(np.array(duration_seq) == duration_max)[0]
                                        )
        # if more than one maximum, then choose the first one
        # TODO: think about how to improve about this implementation (maybe using both?)
        if len(duration_max_idx) > 1:
            duration_max_idx = duration_max_idx[0]

        start_time = sum(duration_seq[0:int(duration_max_idx)])
        end_time = start_time + duration_max

        start_time_list.append(start_time)
        end_time_list.append(end_time)


    cluster_max_time['Start'] = start_time_list
    cluster_max_time['End'] = end_time_list

    # convert from dictionary to pandas dataframe
    cluster_max_time = pd.DataFrame.from_dict(cluster_max_time)

    if cut_video:
        # loop through the dictionary and cut video
        for index, row in cluster_max_time.iterrows():
            label = row['Cluster']
            start_time = row['Start']
            end_time = row['End']
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname = str(label) + '_behaviour.mp4')
    if cut_and_join_video:
        pass
    # TODO work on cutting and joining
    # see: https://stackoverflow.com/questions/50594412/cut-multiple-parts-of-a-video-with-ffmpeg
    # ffmpeg -i video -vf "select='between(t,4,6.5)+between(t,17,26)+between(t,74,91)',setpts=N/FRAME_RATE/TB" -af "aselect='between(t,4,6.5)+between(t,17,26)+between(t,74,91)',asetpts=N/SR/TB" out.mp4


    return cluster_max_time

def viz_cluster_factors(coord_df_binned, cluster_labels, save_fig = True):
    """
    Visualise the value of each feature for each cluster (using a heatmap)
    :param coord_df_binned:
    :param cluster_labels:
    :return:
    """

    # scale features
    min_max_scaler = preprocessing.MinMaxScaler()
    x = coord_df_binned.values
    x_scaled = min_max_scaler.fit_transform(x)

    coord_df_binned_normal = pd.DataFrame(x_scaled)

    # coord_df_binned_normal['cluster'] = pd.Series(cluster_labels, dtype = 'category')

    # coord_df_binned_normal = coord_df_binned_normal.sort_values(coord_df_binned_normal['cluster'] )

    fig, axn = plt.subplots(2, 2, sharex=True, sharey=False, figsize = (8, 8))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.heatmap(coord_df_binned_normal[cluster_labels == i], ax=ax,
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cmap = "YlGnBu",
                    cbar_kws = {'label': 'Scaled score'},
                    cbar_ax = None if i else cbar_ax)
        ax.tick_params(axis='both', which='both', length=0)

    fig.text(0.5, 0.04, 'Behavioural Feature', ha = 'center', fontsize= 14)
    fig.text(0.04, 0.5, 'Time bin', va = 'center', rotation = 'vertical', fontsize=14)

    # set context to make things larger
    # sns.set_context('talk')

    if save_fig is True:
        plt.savefig(fname = 'figures/feature_analysis.png', dpi = 300)

    plt.show()

def viz_cluster_prop(cluster_labels, save_fig = True):
    """
    Visualise the proportion of each cluster
    :param cluster_labels:
    :param save_fig:
    :return:
    """

    unique_label, counts = np.unique(cluster_labels, return_counts = True)
    prop = counts / sum(counts)

    # consistent colour palettes with previous plot
    colors = all_palettes['Viridis'][4]
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x = unique_label + 1, y = prop, palette = colors)
    sns.despine(top = True, bottom = True)
    # ax.set(xlabel = 'Cluster', ylabel = 'Proportion')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion')

    if save_fig is True:
        plt.savefig(fname = 'figures/clustering_labels_prop.png', dpi = 300)

    plt.show()



def main():
    # test data
    t_sne_test_df = pd.DataFrame({"frame": np.array([1, 2, 3, 4, 5]),
                                  "angle": np.array([0.55, 0.60, 1.2, 0.8, 1.4]),
                                   "relative_distance": np.array([2, 3, 7, 2, 2])})

    # actual data
    loc_file = 'data/distance_df.pkl'
    with open(loc_file, 'rb') as f:
        coord_df = pickle.load(f)

    # bin TODO: look whether there is a way to vectorise this

    coord_df_binned = dict()

    for column in coord_df:
        # coord_df_binned[column] = discretize_df(coord_df[column], bin_width = 0.3)
        coord_df_binned[column] = bin_dataframe(coord_df[column], timeBin = 1, fs = 30)

    coord_df_binned = pd.DataFrame.from_dict(coord_df_binned)

    # rename female to girl, then male to boy (allow easier subsetting)
    coord_df_binned.columns = coord_df_binned.columns.str.replace('female', 'girl')
    coord_df_binned.columns = coord_df_binned.columns.str.replace('male', 'boy')

    # subset dataframe to only contain boy and girl
    columns_to_use = coord_df_binned.columns[coord_df_binned.columns.to_series().str.contains('boy')
                                              & coord_df_binned.columns.to_series().str.contains('girl')]

    coord_df_binned = coord_df_binned[columns_to_use]

    print(coord_df_binned.shape)

    tsne_results = run_tsne(coord_df_binned)

    # do clustering and get labels

    labels = cluster(tsne_results, method = 'kmeans', n_clusters = 4)

    # use the label to extract videos
    # extract_cluster_video(labels, cut_video = True, video_path = '/media/timothysit/Seagate Expansion Drive1/18_10_29_mf_interaction_right.avi')

    # feature analysis on clusters
    # viz_cluster_factors(coord_df_binned, labels)

    # plot cluster
    # viz_cluster(tsne_results, labels)

    # extract cluster time bins

    # visualise clusters over time
    # viz_cluster_ethogram(labels, method = 'bokeh')

    # save clustering results
    # with open('data/cluster_result' + '.pkl', 'wb') as f:
    #     pickle.dump(labels, f)

    # visualise cluster proportions
    # viz_cluster_prop(labels)

    # Dynamic time warping clutering

    dtw_labels, _ = cluster(coord_df, method = 'tslearn', n_clusters = 7)
    extract_cluster_video(dtw_labels, cut_video=True,
                          video_path='/media/timothysit/Seagate Expansion Drive1/18_10_29_mf_interaction_right.avi')
    # find_optimal_k(data = coord_df, method = 'dtw_k_means')


if __name__ == '__main__':
    main()

