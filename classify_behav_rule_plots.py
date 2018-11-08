from pickle import dump as pdump
from pickle import load as pload

import numpy as np
import seaborn
from matplotlib import pyplot as plt


# from matplotlib import use as muse
# muse('SVG')


def load_pickle(filepath, mode='rb'):
    return pload(open(filepath, mode))


def save_pickle(filepath, objects, mode='wb'):
    with open(filepath, mode) as f:
        for o in objects:
            pdump(o, f)


def make_plots():
    dpi = 300
    ext = '.png'

    classif = load_pickle('./data/classification.pkl')
    labels = ('nose2body', 'nose2nose', 'nose2genitals', 'above',
              'following', 'standTogether', 'standAlone', 'walkAlone')
    labels_len = np.arange(labels.__len__())

    # ethogram
    data = classif.as_matrix().transpose()[:, :]
    plt.imshow(data)
    plt.gca().set_aspect(300)
    plt.yticks(labels_len, labels)
    plt.xlabel('# frame bin (100 ms)')
    plt.show()
    plt.savefig('./data/classification_rule_based_ethogram' + ext, dpi=dpi)

    # proportions
    prop = []
    out = []
    coldata = []

    for motif in classif:
        coldata = classif[motif]
        out = np.array(np.where(coldata > 0)).shape[1]
        prop.append(out)

    prop_norm = np.array(prop) / coldata.shape[0]

    plt.figure()
    plt.bar(labels_len, prop_norm)
    plt.xticks(labels_len, labels, rotation=45)
    plt.ylim((0, 1))
    plt.ylabel('Proportion')  # [classified (100ms) intervals / all intervals]
    seaborn.despine(top=True, right=True)
    plt.savefig('./data/classification_rule_based_prop' + ext, dpi=dpi)
    plt.show()


    pass


# make_plots()

if __name__ == '__main__':
    make_plots()