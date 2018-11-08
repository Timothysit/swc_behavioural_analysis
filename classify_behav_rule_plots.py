from pickle import dump as pdump
from pickle import load as pload

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

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

    data = classif.as_matrix().transpose()[:, :]

    # ethogram
    plt.imshow(data)
    plt.gca().set_aspect(300)
    plt.yticks(labels_len, labels)
    plt.xlabel('# frame bin (100 ms)')

    plt.savefig('./data/classification_rule_based_ethogram' + ext, bbox_inches='tight', dpi=dpi)
    plt.show()

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
    plt.tight_layout()
    plt.savefig('./data/classification_rule_based_prop' + ext, dpi=dpi)
    plt.show()

    # multiple labels
    plt.figure()
    plt.hist(np.nansum(data, 0), 6, (0, 6), density=True)
    plt.ylabel('Proportion of intervals')
    plt.xlabel('# of categories per interval')
    plt.ylim((0, 1))

    seaborn.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig('./data/classification_rule_based_hist' + ext, dpi=dpi)
    plt.show()

    # set overlap
    set1 = set(np.array(np.where(classif['nose2body'])).flatten())
    set2 = set(np.array(np.where(classif['nose2nose'])).flatten())
    set3 = set(np.array(np.where(classif['nose2genitals'])).flatten())

    plt.figure()
    venn3([set1, set2, set3], ('nose2body', 'nose2nose', 'nose2genitals'))

    plt.tight_layout()
    plt.savefig('./data/classification_rule_based_venn' + ext, dpi=dpi)
    plt.show()

    pass


if __name__ == '__main__':
    make_plots()
