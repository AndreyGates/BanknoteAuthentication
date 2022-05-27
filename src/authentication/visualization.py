"""import matplotlib.colors as mcolors
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

from authentication.auxil import str_column_to_float

"""


def visualize_RF(dataset):
    pass


"""

def visualize_RF(dataset):
    # Перевод данных в float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    ds = np.array(dataset)
    X = ds[:, [0, 1, 2, 3]]
    y = ds[:, [4]]
    y = y.ravel() # flattening

    rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=3)
    rf.fit(X, y)

    colors = ['red', 'blue'] # visualization of nodes

    fig, axes = plt.subplots(nrows=1, ncols=rf.n_estimators, figsize=(10,2), dpi=900)
    for index in range(0, rf.n_estimators):
        artists = tree.plot_tree(rf.estimators_[index],
                    feature_names=["variance", "skewness", 'kurtosis', 'entropy'],
                    class_names=['fake', 'true'],
                    filled=True,
                    ax=axes[index]);

        for artist, impurity, value in zip(artists, rf.estimators_[index].tree_.impurity,  rf.estimators_[index].tree_.value):
            # let the max value decide the color; whiten the color depending on impurity (gini)
            r, g, b = mcolors.to_rgb(colors[np.argmax(value)])
            f = impurity * 2 # for N colors: f = impurity * N/(N-1) if N>1 else 0
            artist.get_bbox_patch().set_facecolor((f + (1 - f) * r, f + (1 - f) * g, f + (1 - f) * b))
            artist.get_bbox_patch().set_edgecolor('black')

    # axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
    fig.savefig('C:/Users/pisar/Desktop/GitHub/Repositories/BanknoteAuthentication/src/random_forest.png')
"""
