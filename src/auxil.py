from csv import reader
from matplotlib.colors import ListedColormap, to_rgb
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier

# Загрузка CSV файла с данными
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# Преобразования string в float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def visualize_DT(max_depth, min_size):
    # Загрузка и подготовка данных
    filename = 'data_banknote_authentication.csv'
    dataset = load_csv(filename)

    # Перевод данных в float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    ds = np.array(dataset)
    X = ds[:, [0, 1, 2, 3]]
    y = ds[:, [4]]

    #clf = tree.DecisionTreeClassifier(random_state=69, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    #model = clf.fit(X, y)

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_size)
    model = clf.fit(X, y)

    fig, ax1 = plt.subplots(figsize=[21, 12])
    colors = ['red', 'green']

    artists = tree.plot_tree(model, feature_names=["variance", "skewness", 'kurtosis', 'entropy'], class_names=['fake', 'true'],
                                filled=True, rounded=True, ax=ax1, fontsize=8)
    for artist, impurity, value in zip(artists, model.tree_.impurity, clf.tree_.value):
        # let the max value decide the color; whiten the color depending on impurity (gini)
        r, g, b = to_rgb(colors[np.argmax(value)])
        f = impurity * 2 # for N colors: f = impurity * N/(N-1) if N>1 else 0
        artist.get_bbox_patch().set_facecolor((f + (1-f)*r, f + (1-f)*g, f + (1-f)*b))
        artist.get_bbox_patch().set_edgecolor('black')

    fig.savefig("decision_tree.png")
    #plt.tight_layout()
    #plt.show()