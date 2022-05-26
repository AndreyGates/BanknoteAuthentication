from csv import reader
from matplotlib.colors import ListedColormap, to_rgb
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from random import randrange

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

# Расчет точности модели
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if predicted == None:
            continue
        elif actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Кросс-валидация k-fold (вследствие ограниченного кол-ва данных)
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds): 
        # разделяем данные на несколько подгрупп, 
        # каждая из которых будет тестовой выборкой при многократной оценке модели
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Оценка алгоритма для каждой пары "обучающая выборка - тестовая выборка"
def evaluate_algorithm(dataset, algorithm, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set, test_set)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores # точности для каждой тестовой выборки

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