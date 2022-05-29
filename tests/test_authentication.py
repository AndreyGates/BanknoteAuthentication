# from authentication.visualization import visualize_RF
from authentication.auxil import load_csv, str_column_to_float, evaluate_algorithm
from authentication.RandomForest import RandomForest

from authentication.visualization import visualize_RF

from random import seed
from math import sqrt


def test_authentication():
    seed(1)

    # Загрузка и подготовка данных
    filename = 'C:/Users/pisar/Desktop/GitHub/Repositories/BanknoteAuthentication/tests/data_banknote_authentication.csv'

    dataset = load_csv(filename)

    # Перевод данных в float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Оценка алгоритма

    # Decision Tree features
    n_folds = 3  # кол-во подгрупп cross-validation разбиения
    max_depth = 2  # максимальная глубина дерева
    min_size = 10  # минимальное число элементов в одном узле

    # Random Forest features
    sample_size = 0.5  # доля ко всей выборке
    n_features = int(sqrt(len(dataset[0])-1))  # кол-во свойств для bagging
    # n_trees = 5  # кол-во деревьев

    for n_trees in [1]:
        RF = RandomForest(max_depth, min_size, sample_size, n_trees, n_features)
        scores = evaluate_algorithm(dataset, RF.random_forest, n_folds)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
        
        visualize_RF(dataset)

        assert sum(scores)/float(len(scores)) >= 80

def test_visualization():
    pass

def main():
    test_authentication()
