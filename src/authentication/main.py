from random import seed
from authentication.RandomForest import RandomForest
from numpy import sqrt

from authentication.auxil import *
from authentication.visualization import *

"""База данных включается в себя такие особенности изображений банкнот как:
    1) дисперсия преобразования вейвлет (непрерывные значения)
    2) асимметрия преобразования вейвлет (непрерывные значения)
    3) эксцесс преобразования вейвлет (непрерывные значения)
    4) энтропия изображения (непрерывные значения)

   А также целевую переменную (0 - банкнота поддельная, 1 - банкнота настоящая)
"""

def main():
    seed(1)

    # Загрузка и подготовка данных
    filename = 'C:/Users/pisar/Desktop/GitHub/Repositories/BanknoteAuthentication/src/data_banknote_authentication.csv'
    dataset = load_csv(filename)

    # Перевод данных в float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Оценка алгоритма

    # Decision Tree features
    n_folds = 3 # кол-во подгрупп cross-validation разбиения
    max_depth = 2 # максимальная глубина дерева
    min_size = 10 # минимальное число элементов в одном узле

    # Random Forest features
    sample_size = 0.5 # доля ко всей выборке
    n_features = int(sqrt(len(dataset[0])-1)) # кол-во свойств для bagging
    #n_trees = 5 # кол-во деревьев

    for n_trees in [1, 5, 10]:
        RF = RandomForest(max_depth, min_size, sample_size, n_trees, n_features)
        scores = evaluate_algorithm(dataset, RF.random_forest, n_folds)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    visualize_RF(dataset)

if __name__ == "__main__":
    main()