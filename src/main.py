from random import seed

from auxil import *
from DecisionTreeClassifier import *

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
    filename = 'data_banknote_authentication.csv'
    dataset = load_csv(filename)

    # Перевод данных в float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Оценка алгоритма
    n_folds = 3 # кол-во подгрупп cross-validation разбиения

    max_depth = 5 # максимальная глубина дерева
    min_size = 15 # минимальное число элементов в одном узле

    DTC = DecisionTreeClassifier(max_depth, min_size)

    scores = DTC.evaluate_algorithm(dataset, DTC.decision_tree, n_folds)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    visualize_DT(max_depth, min_size)

main()