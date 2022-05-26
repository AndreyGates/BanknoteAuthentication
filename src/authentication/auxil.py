from csv import reader
from random import randrange

# Загрузка CSV файла с данными
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# Преобразования string в float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

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