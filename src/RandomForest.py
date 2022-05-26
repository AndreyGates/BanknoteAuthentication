from random import randrange
from DecisionTreeClassifier import * 

class RandomForest():
    def __init__(self, max_depth, min_size, sample_size, n_trees, n_features):
        self.max_depth = max_depth # the max depth of a single tree
        self.min_size = min_size # the min node size 
        self.sample_size = sample_size # the ratio for bootstrapping
        self.n_trees = n_trees # random forest size (number of trees)
        self.n_features = n_features # number of features for feature selection (after bootstrapping)

        self.trees = []

    # Create a random subsample from the dataset with replacement
    def subsample(self, dataset, ratio):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample
    
    # Make a prediction with a list of bagged trees (the class which is most frequent among the trees for an instance)
    def bagging_predict(self, trees, row):

        predictions = [DecisionTreeClassifier.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)
    
    # Random Forest Algorithm
    def random_forest(self, train, test):
        trees = list()
        for i in range(self.n_trees):
            DT = DecisionTreeClassifier(self.max_depth, self.min_size)

            sample = self.subsample(train, self.sample_size)
            tree = DT.build_tree(sample, self.n_features)
            trees.append(tree)

        self.trees = trees

        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions