from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Node:
    """
    Node of a tree
    """
    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class MyDecisionTreeClassifier:
    """
    Build decision tree and predict class
    """
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def gini(self, groups, classes):
        """
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.

        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).

        >>> classifier = MyDecisionTreeClassifier(10)
        >>> classifier.gini([[1,2,3],[2,3,4]], [[1,1,2],[1,2,2]])
        0.8888888888888888
        """

        def difference(dict):
            dif = 0
            for value in dict.values():
                dif += (value / samples_num) ** 2
            return dif

        G = {}
        for group_num, group in enumerate(classes):
            class_samples_count = {}
            samples_num = len(group)
            for sample in group:
                class_samples_count[sample] = class_samples_count.get(sample,
                                                                      0) + 1
            G[group_num] = 1 - difference(class_samples_count)

        res = 0
        for group in G:
            res += G[group]
        return res

    def split_data(self, X, y):
        """
        Test all the possible splits in O(N^2)
        :param X: list of lists with real numbers
        :param y: classes of elements in X
        :return: index and threshold value
        """
        values_possible_dict = {}
        for sample in range(len(X)):
            for target_num, target in enumerate(X[sample]):
                if values_possible_dict.get(target_num, 0):
                    values_possible_dict[target_num].add(target)
                else:
                    values_possible_dict[target_num] = {target}

        gini_values_list = []
        gini_keys_list = []
        for target_num in values_possible_dict:
            for target_value in values_possible_dict[target_num]:
                group1 = []
                group1_class = []
                group2 = []
                group2_class = []
                for i, el in enumerate(X):
                    if el[target_num] <= target_value:
                        group1.append(el)
                        group1_class.append(y[i])
                    else:
                        group2_class.append(y[i])
                        group2.append(el)
                gini_values_list.append(
                    self.gini([group1, group2], [group1_class, group2_class]))
                gini_keys_list.append(tuple([target_num, target_value]))
        min_gini = min(gini_values_list)
        min_gini_index = gini_values_list.index(min_gini)

        return gini_keys_list[min_gini_index]

    def build_tree(self, X, y, depth=0):
        """
        Build recursively decision tree
        :param X: list of lists with real numbers
        :param y: classes of elements in X
        :param depth: max high of tree
        :return: nodes
        """
        gini = 1
        el_total = len(y)
        classes_count = {}
        for el in y:
            if el in classes_count:
                classes_count[el] += 1
            else:
                classes_count[el] = 1
        for class_count in classes_count.values():
            gini -= (class_count / el_total) ** 2

        new_node = Node(X, y, gini)

        if depth < self.max_depth and (len(classes_count) > 1):
            ind, threshold = self.split_data(X, y)
            new_node.threshold = threshold
            new_node.feature_index = ind
            group1 = []
            group1_classes = []
            group2 = []
            group2_classes = []
            for i, x in enumerate(X):
                if x[ind] <= threshold:
                    group1.append(x)
                    group1_classes.append(y[i])
                else:
                    group2.append(x)
                    group2_classes.append(y[i])

            new_node.left = self.build_tree(group1, group1_classes, depth + 1)
            new_node.right = self.build_tree(group2, group2_classes, depth + 1)

        return new_node

    def fit(self, X, y):
        """
        Basically wrapper for build tree
        :param X: list of lists with real numbers
        :param y: classes of elements in X
        """
        tree = self.build_tree(X, y, 0)
        self.tree = tree

    def predict(self, X_test):
        """
        Predict class for value by running decision tree
        :param X_test: list of testing values
        :return: the predicted class
        """
        X_classes = []
        for elements in X_test:
            tree = self.tree
            while tree.left is not None:
                index = tree.feature_index
                if elements[index] <= tree.threshold:
                    tree = tree.left
                else:
                    tree = tree.right
            X_classes.append(max(tree.y, key=tree.y.count))
        return X_classes


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    classifier = MyDecisionTreeClassifier(5)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)
    classifier.fit(X, y)
    predictions = classifier.predict(X_test)
    print(sum(predictions == y_test) / len(y_test))
