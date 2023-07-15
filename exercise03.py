import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


def entropy(option1, option2):
    p1 = option1 / (option1+option2)
    p2 = option2 / (option1+option2)
    return -p1 * math.log2(p1) - p2 * math.log2(p2)


if __name__ == "__main__":

    exercise = 1

    if exercise == 1:
        pass

    else:

        iris = load_iris()

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

        # Initialize a decision tree classifier
        tree_clf = DecisionTreeClassifier(max_depth=3)

        # Fit the decision tree to the training data
        tree_clf.fit(X_train, y_train)

        # Plot the decision tree
        plot_tree(tree_clf, filled=True)

        # Plot the decision tree
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        plot_tree(tree_clf, filled=True, ax=axes)
        plt.show()

        # Make predictions on the test data
        y_pred = tree_clf.predict(X_test)

        # Compute the accuracy of the model
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")


