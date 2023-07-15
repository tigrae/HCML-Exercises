# imports here
import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import statistics

exercise = 2


class TShirtCustomer:
    def __init__(self, id, height, weight, size, dist):
        self.id = id
        self.height = height
        self.weight = weight
        self.size = size
        self.dist = dist

    def __str__(self):
        return f"Customer {self.id} with weight {self.weight} and height {self.height} wears size {self.size}."


def euclidean_distance(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


if __name__ == "__main__":

    if exercise == 1:
        """Code for exercise 1"""
        pass

    if exercise == 2:
        """Code for exercise 2"""
        heights = [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170]
        weights = [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63]
        sizes = ["M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "L", "L", "L", "L", "L", "L"]

        new_height = 165
        new_weight = 63

        customers = []

        for i, (height, weight, size) in enumerate(zip(heights, weights, sizes)):
            dist = euclidean_distance(new_weight, weight, new_height, height)
            customers.append(TShirtCustomer(i+1, height, weight, size, dist))

        customers.sort(key=lambda customer: customer.dist)

        k = 3
        accumulated_sizes = []
        for i in range(k):
            accumulated_sizes.append(customers[i].size)

        result = statistics.mode(accumulated_sizes)

        print(result)




    if exercise == 3:
        """Code for exercise 3"""

        # Load the breast cancer dataset
        data = load_breast_cancer()

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        # Create a support vector machine classifier
        svm = SVC()

        # Fit the classifier to the training data
        svm.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = svm.predict(X_test)

        # Calculate the accuracy, precision, and recall of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Print the results
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)