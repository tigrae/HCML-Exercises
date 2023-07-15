import statistics
import matplotlib.pyplot as plt
import numpy as np


def get_mean(values):
    return sum(values)/len(values)


def get_st_dev(values):
    mean = get_mean(values)
    singular_dev = 0
    for value in values:
        singular_dev += (value - mean)**2
    singular_dev /= len(values)
    return singular_dev**0.5


if __name__ == '__main__':

    happiness = [15, 140, 280, 300, 350, 400, 439, 458, 438, 430, 410, 389, 290, 200, 105, 3]
    salary = [3, 30, 100, 120, 170, 180, 260, 290, 320, 350, 400, 428, 480, 515, 586, 599]
    hours_worked = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    fav_color = ["red", "blue", "yellow", "pink", "blue", "green", "red", "green", "black", "green",
                 "pink", "black", "orange", "green", "purple", "white"]

    print(f"Mean of salary = {get_mean(salary)}")
    print(f"St.-dev. of salary = {get_st_dev(salary)}")
    print(f"Mean of hrs worked = {get_mean(hours_worked)}")

    x = np.array(hours_worked)
    y = np.array(salary)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_dev = x - x_mean
    y_dev = y - y_mean

    beta_1 = np.sum(x_dev * y_dev) / np.sum(x_dev * x_dev)
    beta_0 = y_mean - beta_1 * x_mean

    print(f"beta_0 = {beta_0}")
    print(f"beta_1 = {beta_1}")

    plt.scatter(hours_worked, salary)

    y_pred = beta_0 + beta_1 * x

    plt.plot(x, y_pred, color="red")

    plt.xlabel("hours worked")
    plt.ylabel("salary")
    plt.title("Scatter Plot")

    plt.show()


