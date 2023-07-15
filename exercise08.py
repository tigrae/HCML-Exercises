import numpy as np

if __name__ == "__main__":

    case1 = [0.46, 0.64, 0.64, 0.75, 0.57, 0.79, 0.74, 0.70, 0.65, 0.66]
    case2 = [0.50, 0.366, 0.747, 0.01, 0.999, 0.414, 0.708, 0.834, 0.999, 1]

    mean_case1 = np.mean(case1)
    std_case1 = np.std(case1)

    mean_case2 = np.mean(case2)
    std_case2 = np.std(case2)

    # Print the results
    print("Case 1:")
    print("Mean:", mean_case1)
    print("Standard Deviation:", std_case1)
    print()

    print("Case 2:")
    print("Mean:", mean_case2)
    print("Standard Deviation:", std_case2)
