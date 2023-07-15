import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def computeEER(tar, far):
    """ Compute the equal error rate
    """
    values = np.abs(1 - tar - far)
    idx = np.argmin(values)
    eer = far[idx]
    return eer


if __name__ == "__main__":

    scores_gen = np.load("exercise07_data/scores_gen_adience.npy")
    scores_imp = np.load("exercise07_data/scores_imp_adience.npy")

    num_gen, num_imp = len(scores_gen), len(scores_imp)
    print(num_gen, num_imp)

    scores = np.hstack((scores_gen, scores_imp))
    labels = np.hstack((np.ones(num_gen), np.zeros(num_imp)))

    fpr, tpr, thresholds = roc_curve(labels, scores)

    plt.plot(fpr, tpr)
    plt.xlabel('FMR')
    plt.ylabel('1-FNMR')
    plt.xscale('log')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    eer = computeEER(tpr, fpr)

    desired_fmr = 0.01
    desired_threshold_index = np.where(fpr <= desired_fmr)[0][-1]
    desired_threshold = thresholds[desired_threshold_index]
    fnmr = 1 - tpr[desired_threshold_index]



    plt.figure()
    plt.hist(scores_gen, label="Genuine", bins=100, alpha=0.5)
    plt.hist(scores_imp, label="Imposter", bins=100, alpha=0.5)
    plt.xlabel("Comparison score")

    plt.axvline(x=eer, color='r', linestyle='--', label=f'EER = {eer}')
    plt.axvline(x=fnmr, color='b', linestyle='--', label=f'FNMR @ 0.01 FMR = {fnmr}')

    plt.legend()
    plt.show()


