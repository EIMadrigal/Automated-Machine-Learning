import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data_MNTest(fl="./MCTestData.csv"):
    """
    Loads data stored in McNemarTest.csv
    :param fl: filename of csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(fl, header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest(fl="./TMStTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest(fl="./FTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: evaluations
    """
    errors = np.loadtxt(fl, delimiter=",")
    return errors


def McNemar_test(labels, prediction_1, prediction_2):
    """
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    A, B, C, D = 0, 0, 0, 0
    for i in range(len(labels)):
        if prediction_1[i] == prediction_2[i]:
            if labels[i] == prediction_1[i]:
                A += 1
            else:
                D += 1
        else:
            if prediction_1[i] == labels[i]:
                B += 1
            else:
                C += 1

    chi2_Mc = np.square(B - C - 1) / (B + C)
    return chi2_Mc


def TwoMatchedSamplest_test(y1, y2):
    """
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    d = y1 - y2
    d_bar = np.mean(d)
    d_sigma = np.std(d, ddof=1)
    t_value = np.sqrt(len(y1)) * d_bar / d_sigma
    return t_value


def Friedman_test(errors):
    """
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, you can add anything needed to facilitate
    solving the following post hoc problems
    """
    FData_stats = {'errors': errors}

    ranks = np.zeros(errors.shape)
    for i in range(errors.shape[0]):
        idx = np.argsort(errors[i])
        for j in range(errors.shape[1]):
            ranks[i][idx[j]] = j + 1

    n, k = errors.shape[0], errors.shape[1]
    FData_stats['n'], FData_stats['k'] = n, k
    r_j = np.mean(ranks, axis=0)
    FData_stats['r'] = r_j
    FData_stats['best'], FData_stats['worst'] = r_j.argmin(), r_j.argmax()
    chi2_F = 12.0 * n * (np.sum(r_j ** 2) - k * (k + 1) ** 2 / 4.0) / (k * (k + 1))
    return chi2_F, FData_stats


def Nemenyi_test(FData_stats):
    """
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    n, k = FData_stats['n'], FData_stats['k']
    Q_value = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            Q_value[i][j] = (FData_stats['r'][i] - FData_stats['r'][j]) / np.sqrt(k * (k + 1) / (6.0 * n))
    return Q_value


def box_plot(FData_stats):
    """
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    plt.title('Algorithms Error')
    labels = 'best', 'worst'
    plt.boxplot([FData_stats['errors'][:, FData_stats['best']], FData_stats['errors'][:, FData_stats['worst']]],
                labels=labels)
    plt.legend()
    plt.show()


def main(args):
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)

    # (d)
    Q_value = Nemenyi_test(FData_stats)

    # (e)
    box_plot(FData_stats)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(args)
