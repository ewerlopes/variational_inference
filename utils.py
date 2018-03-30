from __future__ import division
from matplotlib import pyplot as plt
import numpy as np


def read_csv(filename='faithful.txt'):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        line = line.strip().split(' ')
        data.append(tuple([float(line[0]), float(line[1])]))
    return data


def normalize_data(X):
    return (np.array(X) - np.mean(X)) / np.std(X, ddof=1)


def plotFn(X, alpha, m, W, v, loglik, iter_num):
    raise NotImplementedError("To be implemented")


def plot_data(X, Y, normalize=False):
    fig = plt.figure(figsize=(4, 2))

    if normalize:
        X = normalize_data(X)
        Y = normalize_data(Y)

    plt.scatter(X, Y)
    plt.show()
