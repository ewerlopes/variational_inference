import numpy as np

def read_csv(filename='oldFaith.txt'):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        line = line.strip().split(' ')
        data.append(tuple([float(line[0]),int(line[1])]))
    return data


def normalize_data(X):
    return (np.array(X) - np.mean(X)) / np.var(X)
