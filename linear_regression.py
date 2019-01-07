import sys
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearRegression:

    def __init__(self, data, labels, iteration=100, outfile='output2.csv'):
        self.n = data.shape[0]
        self.data = self.normalize(data)
        self.labels = labels
        self.beta = np.zeros(3)
        self.iteration = iteration
        self.outfile = outfile

    def gradient_descent(self):
        learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.7]
        self.outfile = open(self.outfile, 'w+')
        for alpha in learning_rate:
            self.beta = np.zeros(3)
            for iter in range(self.iteration):
                value = np.dot(self.data, self.beta)
                self.beta = self.beta - alpha * \
                    (1 / self.n) * np.dot(np.transpose(self.data), (value - self.labels))
				#self.visualize()

            self.save_to_file(alpha)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        stdev = np.std(data, axis=0)
        ndata = (data - mean) / stdev
        intercept = np.ones((self.n, 1))
        ndata = np.hstack((intercept, ndata))
        return ndata

    def visualize(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.data[:, 1], self.data[:, 2], self.labels)
        ax.set_xlabel('Age(Years)')
        ax.set_ylabel('Weight(Kilograms)')
        ax.set_zlabel('Height(Meters)')
        plt.show()

    def save_to_file(self, alpha):
        self.outfile.write((str(alpha) +
                            ',' +
                            str(self.iteration) +
                            ',' +
                            str(self.beta[0]) +
                            ',' +
                            str(self.beta[1]) +
                            ',' +
                            str(self.beta[2]) +
                            '\n'))


if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    df = pd.read_csv(in_file, names=['age', 'weight', 'height'])

    data = df.iloc[:, :2].as_matrix()
    y = df['height'].values

    lr = LinearRegression(data, y, 100, out_file)
    lr.gradient_descent()
