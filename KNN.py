import numpy as np
from math import sqrt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


class KNN:
    Train_features = list()
    Test_features = list()
    target = list()
    k = 0

    def __init__(self, x, y, k=3):
        self.Train_features = x
        self.k = k
        self.target = y
        self.joining_features_and_targets()
        self.Train_features = list(self.Train_features)
        # self.Train_features=self.calc_avg_classes()

    def joining_features_and_targets(self):
        self.Train_features = np.array(self.Train_features)
        self.target = np.array(self.target)
        self.target = np.reshape(self.target, (len(self.target), 1))
        self.Train_features = np.append(self.Train_features, self.target, axis=1)

    # Locate the most similar neighbors
    def calc_avg_classes(self):
        vec = [0] * 10
        vec1 = [[0 for j in range(32)] for i in range(10)]
        for i in self.Train_features:
            vec[int(i[-1])] += 1
        for i in self.Train_features:
            for j in range(32):
                vec1[int(i[-1])][j] += i[j]
        for i in range(10):
            for j in range(32):
                vec1[i][j] /= vec[i]
        for i in range(10):
            vec1[i].append(i)
        return vec1

    def get_neighbors(self, test_row, num_neighbors):
        distances = list()
        for train_row in self.Train_features:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        # neighbors.append(distances[0][0])
        return neighbors

    def predict(self, test):
        predictions = list()
        for test_row in test:
            neighbors = self.get_neighbors(test_row, self.k)
            output_values = [row[-1] for row in neighbors]
            prediction = max(set(output_values), key=output_values.count)
            predictions.append(prediction)
        return predictions
