import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, train_set, label):
        self.train_set = np.array(train_set)
        self.label = np.array(label)
    def predict(self, test_set):
        # test_set: (n,d)
        # train_set: (m,d)
        dist = np.expand_dims(test_set, 1) - np.expand_dims(self.train_set, 0)
        dist = np.linalg.norm(dist, ord=2,axis=2)
        min_idx = np.argsort(dist, axis = 1)
        k_nbs = min_idx[:, :self.k]
        k_nbs_label = self.label[ k_nbs ]
        print("k_nbs_label: ", k_nbs_label)
        for row in k_nbs_label:
            print("result: ", Counter(row).most_common(1)[0][0])

        predictions = np.array([
            Counter(row).most_common(1)[0][0]
            for row in k_nbs_label
        ])

        return predictions

    def score(self, test_set, gt_label):
        result = self.predict(test_set)
        gt_label = np.array( gt_label )
        return np.mean(result == gt_label)
        # print("prediction: ", predictions)

X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
y_train = [0, 0, 0, 1, 1, 1]

X_test = [[2, 2], [7, 6]]
y_test = [0, 1]

model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy)

np.argpartition
np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
array([1, 3, 1, 1, 0, 0, 0, 1])