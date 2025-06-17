import numpy as np
from collections import Counter

class kmeans:
    def __init__(self, k = 2, max_iter = 100, esp = 1e-4):
        self.k = k
        self.max_iter = max_iter
        self.esp = esp
    def fit(self, train_set):
        self.train_set = np.array( train_set )
        n, d = self.train_set.shape
        random_idx = np.random.choice( n, self.k, replace=False )
        self.centroids = self.train_set[random_idx]
        test = self.train_set[np.array([1,2,3])]
        print("test: ", test)
        for _ in range(self.max_iter):
            centroids_np = np.array( self.centroids ) # (k,2)
            dist = np.linalg.norm( np.expand_dims(self.train_set, 1) - np.expand_dims(self.centroids, 0), axis = 2)
            min_idx = np.argmin(dist, axis = 1)
            print("min_idx: ", min_idx)
            new_centroids = np.array([
                self.train_set[min_idx == label].mean(axis = 0) if np.any(min_idx == label) else self.centroids[label] 
                for label in range(self.k)
            ])
            if np.all( np.linalg.norm(self.centroids - new_centroids, axis = 1) < self.esp ):
                break
            self.centroids = new_centroids



# Example data
X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

model = kmeans(k=2)
model.fit(X)

print("Centroids:\n", model.centroids)
print("Labels:", model.labels)

# Predicting new points
test = np.array([[0, 0], [12, 3]])
# print("Predictions:", model.predict(test))



def k_fold_cv(model_class, X, y, k=5, **model_kwargs):
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices into k folds
    folds = np.array_split(indices, k)
    scores = []

    for i in range(k):
        # Create train and validation sets
        val_idx = folds[i]
        train_idx = np.hstack(folds[:i] + folds[i+1:])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Initialize and train model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        score = model.score(X_val, y_val)
        scores.append(score)
        
    return np.mean(scores), scores