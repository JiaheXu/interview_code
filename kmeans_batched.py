import numpy as np
def batched_kmeans_vectorized(X, k, num_iters=10):
    """
    Parameters:
        X: np.ndarray of shape (B, N, F)
        k: int, number of clusters
        num_iters: int, number of iterations

    Returns:
        centroids: np.ndarray of shape (B, K, F)
        labels: np.ndarray of shape (B, N)
    """
    B, N, F = X.shape

    # Initialize centroids using random indices
    random_indices = np.random.randint(0, N, size=(B, k))
    centroids = np.take_along_axis(
        X, random_indices[:, :, np.newaxis].repeat(F, axis=2), axis=1
    )  # (B, K, F)

    for _ in range(num_iters):
        # Compute distances: (B, N, K)
        diff = X[:, :, np.newaxis, :] - centroids[:, np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        # Assign labels: (B, N)
        labels = np.argmin(distances, axis=-1)
        # One-hot encode the labels: (B, N, K)
        one_hot = np.eye(k)[labels]  # (B, N, K)
        # Sum points per cluster: (B, K, F)
        numerator = np.einsum('bnk,bnf->bkf', one_hot, X)

        # Count of points per cluster: (B, K, 1)
        counts = one_hot.sum(axis=1, keepdims=True).transpose(0, 2, 1)
        counts = np.maximum(counts, 1e-8)  # Prevent division by zero
        # Update centroids: (B, K, F)
        centroids = numerator / counts

    return centroids, labels
X = np.random.randn(5, 200, 3)  # (batch=5, points=200, features=3)
centroids, labels = batched_kmeans_vectorized(X, k=6, num_iters=15)

print("Centroids:", centroids.shape)  # (5, 6, 3)
print("Labels:", labels.shape)        # (5, 200)
