import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
class BatchedGaussianMixture:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    def initialize_parameters(self, X):
        batch_size, n_samples, n_features = X.shape
        self.means = np.zeros((batch_size, self.n_components, n_features))
        self.covariances = np.zeros((batch_size, self.n_components, n_features, n_features))
        self.weights = np.ones((batch_size, self.n_components)) / self.n_components
        for b in range(batch_size):
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.means[b] = X[b, indices]
            for k in range(self.n_components):
                self.covariances[b, k] = np.cov(X[b].T) + 1e-6 * np.eye(n_features)

    def e_step(self, X):
        batch_size, n_samples, n_features = X.shape
        self.resp = np.zeros((batch_size, n_samples, self.n_components))

        for b in range(batch_size):
            for k in range(self.n_components):
                self.resp[b, :, k] = self.weights[b, k] * multivariate_normal.pdf(
                    X[b], mean=self.means[b, k], cov=self.covariances[b, k])
            self.resp[b] /= self.resp[b].sum(axis=1, keepdims=True)
    def m_step(self, X):
        batch_size, n_samples, n_features = X.shape
        Nk = self.resp.sum(axis=1)  # (batch_size, n_components)

        self.weights = Nk / n_samples
        self.means = np.einsum('bni,bnf->bnf', self.resp, X) / Nk[..., np.newaxis]

        for b in range(batch_size):
            for k in range(self.n_components):
                diff = X[b] - self.means[b, k]
                weighted = self.resp[b, :, k][:, np.newaxis] * diff
                self.covariances[b, k] = (weighted.T @ diff) / Nk[b, k]
                self.covariances[b, k] += 1e-6 * np.eye(n_features)  # regularization

    def compute_log_likelihood(self, X):
        batch_size, n_samples, _ = X.shape
        log_likelihood = np.zeros(batch_size)

        for b in range(batch_size):
            tmp = np.zeros((n_samples,))
            for k in range(self.n_components):
                tmp += self.weights[b, k] * multivariate_normal.pdf(
                    X[b], mean=self.means[b, k], cov=self.covariances[b, k])
            log_likelihood[b] = np.sum(np.log(tmp))
        return log_likelihood

    def fit(self, X):
        self.initialize_parameters(X)
        prev_log_likelihood = None

        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            log_likelihood = self.compute_log_likelihood(X)

            if prev_log_likelihood is not None and np.all(np.abs(log_likelihood - prev_log_likelihood) < self.tol):
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X):
        self.e_step(X)
        return np.argmax(self.resp, axis=2)  # (batch_size, n_samples)

    def predict_proba(self, X):
        self.e_step(X)
        return self.resp

# Generate batched data: 5 datasets of shape (300, 2)
batch_size = 5
X_batch = np.stack([make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=i)[0] for i in range(batch_size)])
gmm = BatchedGaussianMixture(n_components=3)
gmm.fit(X_batch)
y_pred = gmm.predict(X_batch)
print("Cluster assignments for batch 0:", y_pred[0])
