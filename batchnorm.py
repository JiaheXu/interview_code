import numpy as np

class BatchNorm:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.eps = eps
        self.training = True

        # Running estimates for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, X):
        if self.training:
            # Compute mean and variance from batch
            self.batch_mean = np.mean(X, axis=0, keepdims=True)
            self.batch_var = np.var(X, axis=0, keepdims=True)

            # Normalize
            self.X_centered = X - self.batch_mean
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
            self.X_norm = self.X_centered * self.std_inv

            # Scale and shift
            out = self.gamma * self.X_norm + self.beta

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running stats during inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * X_norm + self.beta

        return out

    def backward(self, d_out):
        N, D = d_out.shape

        # Gradients of beta and gamma
        d_beta = np.sum(d_out, axis=0, keepdims=True)
        d_gamma = np.sum(d_out * self.X_norm, axis=0, keepdims=True)

        # Gradients of the normalized input
        d_X_norm = d_out * self.gamma

        # Gradients w.r.t variance
        d_var = np.sum(d_X_norm * self.X_centered, axis=0, keepdims=True) * -0.5 * (self.std_inv ** 3)

        # Gradients w.r.t mean
        d_mean = np.sum(d_X_norm * -self.std_inv, axis=0, keepdims=True) + \
            d_var * np.mean(-2.0 * self.X_centered, axis=0, keepdims=True)

        # Gradient w.r.t input
        d_X = (d_X_norm * self.std_inv) + (d_var * 2 * self.X_centered / N) + (d_mean / N)

        return d_X, d_gamma, d_beta
