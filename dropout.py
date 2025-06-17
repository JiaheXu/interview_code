import numpy as np

class SpatialDropout:
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            # X shape: (N, C, H, W)
            # Apply same dropout mask per entire feature map per channel
            # So mask shape = (N, C, 1, 1), then broadcast
            self.mask = (np.random.rand(X.shape[0], X.shape[1], 1, 1) > self.drop_prob).astype(np.float32)
            out = (X * self.mask) / (1.0 - self.drop_prob)
        else:
            out = X
        return out

    def backward(self, d_out):
        if self.training:
            dX = (d_out * self.mask) / (1.0 - self.drop_prob)
        else:
            dX = d_out
        return dX
