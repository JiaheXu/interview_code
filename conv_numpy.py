import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        scale = np.sqrt(1. / (in_channels * kernel_size * kernel_size))
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        C_in, H, W = x.shape
        K = self.kernel_size
        P = self.padding

        x_padded = np.pad(x, ((0, 0), (P, P), (P, P)), mode='constant')
        self.x_padded = x_padded

        H_out = H + 2 * P - K + 1
        W_out = W + 2 * P - K + 1

        out = np.zeros((self.out_channels, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                region = x_padded[:, i:i+K, j:j+K]  # shape: (C_in, K, K)
                out[:, i, j] = np.tensordot(self.kernels, region, axes=([1, 2, 3], [0, 1, 2])) + self.bias

        return out

    def backward(self, d_out, lr=1e-3):
        C_out, H_out, W_out = d_out.shape
        C_in, H, W = self.x.shape
        K = self.kernel_size
        P = self.padding

        d_kernels = np.zeros_like(self.kernels)
        d_bias = np.zeros_like(self.bias)
        d_x_padded = np.zeros_like(self.x_padded)

        for i in range(H_out):
            for j in range(W_out):
                region = self.x_padded[:, i:i+K, j:j+K]  # shape: (C_in, K, K)
                # d_out[:, i, j] has shape (C_out,)
                # Use broadcasting to update all filters in one go:
                d_kernels += d_out[:, i, j][:, None, None, None] * region  # broadcasting
                d_bias += d_out[:, i, j]
                d_x_padded[:, i:i+K, j:j+K] += np.sum(
                    d_out[:, i, j][:, None, None, None] * self.kernels, axis=0
                )

        self.kernels -= lr * d_kernels
        self.bias -= lr * d_bias

        if P > 0:
            d_x = d_x_padded[:, P:-P, P:-P]
        else:
            d_x = d_x_padded

        return d_x

x = np.random.randn(3, 5, 5)  # input
conv = Conv2D(3, 2, kernel_size=3, padding=1)

# forward
out = conv.forward(x)

# fake gradient from next layer
d_out = np.random.randn(*out.shape)

# backprop
d_x = conv.backward(d_out, lr=0.01)

print("Output shape:", out.shape)     # (2, 5, 5)
print("Grad w.r.t input:", d_x.shape) # (3, 5, 5)
