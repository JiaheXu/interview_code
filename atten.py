import numpy as np

class ScaledDotProductAttention:
    def __init__(self):
        self.cache = {}

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def softmax_backward(self, grad_output, softmax_output):
        # Applies Jacobian of softmax to grad_output
        B, H, T, _ = grad_output.shape
        dx = np.empty_like(grad_output)

        for b in range(B):
            for h in range(H):
                for i in range(T):
                    s = softmax_output[b, h, i].reshape(-1, 1)  # (T, 1)
                    jacobian = np.diagflat(s) - s @ s.T         # (T, T)
                    dx[b, h, i] = jacobian @ grad_output[b, h, i]
        return dx

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        weights = self.softmax(scores)
        output = weights @ V

        self.cache = {
            'Q': Q, 'K': K, 'V': V, 'weights': weights,
            'd_k': d_k
        }
        return output, weights

    def backward(self, dout):
        Q, K, V = self.cache['Q'], self.cache['K'], self.cache['V']
        weights = self.cache['weights']
        d_k = self.cache['d_k']

        # Step 1: dV
        dV = weights.transpose(0, 1, 3, 2) @ dout  # (B, H, T, d_v)

        # Step 2: d(weights)
        d_weights = dout @ V.transpose(0, 1, 3, 2)  # (B, H, T, T)

        # Step 3: d(scores) via softmax backward
        d_scores = self.softmax_backward(d_weights, weights)  # (B, H, T, T)

        # Step 4: dQ
        dQ = d_scores @ K / np.sqrt(d_k)

        # Step 5: dK
        dK = d_scores.transpose(0, 1, 3, 2) @ Q / np.sqrt(d_k)

        return dQ, dK, dV

np.random.seed(0)

B, H, T, d_k, d_v = 2, 2, 4, 8, 8
Q = np.random.randn(B, H, T, d_k)
K = np.random.randn(B, H, T, d_k)
V = np.random.randn(B, H, T, d_v)

attn = ScaledDotProductAttention()
out, weights = attn.forward(Q, K, V)

# Dummy gradient coming from next layer
dout = np.random.randn(*out.shape)

dQ, dK, dV = attn.backward(dout)
print("dQ shape:", dQ.shape)
print("dK shape:", dK.shape)
print("dV shape:", dV.shape)
