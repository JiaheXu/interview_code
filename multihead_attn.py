import numpy as np

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        limit = np.sqrt(1 / embed_dim)
        self.W_q = np.random.uniform(-limit, limit, (embed_dim, embed_dim))
        self.W_k = np.random.uniform(-limit, limit, (embed_dim, embed_dim))
        self.W_v = np.random.uniform(-limit, limit, (embed_dim, embed_dim))
        self.W_o = np.random.uniform(-limit, limit, (embed_dim, embed_dim))

        self.b_q = np.zeros((embed_dim,))
        self.b_k = np.zeros((embed_dim,))
        self.b_v = np.zeros((embed_dim,))
        self.b_o = np.zeros((embed_dim,))

        self.cache = {}

    def split_heads(self, x):
        B, T, E = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (B, H, T, D)

    def combine_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def softmax_backward(self, d_out, softmax_out):
        dot = np.sum(d_out * softmax_out, axis=-1, keepdims=True)
        return softmax_out * (d_out - dot)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        weights = self.softmax(scores)
        output = weights @ V

        self.cache.update({
            'Q': Q, 'K': K, 'V': V, 'attn_weights': weights
        })
        return output, weights

    def forward(self, x, mask=None):
        Q_proj = x @ self.W_q + self.b_q
        K_proj = x @ self.W_k + self.b_k
        V_proj = x @ self.W_v + self.b_v

        Q = self.split_heads(Q_proj)
        K = self.split_heads(K_proj)
        V = self.split_heads(V_proj)

        attn_out, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        concat = self.combine_heads(attn_out)
        out = concat @ self.W_o + self.b_o

        self.cache.update({
            'x': x, 'Q_proj': Q_proj, 'K_proj': K_proj, 'V_proj': V_proj,
            'concat': concat
        })
        return out, attn_weights

    def backward(self, dout, learning_rate=0.01):
        x = self.cache['x']
        Q_proj = self.cache['Q_proj']
        K_proj = self.cache['K_proj']
        V_proj = self.cache['V_proj']
        Q, K, V = self.cache['Q'], self.cache['K'], self.cache['V']
        attn_weights = self.cache['attn_weights']
        concat = self.cache['concat']

        # Grad w.r.t. output linear layer
        dW_o = concat.reshape(-1, self.embed_dim).T @ dout.reshape(-1, self.embed_dim)
        db_o = np.sum(dout, axis=(0, 1))

        d_concat = dout @ self.W_o.T
        d_attn_out = self.split_heads(d_concat)

        # Grad w.r.t. attention weights and V
        d_weights = d_attn_out @ V.transpose(0, 1, 3, 2)
        dV = attn_weights.transpose(0, 1, 3, 2) @ d_attn_out

        # Grad w.r.t. scores
        d_scores = self.softmax_backward(d_weights, attn_weights)

        dQ = d_scores @ K / np.sqrt(self.head_dim)
        dK = d_scores.transpose(0, 1, 3, 2) @ Q / np.sqrt(self.head_dim)

        # Merge heads
        dQ_proj = self.combine_heads(dQ)
        dK_proj = self.combine_heads(dK)
        dV_proj = self.combine_heads(dV)

        # Grad w.r.t. QKV linear projections
        dW_q = x.reshape(-1, self.embed_dim).T @ dQ_proj.reshape(-1, self.embed_dim)
        db_q = np.sum(dQ_proj, axis=(0, 1))

        dW_k = x.reshape(-1, self.embed_dim).T @ dK_proj.reshape(-1, self.embed_dim)
        db_k = np.sum(dK_proj, axis=(0, 1))

        dW_v = x.reshape(-1, self.embed_dim).T @ dV_proj.reshape(-1, self.embed_dim)
        db_v = np.sum(dV_proj, axis=(0, 1))

        # Grad w.r.t. input x
        dx_q = dQ_proj @ self.W_q.T
        dx_k = dK_proj @ self.W_k.T
        dx_v = dV_proj @ self.W_v.T
        dx = dx_q + dx_k + dx_v

        # Update weights
        self.W_o -= learning_rate * dW_o
        self.b_o -= learning_rate * db_o

        self.W_q -= learning_rate * dW_q
        self.b_q -= learning_rate * db_q

        self.W_k -= learning_rate * dW_k
        self.b_k -= learning_rate * db_k

        self.W_v -= learning_rate * dW_v
        self.b_v -= learning_rate * db_v

        return dx

np.random.seed(42)
B, T, E, H = 2, 5, 16, 4

x = np.random.randn(B, T, E)
mha = MultiHeadAttention(embed_dim=E, num_heads=H)

out, attn_weights = mha.forward(x)
print("Forward output shape:", out.shape)

# Dummy gradient
dout = np.random.randn(*out.shape)
dx = mha.backward(dout)

print("Backward dx shape:", dx.shape)  # Should match input (B, T, E)
