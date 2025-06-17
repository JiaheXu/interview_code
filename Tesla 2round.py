"""
This question is divided into 3 parts.
(1) Implement forward pass of cross attention in numpy.
(2) Implement backward pass of cross attention in numpy.
(3) Implement memory efficient forward pass of cross attention in numpy.
 
There are 4 functions to implement.
There are test cases that you can run after finishing each part.
 
Shapes:
Q: (N, D) 目标中文
K: (M, D) 源序列中每个元素的特征
V: (M, D) 源序列中每个元素的实际信息, Q K用于计算匹配度是表示信息, V 是被加权的实际信息

In the test case, such as a translation scenario (English -> 中文)

N = 100     目标序列 (中文句子) 需要生成100个词, query里叫N
M = 50      源序列(英文句子)有50个词, key叫M
D = 256     与K对应, 包含源序列中每个词的信息


Attention(Q, K, V) = A * V = softmax(Q * K_T / scaling) * V               (N, D) -> (N, M) -> (N, D)         A refer to 'Attention Weight'
scaling = np.sqrt(D)

为什么除scale:
1. 为了防止在高维情况下，查询和键的点积值累加次数多而过大，导致数值不稳定和梯度问题。
2. 标准化得分的方差，使其与维度无关，保持数值稳定性，确保模型的有效训练。
   e.g., 如果假设Q, K的元素是均值为 0、方差为 1 的独立随机变量, Var(Q * K) = D  Var(Q * K / np.sqrt(D)) = 1
"""

import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple
 
 
"""
Part 1: Implement forward pass of cross attention.
"""
 
npt_f64 = npt.NDArray[np.float64]
 
def softmax(s: npt_f64, dim: int) -> npt_f64:
    """
    Softmax forward
 
    Inputs:
    - s: A numpy array of shape (N, M)
    - dim: Dimension to do softmax over
 
    Returns:
    - output: A numpy array of shape (N, M)
    """

    s_exp = np.exp(s - np.max(s, axis=dim, keepdims=True))        # 和s_exp = np.exp(s)一样，减max就为了防止overflow，比如1000进入exp后太大了
    output = s_exp / np.sum(s_exp, axis=dim, keepdims=True)
    return output
 
 
def scaled_dot_product_attention_regular_forward(
    q: npt_f64,
    k: npt_f64,
    v: npt_f64,
) -> npt_f64:
    """
    Self-attention forward
 
    Inputs:
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M, D)
    - v: A numpy array of shape (M, D)
 
    Returns:
    - output: A numpy array of shape (N, D)
    """

    d_k = k.shape[1]
    scores = np.dot(q, k.T) / np.sqrt(d_k)        # shape: (N, M)
    attention_weights = softmax(scores, dim=1)    # shape: (N, M)
    output = np.dot(attention_weights, v)         # shape: (N, D)
    return output

"""
          键1   键2   键3
查询1    s11   s12   s13
查询2    s21   s22   s23
查询3    s31   s32   s33

dim = 1的原因是对于每个查询向量 (每一行), 计算它对所有键的Softmax (注意力权重)
"""
 
"""
Part 2: Implement backward pass of cross attention.
 
`softmax_backward` is implemented for you.
Note the input shape of `softmax_backward` is different from `softmax`.
"""
 
def softmax_backward(grad_softmax: npt_f64, o_softmax: npt_f64) -> npt_f64:
    """
    Inputs:
    - grad: A numpy array of shape (K,)
    - o: A numpy array of shape (K,)
 
    Returns:
    - output: A numpy array of shape (K,)
    """
    output = np.zeros_like(o_softmax)
    
    for i in range(o_softmax.shape[0]):
        softmax_i = o_softmax[i]
        grad_i = grad_softmax[i]
        # Compute the Jacobian matrix for the softmax
        jacobian = np.diag(softmax_i) - np.outer(softmax_i, softmax_i)
        # Apply the chain rule
        output[i] = jacobian @ grad_i
    return output
 
 
def scaled_dot_product_attention_regular_backward(
    grad: npt_f64,
    q: npt_f64,
    k: npt_f64,
    v: npt_f64,
) -> Tuple[npt_f64, npt_f64, npt_f64]:
    """
    Self-attention backward
 
    Inputs:
    - grad: A numpy array of shape (N, D)
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M, D)
    - v: A numpy array of shape (M, D)
 
    Returns:
    - dq, dk, dv
    """
 
    # *****Implement scaled_dot_product_attention_regular_backward*****
    d_k = k.shape[1]
    scores = np.dot(q, k.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, dim=1)

    grad_attention_weights = np.dot(grad, v.T)
    grad_scores = softmax_backward(grad_attention_weights, attention_weights)

    dq = np.dot(grad_scores, k) / np.sqrt(d_k)
    dk = np.dot(grad_scores.T, q) / np.sqrt(d_k)
    dv = np.dot(attention_weights.T, grad)
    return dq, dk, dv
 
 
"""
Part 3: Implement memory efficient version of cross attention.
 
To be memory efficient, we want to split K and V into S sections.
Assume that M is divisible by S.
Let's call these split tensors as (K_1, K_2, ..., K_S) and (V_1, V_2, ..., V_S).
To be memory efficient, we do not want to materialize any tensor of shape (N, M).
 
We recommend to start by copying over your implementations of softmax and attention from Part 1.
Hint: You need to save some intermediate variables in the attention / softmax.
"""
 
def scaled_dot_product_attention_efficient_forward(
    q: npt_f64,
    k: npt_f64,
    v: npt_f64,
    s: int
) -> npt_f64:
    """
    Memory Efficient Self-attention forward
    Steps:
    (A) Split k and v into s sections.
    (B) Perform attention on (q, k_i, v_i) individually in a for loop.
    (C) Combine the results of attention.
 
    Inputs:
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M, D)
    - v: A numpy array of shape (M, D)
    - s: An integer stating number of sections to split k and v into.
 
    Returns:
    - output: A numpy array of shape (N, D)
    """

    N, D = q.shape
    M, _ = k.shape
    assert M % s == 0
    section_size = M // s

    # Initialize variables
    output = np.zeros((N, D))
    m = np.full((N, 1), -np.inf)  # Per-row max scores for numerical stability

    # First pass: compute the global max scores
    for i in range(s):
        k_i = k[i * section_size: (i + 1) * section_size]
        d_k = k_i.shape[1]
        scores = np.dot(q, k_i.T) / np.sqrt(d_k)  # Shape: (N, section_size)
        m = np.maximum(m, np.max(scores, axis=1, keepdims=True))

    # Initialize sum of exponentials
    sum_s_denominator = np.zeros((N, 1))

    # Second pass: compute exponentials, accumulate sums and outputs
    for i in range(s):
        k_i = k[i * section_size: (i + 1) * section_size]
        v_i = v[i * section_size: (i + 1) * section_size]
        d_k = k_i.shape[1]
        scores = np.dot(q, k_i.T) / np.sqrt(d_k)  # Shape: (N, section_size)
        attention_weights = np.exp(scores - m)  # Shape: (N, section_size) # Adjust scores by the global max for numerical stability  # 可注释掉
        output += np.dot(attention_weights, v_i)  # Shape: (N, D)
        sum_s_denominator += np.sum(attention_weights, axis=1, keepdims=True)  # Shape: (N, 1)
        

    # Normalize the outputs by the accumulated sums
    output /= sum_s_denominator

    return output
 
 
"""
Tests for Part 1
"""
 
N = 100
M = 50
D = 256
 
Q = np.random.randn(N, D)
K = np.random.randn(M, D)
V = np.random.randn(M, D)
 
# test softmax
print("Part 1.1: softmax")
my_softmax = softmax(Q, dim=1)
pt_softmax = torch.softmax(torch.from_numpy(Q), dim=1)
print("Shape is correct:", pt_softmax.shape == my_softmax.shape)
print("Output is correct:", np.allclose(my_softmax, pt_softmax.detach().numpy()))
print("")
 
# test forward
my_self_attn = scaled_dot_product_attention_regular_forward(Q, K, V)
 
Q_torch = torch.nn.Parameter(torch.from_numpy(Q))
K_torch = torch.nn.Parameter(torch.from_numpy(K))
V_torch = torch.nn.Parameter(torch.from_numpy(V))
pt_self_attn = torch.nn.functional.scaled_dot_product_attention(
    Q_torch[None],
    K_torch[None],
    V_torch[None],
)
pt_self_attn_np = pt_self_attn[0].detach().numpy()
 
print("Part 1.2: self attention forward regular")
print("Shape is correct:", my_self_attn.shape == pt_self_attn_np.shape)
print("Output is correct: ", np.allclose(my_self_attn, pt_self_attn_np))
print("")
 
 
"""
Tests for Part 2
"""
 
# test backward
dq, dk, dv = scaled_dot_product_attention_regular_backward(np.ones_like(my_self_attn), Q, K, V)
pt_self_attn.backward(torch.ones_like(pt_self_attn))
 
print("Part 2: self attention backward regular")
print("dv matches:", np.allclose(dv, V_torch.grad.detach().numpy()))
print("dq matches:", np.allclose(dq, Q_torch.grad.detach().numpy()))
print("dk matches:", np.allclose(dk, K_torch.grad.detach().numpy()))
print("")
 
 
"""
Tests for Part 3
"""
 
my_self_attn_efficient = scaled_dot_product_attention_efficient_forward(Q, K, V, 10)
 
print("Part 3: self attention forward efficient")
print("Shape is correct:", my_self_attn.shape == my_self_attn_efficient.shape)
print("Output is correct:", np.allclose(my_self_attn, my_self_attn_efficient))