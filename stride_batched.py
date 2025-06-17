import numpy as np
from collections import Counter

def batched_stride_indices_with_batched_strides(positions, strides, threshold):

    B, N, D = positions.shape
    delta = positions[:, :-1, :] - positions[:,1:,:]
    dist = np.linalg.norm( delta, axis = 2)
    # print("dist: ", dist.shape)
    dist = np.concatenate( [dist, np.zeros( (B , 1) ) ], axis = 1)
    # print("dist: ", dist)
    cum_sum = np.flip( np.cumsum( np.flip(dist, axis = 1), axis = 1 ), axis = 1) # (B, N)
    # strides: (B,M)
    # cum_sum: (B,N)
    # diff: B,N,M
    diff = np.abs( np.expand_dims(strides, axis = 2) - np.expand_dims(cum_sum, axis = 1) )
    min_idx = np.argmin( diff, axis = 2)
    print("min_idx: ", min_idx.shape)
    min_val = np.take_along_axis(diff, min_idx[:,:,None], axis =2).squeeze(-1)

    min_idx[( min_val > threshold)] = -1

    return min_idx
    

positions = np.array([
    [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0]],   # Car 1
    [[0, 0], [0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60]],   # Car 2
])

strides = np.array([
    [25, 10, 5],   # Car 1's strides
    [50, 15, 5]    # Car 2's strides
])

threshold = 2.0

result = batched_stride_indices_with_batched_strides(positions, strides, threshold)
print(result)
