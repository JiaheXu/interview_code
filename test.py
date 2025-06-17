import numpy as np

def find_stride_indices(positions, stride, threshold):

    current_dist = np.linalg.norm(positions[-1] - positions[-2])
    last_dist = 0
    i = positions.shape[0] - 2
    j = len(stride) -1
    result = []
    while True:
        if(i < 0 or j < 0):
            break

        while(current_dist < stride[j]):
            i -= 1
            last_dist = current_dist
            if( i< 0 ):
                break
            current_dist = current_dist + np.linalg.norm(positions[i] - positions[i+1])
        # print("current_dist: ", current_dist)
        # print("i: ", i)
        # print("j: ", j)
        left_diff = abs( current_dist -stride[j] )
        right_diff = abs( last_dist - stride[j] )
        if( min(left_diff, right_diff) > threshold):
            result.append(-1)
        elif(left_diff < right_diff):
            result.append(i)
        else:
            result.append(i+1)
        j -= 1
    result = result[::-1]
    print("solution1: ", result)      


def find_stride_indices_np(positions, stride, threshold):

    n = len(stride)
    delta = positions[:-1] - positions[1:]
    dist = np.linalg.norm( delta , axis = 1)
    dist = np.append( dist, 0)
    cum_sum = np.flip( np.cumsum( np.flip(dist) ) )
    # print("cum_sum: ", cum_sum)
    cum_sum = np.expand_dims(cum_sum, 0)

    stride = np.expand_dims(stride, 1)

    diff = np.abs( stride - cum_sum )
    idx = np.argmin( diff, axis = 1)
    min_value = diff[ range(n), idx]
    result = np.where(min_value <= threshold, idx, -1)
    # print("min_value: ", min_value)
    print("solution2: ", result)

positions = np.array([
    [0, 0],
    [10, 0],
    [20, 0],
    [30, 0],
    [45, 0],
    [47, 0]
])

# strides = [25, 10, 5]  # meters
# result = find_stride_indices(positions, strides, threshold=10.0)
# result = find_stride_indices_np(positions, strides, threshold=10.0)
# print(result)


# Sample 3D array: shape (A, B, C)
arr = np.random.rand(2, 4, 3)

# Get indices of min along axis 1
min_indices = np.argmin(arr, axis=1)  # shape (2, 3)

# Get actual min values using advanced indexing
A, C = arr.shape[0], arr.shape[2]
rows = np.arange(A)[:, None]       # shape (A, 1)
cols = np.arange(C)[None, :]       # shape (1, C)
min_values = arr[rows, min_indices, cols]  # shape (A, C)

# print("min_values: ", min_values)



'''
Autonomous car driving: car’s (x,y) position is given as Nx2 numpy array.
Car’s position = [[x0,y0], [x1,y1], … [xn,yn]]
Where [xn,yn] is last position
There is another variable strides S, which is length m vector. For example, S = [25, 10, 5].
This represents 25m, 10m, 5m.
For each stride in strides, (for each 25m, 10m, 5m for example), find the closest index of car’s position where it’s cumulative distance (from back) is closest to the stride.
The returned value should be length m numpy array.
If the closest index is over a threshold, return -1 instead of the index.
Solve this using numpy.
'''