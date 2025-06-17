import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# Concatenate along axis 0 (rows)
concatenated = np.concatenate((a, b), axis=0)
print(concatenated.shape)



a = np.array([[1, 2]])
b = np.array([[5, 6]])
concatenated = np.stack((a, b),axis =1)
print(concatenated.shape)

# Output:
# [[1 2]
#  [3 4]
#  [5 6]]

# Concatenate along axis 1 (columns)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
concatenated_axis1 = np.concatenate((a, b), axis=1)
print(concatenated_axis1.shape)
# Output:
# [[1 2 5]
#  [3 4 6]]
