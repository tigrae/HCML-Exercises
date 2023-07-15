import numpy as np
from scipy.signal import convolve2d

# Define the input matrix A and filter B
A = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

# add one layer of zero padding
A = np.pad(A, pad_width=1, mode='constant', constant_values=0)

print(A)
print()

B = np.array([[1, 0, 0],
              [0, 0, 1],
              [1, 1, 0]])

# Perform convolution with stride 2 and padding 0
output = convolve2d(A, B, mode='valid')

# apply stride by slicing the output matrix
output = output[::2, ::2]

# Print the result
print(output)
