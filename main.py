import subprocess
import numpy as np
n = 5
matrix = np.random.randint(2, 30, size = (n, n + 1)).astype(np.float64)

# matrix = np.array([
#     [24., 17.,  9.,  3.,  2., 28.],
#     [13.,  3., 29.,  9.,  5., 15.],
#     [21., 16., 24., 22., 20.,  3.],
#     [18., 18., 10., 25.,  4., 19.],
#     [20., 14.,  9.,  4.,  3., 29.]]).astype(np.float64)

# print(matrix[2, :])
print(f'Shape: {matrix.shape}')
filename = 'matrix_with_b.npy'

np.save(filename, matrix)






p = 3
command = 'mpiexec -n ' + str(p) + ' python3 gauss_par.py'
subprocess.run(command, shell=True)

