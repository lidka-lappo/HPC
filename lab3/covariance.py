import numpy as np
import time
from ctypes import CDLL, c_long, c_double
from numpy.ctypeslib import ndpointer

# Load the shared library

covariance = CDLL("./covariance.so")

# Define the function prototypes
covariance.connect.restype = None
covariance.connect.argtypes = None

covariance.get_var.restype = None
covariance.get_var.argtypes = [c_long, c_long, ndpointer(c_double)]

covariance.get_covariance.restype = None
covariance.get_covariance.argtypes = [c_long, c_long, ndpointer(c_double), ndpointer(c_double)]

# Read data from files
print("# READING DATA")
file_paths = [f"/home2/archive/mct/labs/lab1/var{i+1}.dat" for i in range(5)]
arrays = [np.fromfile(file_path, dtype=np.double) for file_path in file_paths]
arrays += [np.zeros_like(arrays[0]) for _ in range(5)]
arrays = np.stack(arrays, axis=0)

start_time = time.time()

# Call the functions from the shared library
covariance.connect()

get_var = covariance.get_var
get_var(arrays.shape[1], 5, arrays)

cov_matrix = np.zeros(shape=(10, 10), dtype=np.double)
covariance.compute_covariance(arrays.shape[1], 5, arrays, cov_matrix)

end_time = time.time()
computation_time = end_time - start_time
print(f"COMPUTATION TIME: {computation_time} s")
