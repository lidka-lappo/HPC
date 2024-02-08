import ctypes
import os
import numpy as np
from time import time

# Load the C library
lib = ctypes.CDLL(os.path.abspath("covariance.so"))

# Define the C function prototypes
covariance_compute = lib.covariance_compute
covariance_compute.restype = None
covariance_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

# Define constants
NELEMENTS = 134217728
NVARS = 5

def compute_covariance(data):
    """
    Compute covariance matrix of the given data.

    Parameters:
    - data: numpy array of shape (NVARS, NELEMENTS) containing the data.

    Returns:
    - covariance matrix: numpy array of shape (NVARS+5, NVARS+5) containing the covariance matrix.
    """
    t_start = time()

    # Compute auxiliary variables
    aux_vars = np.empty((NVARS+5, NELEMENTS), dtype=np.double)
    lib.compute_auxiliary_variables(data.ctypes.data_as(ctypes.c_void_p), aux_vars.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(NELEMENTS))

    # Compute averages and subtract them from variables
    avg = np.mean(data, axis=1)
    data -= avg[:, np.newaxis]

    # Compute covariance matrix
    covariance = np.empty((NVARS+5, NVARS+5), dtype=np.double)
    covariance_compute(data.ctypes.data_as(ctypes.c_void_p), covariance.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(NELEMENTS), ctypes.c_size_t(NVARS))

    t_end = time()
    print("Computation time:", t_end - t_start, "sec")

    return covariance

# Load data
data = np.empty((NVARS, NELEMENTS), dtype=np.double)
for i in range(NVARS):
    file_name = "/home2/archive/mct/labs/lab1/var{}.dat".format(i+1)
    with open(file_name, "rb") as f:
        data[i] = np.fromfile(f, dtype=np.double)

# Compute covariance matrix
covariance_matrix = compute_covariance(data)

# Print results
for i in range(NVARS+5):
    for j in range(i+1):
        print("cov({},{}) = {}".format(i+1, j+1, covariance_matrix[i,j]))
