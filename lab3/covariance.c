// covariance.c
//gcc -shared -o covariance.so -fopenmp -fPIC covariance.c 
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "mct_utils.h"

#define NELEMENTS 134217728
#define NVARS 5

// Function to compute auxiliary variables
void compute_auxiliary_variables(double* data, double* aux_vars, size_t nelements) {
    size_t k;
    for (k = 0; k < nelements; k++) {
        aux_vars[0 * nelements + k] = sin(data[0 * nelements + k]) + cos(data[1 * nelements + k]);
        aux_vars[1 * nelements + k] = exp(data[2 * nelements + k]) + exp(-1. * data[3 * nelements + k]);
        aux_vars[2 * nelements + k] = sin(data[0 * nelements + k]) * cos(data[1 * nelements + k]) + cos(data[2 * nelements + k]) * sin(data[3 * nelements + k]);
        aux_vars[3 * nelements + k] = hypot(data[1 * nelements + k], data[2 * nelements + k]);
        aux_vars[4 * nelements + k] = cbrt(data[4 * nelements + k]);
    }
}

// Function to compute covariance matrix
void covariance_compute(double* data, double* covariance, size_t nelements, size_t nvars) {
    size_t i, j, k;
    double avg[NVARS + 5] = {0};

    // Compute averages
    for (i = 0; i < nvars + 5; i++) {
        for (k = 0; k < nelements; k++)
            avg[i] += data[i * nelements + k];
        avg[i] /= nelements;

        // Subtract average from variables
        for (k = 0; k < nelements; k++)
            data[i * nelements + k] -= avg[i];
    }

    // Compute covariance matrix
    for (i = 0; i < nvars + 5; i++) {
        for (j = 0; j <= i; j++) {
            covariance[i * (nvars + 5) + j] = 0.0;
            for (k = 0; k < nelements; k++)
                covariance[i * (nvars + 5) + j] += data[i * nelements + k] * data[j * nelements + k];
            covariance[i * (nvars + 5) + j] /= (nelements - 1);
            covariance[j * (nvars + 5) + i] = covariance[i * (nvars + 5) + j]; // Covariance matrix is symmetric
        }
    }
}
