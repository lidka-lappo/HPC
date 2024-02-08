#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "mct_utils.h"

// use C interface 
#include <cblas.h>
#include <lapacke.h> // Include LAPACKE header

#define N 400

/**
 * Function returns value of matrix element A_{ij}
 * i,j iterates from 1 to M as in standard mathematical notation
 */
double matrix_H(int k, int l)
{
    #define omega 0.001
    #define a 1.0
    #define hbar 1.0
    #define m 1.0
    
    double K, V;
    if(k == l)
    {
        K = pow(hbar * M_PI / a, 2) / (6. * m) * (1.0 + 2.0 / (N * N));
        V = 0.5 * m * pow(omega * a * (k - N / 2), 2);
    }
    else
    {
        K = pow(hbar * M_PI / a, 2) / (1. * m * N * N) * pow(-1, k - l) / pow(sin(M_PI * (k - l) / N), 2);
        V = 0.0;
    }
    
    return K + V;
}

#define IDX1(i,j) ((j-1) * N + (i-1))

int main()
{
    double *H; // hamiltonian matrix
    H = (double *)malloc(N * N * sizeof(double)); // Allocate memory using malloc

    double *E; // buffer for eigen values
    E = (double *)malloc(N * sizeof(double)); // Allocate memory using malloc
        
    int i, j;
    double rt;
    
    // set matrix
    for(i = 1; i <= N; i++) {
        for(j = 1; j <= N; j++) {
            H[IDX1(i, j)] = matrix_H(i, j);
        }
    }
    
    // Compute eigen values and eigen vectors
    b_t();
    
    // Compute eigenvalues using LAPACK
    lapack_int info;
    info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'U', N, H, N, E); // 'N' means eigenvalues only, 'U' means upper triangular part is stored
    
    rt = e_t();
    printf("Computation time: %f sec\n", rt);

    if (info == 0) {
        // Eigenvalues are stored in array E
        printf("Eigenvalues:\n");
        for (i = 0; i < N; i++) {
            printf("%d: %f\n", i + 1, E[i]);
        }
    } else {
        printf("Eigenvalue computation failed with error code: %d\n", info);
    }

    // Free allocated memory
    free(H);
    free(E);
    
    return 0;
}
