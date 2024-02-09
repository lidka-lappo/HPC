#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "mct_utils.h"

#include <cblas.h>


double matrix_H(int k, int l)
{
    #define omega 0.001
    #define a 1.0
    #define hbar 1.0
    #define m 1.0
    
    double K, V;
    if(k==l)
    {
        K=pow(hbar*M_PI/a,2)/(6.*m)*(1.0 + 2.0/(N*N));
        V=0.5*m*pow(omega*a*(k-N/2),2);
    }
    else
    {
        K=pow(hbar*M_PI/a,2)/(1.*m*N*N) * pow(-1,k-l) / pow(sin(M_PI*(k-l)/N),2);
        V=0.0;
    }
    
    return K+V;
    
    #undef a
    #undef m
}

#define IDX1(i,j) (j-1)*N + (i-1)

int main()
{
    
    double *H; // hamiltonian matrix
    H = (double*)malloc(N*N*sizeof(double));
    if (H == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    double *E; // buffer for eigen values
    E = (double*)malloc(N*sizeof(double));
    if (E == NULL) {
        printf("Memory allocation failed.\n");
        free(H);
        return -1;
    }
        
    int i,j;
    double rt;
    
    // set matrix
    for(i=1; i<=N; i++) for(j=1; j<=N; j++) H[IDX1(i,j)]=matrix_H(i,j);
    
    // Compute eigen values and eigen vectors
    b_t();
    
    // Compute eigenvalues using LAPACK dsyevd routine
    char jobz = 'N'; // Compute eigenvalues only
    char uplo = 'L'; // Lower triangular part of matrix is stored
    int lda = N;
    int info;
    
    dsyevd(&jobz, &uplo, &N, H, &lda, E, &info);

    if (info != 0) {
        printf("Error: dsyevd failed with error code %d\n", info);
        free(H);
        free(E);
        return -1;
    }

    rt = e_t();
    printf("Computation time: %fsec\n", rt);

    // Print first 10 eigenvalues
    printf("# n E_n(numerics)\n");
    for(i=0; i<10; i++)
    {
        printf("%6d %16.8g\n", i, E[i]);
    }

    free(H);
    free(E);
        
    return 0;
}
