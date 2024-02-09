/**
 * Template for lab5: compute laplace of function
 * gcc laplace2d.c -o laplace2d -lm -lfftw3
 *
 * */ 

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif 

// See for reference:
// https://en.cppreference.com/w/c/numeric/complex
#include <complex.h>

// See for reference:
// http://www.fftw.org/fftw3_doc/
//#include <fftw3.h>
#include <cufftw.h>
#include <cuda_runtime_api.h>
/**
 * Test function
 * */
double function_xy(double x, double y)
{
    #define A -0.03
    #define B -0.01
    #define C -0.005 
    return exp(A*x*x + B*y*y + C*x*y);
}

/**
 * Analytical result
 * Use it for checking correctness!
 * */
double laplace_function_xy(double x, double y)
{
    double rdf2d_dx = function_xy(x,y)*(2.*A*x + C*y); // d/dx
    double rdf2d_dy = function_xy(x,y)*(2.*B*y + C*x); // d/dy
    
    double rlaplacef2d = rdf2d_dx*(2.*A*x + C*y)+function_xy(x,y)*(2.*A)  +  rdf2d_dy*(2.*B*y + C*x)+function_xy(x,y)*(2.*B); // laplace
    
    return rlaplacef2d;
    
    #undef A
    #undef B
    #undef C
}


/**
 * You can use this function to check diff between two arrays
 * */
void test_array_diff(int N, double *a, double *b)
{
    int ixyz=0;
    
    double d,d2;
    double maxd2 = 0.0;
    double sumd2 = 0.0;
    for(ixyz=0; ixyz<N; ixyz++)
    {
        d = a[ixyz]-b[ixyz];
        
        d2=d*d;
        sumd2+=d2;
        if(d2>maxd2) maxd2=d2;
    }
    
    printf("#    COMPARISON RESULTS:\n");
    printf("#           |max[a-b]| : %16.8g\n", sqrt(maxd2));
    printf("#         SUM[(a-b)^2] : %16.8g\n", sumd2);
    printf("# SQRT(SUM[(a-b)^2])/N : %16.8g\n", sqrt(sumd2)/N);
}

//cuda
__global__ void mykernel(cufftDoubleComplex *fk,int ny , int nx, double dx, double dy, size_t n)
{
  size_t ixy = blockIdx.x*blockDim.x+threadIdx.x;
  double ky,kx;
  if(ixy<n)
  {

        int ix=ixy/ny;
        int iy=ixy-ix*ny;


        if(ix<nx/2) kx=2.*M_PI/(dx*nx)*(ix);
        else        kx=2.*M_PI/(dx*nx)*(ix-nx);

        if(iy<ny/2) ky=2.*M_PI/(dy*ny)*(iy);
        else        ky=2.*M_PI/(dy*ny)*(iy-ny);


        fk[ixy].x *= -1.0*(kx*kx + ky*ky) / (nx*ny);


  }
}





int main()
{
    
    // Settings
    int nx = 512; // number of points in x-direction
    int ny = 512; // number of points in y-direction
    
    double Lx = 100.0; // width in x-direction
    double Ly = 100.0; // width in y-direction
    
    double x0 = -Lx/2;
    double y0 = -Ly/2;
    
    double dx = Lx/nx;
    double dy = Ly/ny;
    
    
    double *fxy; // function
    double *formula_laplacefxy; // array with results computed according formula
    fxy = (double *) malloc(nx*ny*sizeof(double)); 
    if(fxy==NULL) { printf("Cannot allocate array fx!\n"); return 0;}
    
    formula_laplacefxy = (double *) malloc(nx*ny*sizeof(double)); 
    if(formula_laplacefxy==NULL) { printf("Cannot allocate array fx!\n"); return 0;}
    
    int ix, iy, ixy;
    
    ixy=0;
    for(ix=0; ix<nx; ix++) for(iy=0; iy<ny; iy++) 
    {
        // function
        fxy[ixy] = function_xy(x0 + dx*ix, y0 + dy*iy);
        
        // result for comparion
        formula_laplacefxy[ixy] = laplace_function_xy(x0 + dx*ix, y0 + dy*iy);
        ixy++;
    }
    
    double *laplacefxy; // pointer to array with laplace computed numerically
    laplacefxy = (double *) malloc(nx*ny*sizeof(double)); 

    cufftDoubleComplex *hfk;
    cudaMallocHost((void**)&hfk,  nx*ny*sizeof(double complex));

    cufftDoubleComplex *fk;
    cudaMalloc(&fk,nx*ny*sizeof(cufftDoubleComplex));


    // Create fftw plan
    cufftHandle plan_f;
    cufftPlan2d(&plan_f,nx,ny,CUFFT_Z2Z);
    cufftHandle plan_b;
    cufftPlan2d(&plan_b,nx,ny,CUFFT_Z2Z);   


    cudaMemcpy2D (fk, nx*sizeof(cufftDoubleComplex), fxy, nx*sizeof(double complex), nx*sizeof(double complex),ny, cudaMemcpyHostToDevice);

    ixy=0;
    
    cudaMemcpy2D (laplacefxy, nx*sizeof(double complex),fk, nx*sizeof(cufftDoubleComplex), nx*sizeof(double complex ),ny, cudaMemcpyDeviceToHost);



    cudaDeviceSynchronize();

    double *laplacefxy2= (double*) malloc(nx*ny*sizeof(double));

    for(int i=0;i<n;i++)
    {
      laplacefxy2[i] = creal(laplacefxy[i]);
    }

    test_array_diff(nx*ny, laplacefxy2, formula_laplacefxy);
    
    return 1;
}