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
#include <fftw3.h>
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
    
    // TODO: add code here that computes laplace 2d numerically
    double *laplacefxy; // pointer to array with laplace computed numerically
    laplacefxy = (double *) malloc(nx*ny*sizeof(double)); 

    // wotking buffer for fftw
    double complex *fk = (double complex *) malloc(nx*ny*sizeof(double complex)); if(fk==NULL) { printf("Cannot allocate array fk!\n"); return 0;}

    // Create fftw plan
    fftw_plan plan_f = fftw_plan_dft_2d(nx, ny, fk, fk, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_b = fftw_plan_dft_2d(nx, ny, fk, fk, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    // copy data to working buffer
    ixy=0;
    for(ix=0; ix<nx; ix++) for(iy=0; iy<ny; iy++) { fk[ixy] =fxy[ixy] + I*0.0; ixy++; }

    // compute second derivative
    fftw_execute(plan_f);
    // momentum space
    double kx, ky;
    ixy=0;
    for(ix=0; ix<nx; ix++) for(iy=0; iy<ny; iy++)
    {
        if(ix<nx/2) kx=2.*M_PI/(dx*nx)*(ix   );
        else        kx=2.*M_PI/(dx*nx)*(ix-nx);

        if(iy<ny/2) ky=2.*M_PI/(dy*ny)*(iy   );
        else        ky=2.*M_PI/(dy*ny)*(iy-ny);

        fk[ixy] *= -1.0*(kx*kx + ky*ky) / (nx*ny);

        ixy++;
    }

    // // NOTE: you can realize the above double loop in sigle loop
    // for(ixy=0; ixy<nx*ny; ixy++)
    // {
    //     ix=ixy/ny;
    //     iy=ixy-ix*ny;
    //     if(ix<nx/2) kx=2.*M_PI/(dx*nx)*(ix   );
    //     else        kx=2.*M_PI/(dx*nx)*(ix-nx);
    //
    //     if(iy<ny/2) ky=2.*M_PI/(dy*ny)*(iy   );
    //     else        ky=2.*M_PI/(dy*ny)*(iy-ny);
    //
    //     fk[ixy] *= -1.0*(kx*kx + ky*ky) / (nx*ny);
    // }

    fftw_execute(plan_b);

    // copy result to final buffer
    ixy=0;
    for(ix=0; ix<nx; ix++) for(iy=0; iy<ny; iy++) { laplacefxy[ixy] = creal(fk[ixy]); ixy++; }
    

    // Check correctness of computation
    test_array_diff(nx*ny, laplacefxy, formula_laplacefxy);
    
//     // you can also print section of function along some direction
//     ixy=0;
//     for(ix=0; ix<nx; ix++) for(iy=0; iy<ny; iy++) 
//     {
//         if(ix==nx/2) printf("%10.6g %10.6f %10.6f %10.6f\n", x0 + dx*ix, y0 + dy*iy, laplacefxy[ixy], formula_laplacefxy[ixy]);
// 
//         ixy++;
//     } 
    
    return 1;
}
