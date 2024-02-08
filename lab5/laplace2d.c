#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cufft.h>

#define BLOCK_SIZE 16

__global__ void multiply_kernel(double *fk_real, double *fk_imag, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        double kx = (idx < nx / 2) ? 2.0 * M_PI / (dx * nx) * idx : 2.0 * M_PI / (dx * nx) * (idx - nx);
        double ky = (idy < ny / 2) ? 2.0 * M_PI / (dy * ny) * idy : 2.0 * M_PI / (dy * ny) * (idy - ny);
        double factor = -1.0 * (kx * kx + ky * ky) / (nx * ny);
        double real = fk_real[idy * nx + idx];
        double imag = fk_imag[idy * nx + idx];
        fk_real[idy * nx + idx] = real * factor;
        fk_imag[idy * nx + idx] = imag * factor;
    }
}

double function_xy(double x, double y) {
    #define A -0.03
    #define B -0.01
    #define C -0.005
    return exp(A * x * x + B * y * y + C * x * y);
}

double laplace_function_xy(double x, double y) {
    double rdf2d_dx = function_xy(x, y) * (2.0 * A * x + C * y); // d/dx
    double rdf2d_dy = function_xy(x, y) * (2.0 * B * y + C * x); // d/dy
    double rlaplacef2d = rdf2d_dx * (2.0 * A * x + C * y) + function_xy(x, y) * (2.0 * A) +
                         rdf2d_dy * (2.0 * B * y + C * x) + function_xy(x, y) * (2.0 * B); // laplace
    return rlaplacef2d;
}

void test_array_diff(int N, double *a, double *b) {
    double maxd2 = 0.0;
    double sumd2 = 0.0;
    for (int i = 0; i < N; i++) {
        double d = a[i] - b[i];
        double d2 = d * d;
        sumd2 += d2;
        if (d2 > maxd2) maxd2 = d2;
    }
    printf("#    COMPARISON RESULTS:\n");
    printf("#           |max[a-b]| : %16.8g\n", sqrt(maxd2));
    printf("#         SUM[(a-b)^2] : %16.8g\n", sumd2);
    printf("# SQRT(SUM[(a-b)^2])/N : %16.8g\n", sqrt(sumd2) / N);
}

int main() {
    int nx = 512; // number of points in x-direction
    int ny = 512; // number of points in y-direction
    double Lx = 100.0; // width in x-direction
    double Ly = 100.0; // width in y-direction
    double x0 = -Lx / 2;
    double y0 = -Ly / 2;
    double dx = Lx / nx;
    double dy = Ly / ny;

    double *fxy;
    double *formula_laplacefxy;
    fxy = (double *)malloc(nx * ny * sizeof(double));
    formula_laplacefxy = (double *)malloc(nx * ny * sizeof(double));

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            fxy[iy * nx + ix] = function_xy(x0 + dx * ix, y0 + dy * iy);
            formula_laplacefxy[iy * nx + ix] = laplace_function_xy(x0 + dx * ix, y0 + dy * iy);
        }
    }

    // Allocate memory on the GPU
    double *fk_real, *fk_imag;
    cudaMalloc((void **)&fk_real, nx * ny * sizeof(double));
    cudaMalloc((void **)&fk_imag, nx * ny * sizeof(double));

    // Copy data from CPU to GPU
    cudaMemcpy(fk_real, fxy, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(fk_imag, fxy, nx * ny * sizeof(double), cudaMemcpyHostToDevice);

    // Compute FFT
    cufftHandle plan;
    cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z);
    cufftExecZ2Z(plan, (cufftDoubleComplex *)fk_real, (cufftDoubleComplex *)fk_imag, CUFFT_FORWARD);

    // Launch CUDA kernel for multiplication
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);
    multiply_kernel<<<numBlocks, threadsPerBlock>>>(fk_real, fk_imag, nx, ny, dx, dy);
    cudaDeviceSynchronize();

    // Compute inverse FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex *)fk_real, (cufftDoubleComplex *)fk_imag, CUFFT_INVERSE);

    // Copy result back to CPU
    cudaMemcpy(fxy, fk_real, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

    // Check correctness of computation
    test_array_diff(nx * ny, fxy, formula_laplacefxy);

    // Free allocated memory
    cufftDestroy(plan);
    cudaFree(fk_real);
    cudaFree(fk_imag);
    free(fxy);
    free(formula_laplacefxy);

    return 0;
}
