/**
 * High Performance Computing in Scientific Applications, WUT, 2023
 * 
 * Author: ???
 * Date: ???
 * 
 * Compilation command:
 *      gcc covariance.c -o covariance -lm -O3
 * */

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "mct_utils.h"

#include <mpi.h>


#define NELEMENTS 134217728
#define NVARS 5
#define CHUNKSIZE 10000

long int get_Nip( int ip , int np , long int N ) 
{
  long int Nip;
  Nip  = N / np ;
  if ( ip < (N % np) ) 
    Nip++ ;
  return Nip;
}

long int get_i0( int ip , int np , long int N ) 
{
  long int Nip;
  long int i0=0; 
  int i;
  for(i=0; i<ip; i++)
  {
    Nip  = N / np ;
    if ( i < (N % np) ) 
        Nip++ ;
    i0+=Nip;
  }
  return i0;
}

int main( int argc , char ** argv ) 
{
    // Initilize MPI 
    int ip, np; // basic MPI indicators
    MPI_Init( &argc , &argv ) ; /* set up the parallel WORLD */
    MPI_Comm_size( MPI_COMM_WORLD , &np ) ; /* total number of processes */
    MPI_Comm_rank( MPI_COMM_WORLD , &ip ) ; /* id of process st 0 <= ip < np */ 


    size_t i, j, k;
    
    // allocate memory
    double *var[NVARS+5];
    for(i=0; i<NVARS; i++) cppmallocl(var[i],NELEMENTS,double);     

    
    // Read binary data
    b_t(); // start timing
    FILE * fp;
    char file_name[128];
    for(i=0; i<NVARS; i++)
    {
        sprintf(file_name, "/home2/archive/mct/labs/lab1/var%d.dat", i+1);
        printf("# READING DATA: `%s`...\n", file_name);
        
        fp = fopen(file_name, "rb");
        if(fp==NULL) { printf("# ERROR: Cannot open `%s`!\n", file_name); return EXIT_FAILURE; }
        
        size_t ftest = fread(var[i], sizeof(double), NELEMENTS, fp);
        if(ftest!=NELEMENTS) { printf("# ERROR: Cannot read `%s`!\n", file_name); return EXIT_FAILURE; }
        
        fclose(fp);
    }

    // Test data: print out first and last elements
    if(ip == 0){
        printf("# READ TIME (process 0): %f sec\n", tio);
        // Test data: print out first and last elements
        for(i=0; i<NVARS; i++) printf("%8d %16.8f\n", 0, var[i][0]);
        for(i=0; i<NVARS; i++) printf("%8d %16.8f\n", -1, var[i][getNip(ip,np,NELEMENTS)-1]);
    }

    // allocate additional buffers
    for(i=0; i<5; i++) cppmallocl(var[NVARS+i],NELEMENTS,double);     
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(ip == 0){
        b_t(); // start timing
    }

    
    // create auxliary variables
    printf("# COMPUTING AUXILIARY VARIABLES\n");
    for(k=0; k<NELEMENTS; k++)
    {   
        var[NVARS+0][k] = sin(var[0][k]) + cos(var[1][k]);
        var[NVARS+1][k] = exp(var[2][k]) + exp(-1.*var[3][k]);
        var[NVARS+2][k] = sin(var[0][k])*cos(var[1][k]) + cos(var[2][k])*sin(var[3][k]);
        var[NVARS+3][k] = hypot(var[1][k], var[2][k]);
        var[NVARS+4][k] = cbrt(var[4][k]);
    }
    
    // compute averages
    printf("# COMPUTING AVERAGES\n");
    double avg[NVARS+5];
    double all_avg[NVARS+5];
    for(i=0; i<NVARS+5; i++)
    {
        avg[i]=0.0;
        for(k=0; k<get_Nip(ip,np,NELEMENTS); k++) avg[i]+=var[i][k];
        MPI_Allreduce(&avg[i],&all_avg[i],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        all_avg[i]/=NELEMENTS;
        
        for(k=0; k<get_Nip(ip,np,NELEMENTS); k++) var[i][k]-=all_avg[i];
        
    }
    
    // compute covariance matrix
    printf("# COMPUTING COVARIANCE MATRIX\n");
    double cov[NVARS+5][NVARS+5];
    for(i=0; i<NVARS+5; i++) for(j=0; j<=i; j++)
    {
        cov[i][j]=0.0;
        for(k=0; k<get_Nip(ip,np,NELEMENTS); k++) 
            cov[i][j]+=var[i][k]*var[j][k]; 
        MPI_Allreduce(&cov[i][j],&cov_global[i][j],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        cov_global[i][j]/=(NELEMENTS-1);

    }



    if(ip ==0)
    {
        double tcmp = e_t(); // stop timing
        printf("# COMPUTATION TIME: %f sec\n", tcmp);
        
        // print results
        for(i=0; i<NVARS+5; i++) for(j=0; j<=i; j++)
        {
            printf("cov(%2d,%2d)=%16.8g\n", i+1, j+1, cov[i][j]);
        }

        FILE *fptr;
        fptr = fopen("/home/ops003/HPC/lab2/t.txt", "a");
        fprintf(fptr, "%f\t%d\n",tcmp,np);
        fclose(fptr); 

    }
    
    // done with MPI  
    MPI_Finalize ();
    return( EXIT_SUCCESS ) ;
}

// # COMPUTATION TIME: 257.470598 sec
// cov( 1, 1)=      0.33330407
// cov( 2, 1)=      0.33330627
// cov( 2, 2)=        0.354141
// cov( 3, 1)=       0.9999136
// cov( 3, 2)=       1.0415912
// cov( 3, 3)=       3.4164322
// cov( 4, 1)=     0.083315582
// cov( 4, 2)=     0.083314822
// cov( 4, 3)=      0.24996499
// cov( 4, 4)=      0.11665102
// cov( 5, 1)=      0.99991222
// cov( 5, 2)=      0.99991882
// cov( 5, 3)=       2.9997408
// cov( 5, 4)=      0.24994675
// cov( 5, 5)=       2.9997367
// cov( 6, 1)=      0.22741816
// cov( 6, 2)=      0.22311045
// cov( 6, 3)=      0.67362109
// cov( 6, 4)=     0.011154728
// cov( 6, 5)=      0.68225449
// cov( 6, 6)=      0.17943495
// cov( 7, 1)=       4.4811931
// cov( 7, 2)=       4.7587534
// cov( 7, 3)=       16.109993
// cov( 7, 4)=       2.7202396
// cov( 7, 5)=       13.443579
// cov( 7, 6)=       2.1790936
// cov( 7, 7)=       146.11918
// cov( 8, 1)=      0.13701535
// cov( 8, 2)=      0.12998218
// cov( 8, 3)=      0.40114765
// cov( 8, 4)=     -0.04420721
// cov( 8, 5)=      0.41104604
// cov( 8, 6)=      0.13412504
// cov( 8, 7)=      0.23848724
// cov( 8, 8)=      0.15351441
// cov( 9, 1)=      0.26130321
// cov( 9, 2)=      0.26987497
// cov( 9, 3)=      0.84780182
// cov( 9, 4)=      0.31189547
// cov( 9, 5)=      0.78390963
// cov( 9, 6)=     0.058905463
// cov( 9, 7)=       9.1730406
// cov( 9, 8)=     -0.12185077
// cov( 9, 9)=       1.1349119
// cov(10, 1)=      0.61807095
// cov(10, 2)=      0.61807478
// cov(10, 3)=       1.8542035
// cov(10, 4)=      0.15449216
// cov(10, 5)=       1.8542129
// cov(10, 6)=      0.42628017
// cov(10, 7)=       7.7737301
// cov(10, 8)=      0.28316258
// cov(10, 9)=      0.52532319
// cov(10,10)=       1.2480151
