#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "mct_utils.h"

#define CHUNKSIZE 1000

#include <omp.h>

void get_var(long int NELEMENTS,long int NVARS,double *var){
    
    
    int chunk = CHUNKSIZE;
    long int k;
    #pragma omp parallel shared(var,chunk) private(k)
    {
        #pragma omp for schedule(static,chunk)  
        for(k=0; k<NELEMENTS; k++)
        {   
            var[(NVARS+0)*NELEMENTS+k] = sin(var[0*NELEMENTS+k]) + cos(var[1*NELEMENTS+k]);
            var[(NVARS+1)*NELEMENTS+k] = exp(var[2*NELEMENTS+k]) + exp(-1.*var[3*NELEMENTS+k]);
            var[(NVARS+2)*NELEMENTS+k] = sin(var[0*NELEMENTS+k])*cos(var[1*NELEMENTS+k]) + cos(var[2*NELEMENTS+k])*sin(var[3*NELEMENTS+k]);
            var[(NVARS+3)*NELEMENTS+k] = hypot(var[1*NELEMENTS+k], var[2*NELEMENTS+k]);
            var[(NVARS+4)*NELEMENTS+k] = cbrt(var[4*NELEMENTS+k]);
        }
    }   
}


void get_covariance(long int NELEMENTS,long int NVARS,double *var,double *cov){
    long int i,j,k;
    double avg[NVARS+5];
    double tmp_avg;
    int ch = CHUNKSIZE;
    
    int nthreads, tid;
    
    #pragma omp parallel private(tid)
    {
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        if(tid==0) printf("Number of threads = %d\n", nthreads);
    }
    
    
    
    for(i=0; i<NVARS+5; i++)
    {   
        avg[i]=0.0;
        tmp_avg=0.0;
    #pragma omp parallel shared(var,i,ch) private(k) reduction(+:tmp_avg)
    {
       #pragma omp for schedule(static,ch)
           for(k=0; k<NELEMENTS; k++) tmp_avg+=var[i*NELEMENTS+k];
    }
    avg[i]=tmp_avg/NELEMENTS;
        

    #pragma omp parallel shared(avg,var,i,ch) private(k)
    {
       #pragma omp for schedule(static,ch)
           for(k=0; k<NELEMENTS; k++) var[i*NELEMENTS+k]-=avg[i];
    }
    }

    double tmp_cov;
    for(i=0; i<NVARS+5; i++) for(j=0; j<=i; j++)
    {
        cov[i*(NVARS+5)+j]=0.0;
        tmp_cov=0.0;
        #pragma omp parallel shared(i,j,var,ch) private(k) reduction(+:tmp_cov)
    {
       #pragma omp for schedule(static,ch)
           for(k=0; k<NELEMENTS; k++) tmp_cov+=var[i*NELEMENTS+k]*var[j*NELEMENTS+k]; 
    }
        cov[i*(NVARS+5)+j]=tmp_cov/(NELEMENTS-1);
        cov[j*(NVARS+5)+i]=cov[i*(NVARS+5)+j];
    }
    
}
