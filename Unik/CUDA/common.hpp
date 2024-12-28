#ifndef COMMON_HPP
#define COMMON_HPP

#include <stdio.h>

# define check(expr, message) __check(expr, message, __FILE__, __LINE__)

static void __check(cudaError_t err, const char *message, const char *file, long line)
{
    if (err != cudaSuccess) {
        printf("%s:%ld: ERROR: %s: %s\n", file, line, message, cudaGetErrorString(err));
        fflush(stdout);
    }
}

static cudaEvent_t __start, __stop;                                                
static float __time;                                                           

static void timer_start(void)
{
                                                                                
    check(cudaEventCreate(&__start), "start event create fail");                
    check(cudaEventCreate(&__stop), "end event create fail");                   
                                                                                
    check(cudaEventRecord(__start, 0), "start event record fail");              
}

static float timer_stop(void)
{
    //check(cudaGetLastError(), "kernel run fail");                               
    check(cudaEventRecord(__stop), "end event record fail");                    
    check(cudaEventSynchronize(__stop), "stop synchronize fail");               
                                                                                
    check(cudaEventElapsedTime(&__time, __start, __stop), "elapsed time fail"); 
                                                                                
    check(cudaEventDestroy(__start), "start event destroy");                    
    check(cudaEventDestroy(__stop), "stop event destroy");                      
    
    return __time;
}


#endif
