#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * CUDA Project - HPC 2019 I
 *
 * This program demonstrates a simple simulation box which attempts to model 
 * traffic flow on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

void checkResult(bool *hostRef, bool *gpuRef, const int N)
{
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] ^ gpuRef[i])
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %d gpu %d at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void show(bool *ip, int size)
{
    // show a bool array
    for (int i = 0; i < size; i++)
    {
        printf("%d ", ip[i]);
    }
    printf("\n");
}

void initialData(bool *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (bool)(rand() & 1);
    }

    return;
}

void simulateFlowOnHost(bool *road_prev, bool *road_curr, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        if (road_prev[idx] == 1 && road_prev[(idx + 1) % N] == 0)
        {
            road_curr[idx] = 0;
            road_curr[(idx + 1) % N] = 1;
        }
    }
}

__global__ void simulateFlowOnGPU(bool *road_prev, bool *road_curr, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && road_prev[i] == 1 && road_prev[(i + 1) % N] == 0)
    {
        road_curr[i] = 0;
        road_curr[(i + 1) % N] = 1;
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem, nTimes;
    scanf("%d %d", &nElem, &nTimes);
    printf("Vector size %d\n", nElem);
    printf("Number of times %d\n", nTimes);

    // malloc host memory
    size_t nBytes = nElem * sizeof(bool);

    bool *h_road_init, *h_road_curr, *h_road_prev, *hostRef, *gpuRef;
    h_road_init = (bool *)malloc(nBytes);
    h_road_prev = (bool *)malloc(nBytes);
    h_road_curr = (bool *)malloc(nBytes);
    hostRef     = (bool *)malloc(nBytes);
    gpuRef      = (bool *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_road_init, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memcpy(h_road_curr, h_road_init, nBytes);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    for (int i = 0; i < nTimes; i++)
    {
        memcpy(h_road_prev, h_road_curr, nBytes);
        simulateFlowOnHost(h_road_prev, h_road_curr, nElem);
    }
    iElaps = seconds() - iStart;
    printf("simulateFlowOnHost Time elapsed %f sec\n", iElaps);
    memcpy(hostRef, h_road_curr, nBytes);

    // malloc device global memory
    bool *d_road_prev, *d_road_curr;
    CHECK(cudaMalloc((bool**)&d_road_prev, nBytes));
    CHECK(cudaMalloc((bool**)&d_road_curr, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_road_curr, h_road_init, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    iStart = seconds();
    for (int i = 0; i < nTimes; i++)
    {
        CHECK(cudaMemcpy(d_road_prev, d_road_curr, nBytes, cudaMemcpyDeviceToDevice));
        simulateFlowOnGPU<<<grid, block>>>(d_road_prev, d_road_curr, nElem);
        CHECK(cudaDeviceSynchronize());
    }
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_road_curr, nBytes, cudaMemcpyDeviceToHost));
    
    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_road_prev));
    CHECK(cudaFree(d_road_curr));

    // free host memory
    free(h_road_init);
    free(h_road_prev);
    free(h_road_curr);
    free(hostRef);
    free(gpuRef);

    return(0);
}