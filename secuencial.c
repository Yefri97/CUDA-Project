#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


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

void show(bool *ip, int size)
{
    // show a bool array
    for (int i = 0; i < size; i++)
    {
        printf("%d ", ip[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    int nElem, nTimes;
    scanf("%d %d", &nElem, &nTimes);
    size_t nBytes = nElem * sizeof(bool);

    bool *h_road_prev, *h_road_curr;
    h_road_prev = (bool *)malloc(nBytes);
    h_road_curr = (bool *)malloc(nBytes);

    initialData(h_road_curr, nElem);
    // show(h_road_curr, nElem);
    
    clock_t tStart = clock();
    for (int i = 0; i < nTimes; i++)
    {
        memcpy(h_road_prev, h_road_curr, nBytes);
        simulateFlowOnHost(h_road_prev, h_road_curr, nElem);
        // show(h_road_curr, nElem);
    }
    clock_t tEnd = clock();

    free(h_road_prev);
    free(h_road_curr);

    printf("Time taken: %.8fs\n", (double)(tEnd - tStart)/CLOCKS_PER_SEC);

    return(0);
}