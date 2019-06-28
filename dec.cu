#include "lusol.h"

#define ZERO_PLUS 0x0000000000000001 //[0, 1]
/* in computers, it is stored as [low high] : [1, 0] */

#define PLUS_MINUS 0xFFFFFFFF00000001 //[-1, 1]
/* in computers, it is stored as [low high] : [1, -1] */

__forceinline__ __device__ __host__
int *ull_low(unsigned long long int *x) {
  return ((int*) x);
}

__forceinline__ __device__ __host__
int *ull_high(unsigned long long int *x) {
  return ((int*) x + 1);
}

__forceinline__ __device__ 
static int atomicDecL0(unsigned long long int *address) {
  volatile unsigned long long *vaddr = address;
  unsigned long long old_ull = *vaddr, assumed_ull;
  do {
    assumed_ull = old_ull;
    //unsigned int old_count = ((unsigned int *) &assumed_ull)[1];
    int old_count =  *(ull_high(&assumed_ull));
    unsigned long long int new_ull = old_count > 0 ? assumed_ull + PLUS_MINUS : assumed_ull;
    old_ull = atomicCAS(address, assumed_ull, new_ull);
  } while (old_ull != assumed_ull);

  return old_ull;
}

__forceinline__ __device__ 
static void atomicInc_ULL_Low(unsigned long long int *address, int *old_low, int *old_high) {
  volatile unsigned long long *vaddr = address;
  unsigned long long old_ull = *vaddr, assumed_ull;
  do {
    assumed_ull = old_ull;
    int assumed_low  =  *(ull_low (&assumed_ull));
    int assumed_high =  *(ull_high(&assumed_ull));
    unsigned long long int new_ull = assumed_low < assumed_high ? assumed_ull + ZERO_PLUS : assumed_ull;
    old_ull = atomicCAS(address, assumed_ull, new_ull);
  } while (old_ull != assumed_ull);

  *old_low  = *(ull_low (&old_ull));
  *old_high = *(ull_high(&old_ull));
}


__global__ void doatomicDecL0(unsigned long long int *val) {
  atomicDecL0(val);
}

__global__ void doatomicInc_ULL_Low(unsigned long long int *val, int *low, int *high) {
  atomicInc_ULL_Low(val, low, high);
}

__global__ void doatomicDec(unsigned int *address, unsigned int val) {
  atomicDec(address, val);
}


int main() {
  unsigned long long oldval, *d_val, newval, inc=PLUS_MINUS, newvalcpu, inc2=ZERO_PLUS;
  int *d_low, *d_high, low, high;

  unsigned int *d_counter, counter=1000, newcounter;
  cudaMalloc((void **)&d_counter, sizeof(unsigned int));
  cudaMemcpy(d_counter, &counter, sizeof(unsigned int), cudaMemcpyHostToDevice);
  doatomicDec<<<1, 103>>>(d_counter, 100);
  cudaMemcpy(&newcounter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("new counter %u\n", newcounter);

  return 0;

  /*
  *(ull_high(&oldval)) = 33; // count
  *(ull_low (&oldval)) = 0;  // first

  cudaMalloc((void **)&d_val, sizeof(unsigned long long int));
  cudaMemcpy(d_val, &oldval, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

  doatomicDecL0<<<1, 34>>>(d_val);

  cudaMemcpy(&newval, d_val, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  newvalcpu = oldval + inc;

  printf("before: %d %d\n", *ull_high(&oldval), *ull_low(&oldval));
  printf("add   : %d %d\n", *ull_high(&inc),    *ull_low(&inc));

  //printf("CPU   : %d %d\n", *ull_high(&newvalcpu), *ull_low(&newvalcpu));
  printf("GPU   : %d %d\n", *ull_high(&newval), *ull_low(&newval));
*/

  *(ull_high(&oldval)) = 5; // count
  *(ull_low (&oldval)) = 0;  // first

  cudaMalloc((void **)&d_low, sizeof(int));
  cudaMalloc((void **)&d_high, sizeof(int));
  cudaMalloc((void **)&d_val, sizeof(unsigned long long int));
  cudaMemcpy(d_val, &oldval, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

  int gDim = 3;
  int bDim = 1;
  doatomicInc_ULL_Low<<<gDim, bDim>>>(d_val, d_low, d_high);

  cudaMemcpy(&newval, d_val, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&low, d_low, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&high, d_high, sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for (i=0, newvalcpu=oldval; i<gDim*bDim; i++) {
    newvalcpu += inc2;
  }

  printf("before: %d %d\n", *ull_high(&oldval), *ull_low(&oldval));
  printf("add   : %d %d\n", *ull_high(&inc2),   *ull_low(&inc2));

  printf("CPU   : %d %d\n", *ull_high(&newvalcpu), *ull_low(&newvalcpu));
  printf("GPU   : %d %d\n", *ull_high(&newval), *ull_low(&newval));

  printf("Old: low %d high %d\n", low, high);

  cudaFree(d_low);
  cudaFree(d_high);
  cudaFree(d_val);
}
