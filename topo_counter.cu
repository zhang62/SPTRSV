#include "lusol.h"

__global__ void LEVEL_CSC_SYNC_0(int n, int *dp, int *jlev, int *tail);

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
static void atomicDecL0(unsigned long long int *address, int *oldlow, int *oldhigh) {
  volatile unsigned long long *vaddr = address;
  unsigned long long old_ull = *vaddr, assumed_ull;
  do {
    assumed_ull = old_ull;
    int old_count =  *(ull_high(&assumed_ull));
    unsigned long long int new_ull = old_count > 0 ? assumed_ull + PLUS_MINUS : assumed_ull;
    old_ull = atomicCAS(address, assumed_ull, new_ull);
  } while (old_ull != assumed_ull);

  *oldlow  = *(ull_low (&old_ull));
  *oldhigh = *(ull_high(&old_ull));
}

__global__ void TOPO_CSC_L(int n, int *ib, int *jb, int *db, int *dp, 
                           int *jlev, int *last, unsigned long long *ull) {
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  // local warp id
  const int wlane = threadIdx.x / WARP;
  int i, first, count;
  volatile __shared__ int s_first[BLOCKDIM / WARP];
  volatile __shared__ int s_count[BLOCKDIM / WARP];
  volatile int *vjlev = jlev;

  do {
    if (lane == 0) {
      atomicDecL0(ull, &first, &count);
      s_first[wlane] = first;
      s_count[wlane] = count;
    }
    first = s_first[wlane];
    count = s_count[wlane];

    if (count > 0) {
      while ((i = vjlev[first]) == 0);
      --i;
      int q1 = db[i] + 1;
      int q2 = ib[i+1];
      for (int j = q1 + lane; j < q2; j += WARP) {
        int k = jb[j-1]-1;
        int old = atomicSub(&dp[k], 1);
        if (old == 1) {
          int p = atomicAdd(last, 1);
          vjlev[p] = k + 1;
          atomicAdd(ull_high(ull), 1);
        }
      }
    }
  } while (first < n);
}

__global__ void TOPO_CSC_U(int n, int *ib, int *jb, int *db, int *dp, 
                           int *jlev, int *last, unsigned long long *ull) {
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  // local warp id
  const int wlane = threadIdx.x / WARP;
  int i, first, count;
  volatile __shared__ int s_first[BLOCKDIM / WARP];
  volatile __shared__ int s_count[BLOCKDIM / WARP];
  volatile int *vjlev = jlev;

  do {
    if (lane == 0) {
      atomicDecL0(ull, &first, &count);
      s_first[wlane] = first;
      s_count[wlane] = count;
    }
    first = s_first[wlane];
    count = s_count[wlane];

    if (count > 0) {
      while ((i = vjlev[first]) == 0);
      --i;
      int q1 = ib[i];
      int q2 = db[i];
      for (int j = q1 + lane; j < q2; j += WARP) {
        int k = jb[j-1]-1;
        int old = atomicSub(&dp[k], 1);
        if (old == 1) {
          int p = atomicAdd(last, 1);
          vjlev[p] = k + 1;
          atomicAdd(ull_high(ull), 1);
        }
      }
    }
  } while (first < n);
}

void makeTopoCSC(int n, int *d_ib, int *d_jb, int *d_db, 
                 int *d_dp, int *d_jlevL, int *d_jlevU) {
  int gDim;
  int *d_dpL = d_dp;
  int *d_dpU = d_dp + n;

  unsigned long long *d_ull;
  cudaMalloc((void **)&d_ull, sizeof(unsigned long long));
  
  int *d_count = ull_high(d_ull);
  int *d_first = ull_low(d_ull);

  int *d_last;
  cudaMalloc((void **)&d_last, sizeof(int));

  int nthreads = 500 * WARP;
  
  // L
  cudaMemset(d_ull, 0, sizeof(unsigned long long));
  cudaMemset(d_jlevL, 0, n*sizeof(int));

  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpL, d_jlevL, d_count);
  cudaMemcpy(d_last, d_count, sizeof(int), cudaMemcpyDeviceToDevice);
  
  gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;
  TOPO_CSC_L<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpL, d_jlevL, d_last, d_ull);

  // U
  cudaMemset(d_ull, 0, sizeof(unsigned long long));
  cudaMemset(d_jlevU, 0, n*sizeof(int));

  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpU, d_jlevU, d_count);
  cudaMemcpy(d_last, d_count, sizeof(int), cudaMemcpyDeviceToDevice);
  
  gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;
  TOPO_CSC_U<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpU, d_jlevU, d_last, d_ull);

  cudaFree(d_ull);
  cudaFree(d_last);
}

void checktopo(int n, int *ib, int *jb, int *db, int *d_jlevL, 
               int *d_jlevU, int *d_dp) {
  int *jlevL = (int *) malloc(n*sizeof(int));
  int *jlevU = (int *) malloc(n*sizeof(int));
  int *dp    = (int *) malloc(2*n*sizeof(int));
  cudaMemcpy(jlevL, d_jlevL, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(jlevU, d_jlevU, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(dp, d_dp, 2*n*sizeof(int), cudaMemcpyDeviceToHost);
  int *dpL = dp;
  int *dpU = dp + n;

  //for (int i = 0; i < n; i++) { printf("%d ", jlevL[i]); } printf("\n");
  //for (int i = 0; i < n; i++) { printf("%d ", jlevU[i]); } printf("\n");

  for (int i = 0; i < n; i++) {
    int jl = jlevL[i];
    int ju = jlevU[i];
    
    if (jl < 1 || jl > n) {
      printf("topo error: jl = %d\n", jl);
      exit(0);
    }
    if (ju < 1 || ju > n) {
      printf("topo error: ju = %d\n", ju);
      exit(0);
    }
    if (dpL[jl-1] != 0) {
      printf("topo error: dpL[%d] = %d\n", jl-1, dpL[jl-1]);
      exit(0);
    }
    if (dpU[ju-1] != 0) {
      printf("topo error: dpU[%d] = %d\n", ju-1, dpU[ju-1]);
      exit(0);
    }

    for (int j = db[jl-1]+1; j < ib[jl]; j++) {
      int k = jb[j-1]-1;
      dpL[k]--;
    }
    for (int j = ib[ju-1]; j < db[ju-1]; j++) {
      int k = jb[j-1]-1;
      dpU[k]--;
    }
  }

  free(jlevL);
  free(jlevU);
  free(dp);
}

