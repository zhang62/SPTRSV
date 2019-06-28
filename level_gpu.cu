#include "lusol.h"

__global__ void LEVEL_CSC_SYNC_0(int n, int *dp, int *jlev, int *tail) {

  int gid = blockIdx.x * BLOCKDIM + threadIdx.x;

  if (gid >= n) {
    return;
  }

  if (dp[gid] == 0) {
    int p = atomicAdd(tail, 1);
    jlev[p] = gid + 1;
  }
}

__global__ void LEVEL_CSC_SYNC_L(int n, int *ib, int *jb, int *db, int *dp, 
                                 int *jlev, int count, int head, int *tail) {
  // global warp id
  const int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  
  if (wid >= count) {
    return;
  }

  int i = jlev[head+wid] - 1;
  int q1 = db[i] + 1;
  int q2 = ib[i+1];
  for (int j = q1 + lane; j < q2; j += WARP) {
    int k = jb[j-1]-1;
    int old = atomicSub(&dp[k], 1);
    if (old == 1) {
      int p = atomicAdd(tail, 1);
      jlev[p] = k + 1;
    }
  }
}

__global__ void LEVEL_CSC_SYNC_U(int n, int *ib, int *jb, int *db, int *dp, 
                                 int *jlev, int count, int head, int *tail) {
  // global warp id
  const int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  
  if (wid >= count) {
    return;
  }

  int i = jlev[head+wid] - 1;
  int q1 = ib[i];
  int q2 = db[i];
  for (int j = q1 + lane; j < q2; j += WARP) {
    int k = jb[j-1]-1;
    int old = atomicSub(&dp[k], 1);
    if (old == 1) {
      int p = atomicAdd(tail, 1);
      jlev[p] = k + 1;
    }
  }
}

void makeLevelCSC_SYNC(int n, int *d_ib, int *d_jb, int *d_db, 
                       int *d_dp, 
                       int *d_jlevL, int *ilevL, int *nlevL,
                       int *d_jlevU, int *ilevU, int *nlevU) {
  int gDim,lev;
  int *d_tail, tail, head;
  int *d_dpL = d_dp;
  int *d_dpU = d_dp + n;

  cudaMalloc((void **)&d_tail, sizeof(int));

  // L
  lev = 0;
  ilevL[lev++] = 1;
  head = 0;
  cudaMemset(d_tail, 0, sizeof(int));
  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpL, d_jlevL, d_tail);
  cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);

  while (tail < n) {
    int count = tail - head;
    int nthreads = count * WARP;
    int gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;
   
    LEVEL_CSC_SYNC_L<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpL, d_jlevL, count, head, d_tail);
  
    head = tail;
    cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
    ilevL[lev++] = head + 1;
  }
  ilevL[lev] = n + 1;
  *nlevL = lev;

  // U
  lev = 0;
  ilevU[lev++] = 1;
  head = 0;
  cudaMemset(d_tail, 0, sizeof(int));
  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpU, d_jlevU, d_tail);
  cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);

  while (tail < n) {
    int count = tail - head;
    int nthreads = count * WARP;
    int gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;

    LEVEL_CSC_SYNC_U<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpU, d_jlevU, count, head, d_tail);
  
    head = tail;
    cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
    ilevU[lev++] = head + 1;
  }
  ilevU[lev] = n + 1;
  *nlevU = lev;

  cudaFree(d_tail);

}
