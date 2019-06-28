#include "lusol.h"

#undef BLOCKDIM
#define BLOCKDIM 1024

#define WARP0 8 // ``warp size'' used in the 1-block kernels

#define MIN_BLOCK 1

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

__global__ void LEVEL_CSC_SYNC_L_1BLOCK(int n, int *ib, int *jb, int *db, int *dp,
                                        int *jlev, int head, int old_tail, 
                                        int *tail, int *ntails) {
  // global warp id
  const int wid = threadIdx.x / WARP0;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP0 - 1);
  __shared__ volatile int s_tail;
  int nt = 0;

  if (threadIdx.x == 0) {
    s_tail = old_tail;
  }

  __syncthreads();

  do {
    int count = old_tail - head;
    for (int k = wid;  k < count; k += BLOCKDIM/WARP0) {
      int i = jlev[head+k] - 1;
      int q1 = db[i] + 1;
      int q2 = ib[i+1];
      for (int j = q1 + lane; j < q2; j += WARP0) {
        int k = jb[j-1]-1;
        int old = atomicSub(&dp[k], 1);
        if (old == 1) {
          int p = atomicAdd((int *) &s_tail, 1);
          jlev[p] = k + 1;
        }
      }
    }
    
    __syncthreads();

    int new_tail = s_tail;
    int new_count = new_tail - old_tail;

    if (threadIdx.x == 0) {
      tail[nt++] = new_tail;
    }

    if (new_tail >= n || new_count * WARP0 > MIN_BLOCK * BLOCKDIM) {
      if (threadIdx.x == 0) {
        *ntails = nt;
      }
      return;
    }

    head = old_tail;
    old_tail = new_tail;
  } while (1);
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

__global__ void LEVEL_CSC_SYNC_U_1BLOCK(int n, int *ib, int *jb, int *db, int *dp,
                                        int *jlev, int head, int old_tail, 
                                        int *tail, int *ntails) {
  // global warp id
  const int wid = threadIdx.x / WARP0;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP0 - 1);
  __shared__ volatile int s_tail;
  int nt = 0;

  if (threadIdx.x == 0) {
    s_tail = old_tail;
  }

  __syncthreads();

  do {
    int count = old_tail - head;
    for (int k = wid; k < count; k += BLOCKDIM/WARP0) {
      int i = jlev[head+k] - 1;
      int q1 = ib[i];
      int q2 = db[i];
      for (int j = q1 + lane; j < q2; j += WARP0) {
        int k = jb[j-1]-1;
        int old = atomicSub(&dp[k], 1);
        if (old == 1) {
          int p = atomicAdd((int *) &s_tail, 1);
          jlev[p] = k + 1;
        }
      }
    }
    
    __syncthreads();

    int new_tail = s_tail;
    int new_count = new_tail - old_tail;

    if (threadIdx.x == 0) {
      tail[nt++] = new_tail;
    }

    if (new_tail >= n || new_count * WARP0 > MIN_BLOCK * BLOCKDIM) {
      if (threadIdx.x == 0) {
        *ntails = nt;
      }
      return;
    }

    head = old_tail;
    old_tail = new_tail;
  } while (1);
}

void makeLevelCSC_SYNC(int n, int *d_ib, int *d_jb, int *d_db, 
                       int *d_dp, 
                       int *d_jlevL, int *ilevL, int *nlevL,
                       int *d_jlevU, int *ilevU, int *nlevU) {
  int gDim,lev;
  int *d_tail, *d_out, *d_Nout, tail, head;
  int *d_dpL = d_dp;
  int *d_dpU = d_dp + n;

  cudaMalloc((void **)&d_out, (n+2)*sizeof(int));
  d_Nout = d_out + n;
  d_tail = d_Nout + 1;

  // L
  ilevL[0] = 0;
  head = 0;
  cudaMemset(d_tail, 0, sizeof(int));
  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpL, d_jlevL, d_tail);
  cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
  ilevL[1] = tail;
  lev = 1;

  //int *jlevL = (int *) malloc(n*sizeof(int));
  
 // int nkernels;

//  nkernels = 0;
  while (tail < n) {
    int count = tail - head;

    /* how many threads needed to run 1-block kernels */
    int nthreads = count * WARP0;
    int gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;
   
    /*
    printf("Launch kernel <<<%d, %d>>>\n", gDim, BLOCKDIM);
    */

//    nkernels ++;
    if (gDim <= MIN_BLOCK) {
      LEVEL_CSC_SYNC_L_1BLOCK<<<1, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpL, d_jlevL, head, tail, d_out, d_Nout);
      int Nout;
      cudaMemcpy(&Nout, d_Nout, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(ilevL+lev+1, d_out, Nout*sizeof(int), cudaMemcpyDeviceToHost);
      lev += Nout;
      head = ilevL[lev-1];
      tail = ilevL[lev];
  
      d_tail = d_out + Nout - 1;

      //for (int i = 0; i < lev+1; i++) { printf("%d ", ilevL[i]); } printf("\n");
      //printf("Nout %d, head %d, tail %d\n", Nout, head, tail);
      //cudaMemcpy(jlevL, d_jlevL, n*sizeof(int), cudaMemcpyDeviceToHost);
      //for (int i = 0; i < n; i++) { printf("%d ", jlevL[i]); } printf("\n");

    } else {

      nthreads = count * WARP;
      gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;
      
      LEVEL_CSC_SYNC_L<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpL, d_jlevL, count, head, d_tail);
      head = tail;
      cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
      ilevL[++lev] = tail;
      
      //for (int i = 0; i < lev+1; i++) { printf("%d ", ilevL[i]); } printf("\n");
      //printf("head %d, tail %d\n", head, tail);
    }
  }
  *nlevL = lev;

  //for (int i = 0; i < lev+1; i++) {
  //  ++ilevL[i];
  //}
  
  //printf("L: nkernels %d\n", nkernels);
  /*
  cudaMemcpy(jlevL, d_jlevL, n*sizeof(int), cudaMemcpyDeviceToHost);
  printf("nlevL %d\n", *nlevL);
  for (int i = 0; i < n; i++) { printf("%d ", jlevL[i]); } printf("\n");
  for (int i = 0; i < *nlevL+1; i++) { printf("%d ", ilevL[i]); } printf("\n");
  exit(0);
  */

  // U
  ilevU[0] = 0;
  head = 0;
  cudaMemset(d_tail, 0, sizeof(int));
  gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
  LEVEL_CSC_SYNC_0<<<gDim, BLOCKDIM>>>(n, d_dpU, d_jlevU, d_tail);
  cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
  ilevU[1] = tail;
  lev = 1;

  //int *jlevU = (int *) malloc(n*sizeof(int));

 // nkernels = 0;
  while (tail < n) {
    int count = tail - head;

    /* how many threads needed to run 1-block kernels */
    int nthreads = count * WARP0;
    int gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;

  //  nkernels ++;
    if (gDim <= MIN_BLOCK) {
      LEVEL_CSC_SYNC_U_1BLOCK<<<1, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpU, d_jlevU, head, tail, d_out, d_Nout);
      int Nout;
      cudaMemcpy(&Nout, d_Nout, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(ilevU+lev+1, d_out, Nout*sizeof(int), cudaMemcpyDeviceToHost);
      lev += Nout;
      head = ilevU[lev-1];
      tail = ilevU[lev];
      d_tail = d_out + Nout - 1;
    } else {

      nthreads = count * WARP;
      gDim = (nthreads + BLOCKDIM - 1) / BLOCKDIM;

      LEVEL_CSC_SYNC_U<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_db, d_dpU, d_jlevU, count, head, d_tail);
    head = tail;
    cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost);
    ilevU[++lev] = tail;
    }
  }
  *nlevU = lev;
 
//  printf("U: nkernels %d\n", nkernels);
  /*
  cudaMemcpy(jlevU, d_jlevU, n*sizeof(int), cudaMemcpyDeviceToHost);
  printf("nlevU %d\n", *nlevU);
  for (int i = 0; i < n; i++) { printf("%d ", jlevU[i]); } printf("\n");
  for (int i = 0; i < *nlevU+1; i++) { printf("%d ", ilevU[i]); } printf("\n");
  exit(0);
  */

  //for (int i = 0; i < lev+1; i++) {
  //  ++ilevU[i];
  //}

  cudaFree(d_out);
}
