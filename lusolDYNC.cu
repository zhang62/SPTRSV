#include "lusol.h"
//#include "cusparse.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 && DOUBLEPRECISION == 1
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/*
__global__
void LU_SOL_DYNC_INIT(int n, int *ia, int *da, int *dpl, int *dpu,
                     REAL* x, REAL *b) {
  int gid = blockIdx.x * BLOCKDIM + threadIdx.x;
  if (gid < n) {
    int t = da[gid];
    dpl[gid] = t - ia[gid];
    dpu[gid] = ia[gid+1] - t - 1;
    x[gid] = b[gid];
  }
}
*/

__global__
void LU_SOL_DYNC_CSC_INIT(int n, int *ib, int *jb, int *dp) {
  int *dpl = dp;
  int *dpu = dp + n;
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  int lane = threadIdx.x & (HALFWARP-1);

  if (hwid >= n) {
    return;
  }

  int p1 = ib[hwid];
  int q1 = ib[hwid+1];
  for (int i=p1+lane; i<q1; i+=HALFWARP) {
    int row = jb[i-1] - 1;
    if (row < hwid) {
      atomicAdd(&dpu[row], 1);
    } else if (row > hwid) {
      atomicAdd(&dpl[row], 1);
    }
  }
}

__global__
void L_SOL_DYNC(int n,
               REAL *x,
               const int * __restrict__ jb,
               const int * __restrict__ ib,
               const REAL* __restrict__ bb,
               const int * __restrict__ db,
                     int *              dp,
                     int *jlev) {
  // global warp id
  const int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // local warp id
  const int wlane = threadIdx.x / WARP;
  // global warp id of the first warp in this block
  //const int fid = blockIdx.x * BLOCKDIM / WARP;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  //volatile __shared__ int  s_dp[BLOCKDIM / WARP];
  volatile __shared__ REAL s_dx[BLOCKDIM / WARP];
  // make dp volatile to tell compiler do not use cached value
  volatile int *vdp = dp;
  volatile REAL *vx = x;

  if (wid >= n) {
    return;
  }

  /*
  if (lane == 0) {
    s_dp[wlane] = 0;
    s_dx[wlane] = 0.0;
  }
  __syncthreads();
  */

  //int i = wid;
  int i = jlev[wid] - 1;
  int q1 = db[i];
  int q2 = ib[i+1];
  REAL dinv, xi;

  if (lane == 0) {
    dinv = 1.0 / bb[q1-1];

    // busy waiting
    //while (s_dp[wlane] != vdp[i]);
    while (0 != vdp[i]);

    //REAL xi = dinv * (vx[i] - s_dx[wlane]);
    REAL xi = dinv * vx[i];
    s_dx[wlane] = xi;
    vx[i] = xi;
  }

  xi = s_dx[wlane];

//  if (lane == 0) {
//    x[i] = xi;
//  }

  for (int j = q1+1+lane; j < q2; j += WARP) {
    int k = jb[j-1]-1;
    /*
    if (k < fid + BLOCKDIM / WARP) {
      atomicAdd((REAL *)&s_dx[k-fid], xi * bb[j-1]);
      atomicAdd(( int *)&s_dp[k-fid], 1);
    }
    else*/ {
      //atomicAdd((REAL*)&vx[k], -xi * bb[j-1]);
      atomicAdd(&x[k], -xi * bb[j-1]);
      __threadfence();
      atomicSub(&dp[k], 1);
    }
  }
  //if (lane == 0) {
  //  vx[i] = xi;
  //}
}

__global__
void U_SOL_DYNC(int n,
               REAL *x,
               const int * __restrict__ jb,
               const int * __restrict__ ib,
               const REAL* __restrict__ bb,
               const int * __restrict__ db,
                     int *              dp,
                     int *jlev) {
  // global warp id
  const int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // local warp id
  const int wlane = threadIdx.x / WARP;
  // global warp id of the first warp in this block
  //const int fid = blockIdx.x * BLOCKDIM / WARP;
  // thread lane in each warp
  const int lane = threadIdx.x & (WARP - 1);
  //volatile __shared__ int  s_dp[BLOCKDIM / WARP];
  volatile __shared__ REAL s_dx[BLOCKDIM / WARP];
  // make dp volatile to tell compiler do not use cached value
  volatile int *vdp = dp;
  volatile REAL *vx = x;

  if (wid >= n) {
    return;
  }

  /*
  if (lane == 0) {
    s_dp[wlane] = 0;
    s_dx[wlane] = 0.0;
  }
  __syncthreads();
  */

  //int i = wid;
  int i = jlev[wid] - 1;
  int q1 = ib[i];
  int q2 = db[i];
  REAL dinv, xi;

  if (lane == 0) {
    dinv = 1.0 / bb[q2-1];

    // busy waiting
    //while (s_dp[wlane] != vdp[i]);
    while (0 != vdp[i]);

    //REAL xi = dinv * (vx[i] - s_dx[wlane]);
    REAL xi = dinv * vx[i];
    s_dx[wlane] = xi;
    vx[i] = xi;
  }

  xi = s_dx[wlane];

//  if (lane == 0) {
//    x[i] = xi;
//  }

  for (int j = q1+lane; j < q2; j += WARP) {
    int k = jb[j-1]-1;
    /*
    if (k < fid + BLOCKDIM / WARP) {
      atomicAdd((REAL *)&s_dx[k-fid], xi * bb[j-1]);
      atomicAdd(( int *)&s_dp[k-fid], 1);
    }
    else*/ {
      //atomicAdd((REAL *)&vx[k], -xi * bb[j-1]);
      atomicAdd(&x[k], -xi * bb[j-1]);
      __threadfence();
      atomicSub(&dp[k], 1);
    }
  }
}


//--------------------------------------------------------
void luSolvDYNC(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b,
                int REPEAT, bool print)
{
  int j, *d_ia, *d_ja, *d_da, *d_ib, *d_jb, *d_db, *d_dp, *d_dpl,
      *d_dpu, *d_jlevL, *d_jlevU, *d_dp_saved;
  REAL *d_a, *d_b, *d_x, *d_bb;
  double t1, t2;

  REAL *bb;
  int *ib, *jb, *db;
  struct level_t lev;
  double ta;
  ib = (int *) malloc((n+1)*sizeof(int));
  jb = (int *) malloc(nnz*sizeof(int));
  bb = (REAL *) malloc(nnz*sizeof(REAL));
  db = (int *) malloc(n*sizeof(int));
  allocLevel(n, &lev);

  csrcsc(n, n, 1, 1, csr->a, csr->ja, csr->ia, bb, jb, ib);
  diagpos(n, ib, jb, bb, db);

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_da, n*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_jb, nnz*sizeof(int));
  cudaMalloc((void **)&d_db, n*sizeof(int));
  cudaMalloc((void **)&d_bb, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_dp, 2*n*sizeof(int));
  cudaMalloc((void **)&d_dp_saved, 2*n*sizeof(int));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_da, csr->di, n*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ib, ib, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_jb, jb, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_bb, bb, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_db, db, n*sizeof(int),
  cudaMemcpyHostToDevice);

#if 0
  {
    double tt = wall_timer();

    for (int j=0; j<REPEAT; j++) {
      int nhwb = BLOCKDIM / HALFWARP;
      int gDim = (n + nhwb-1) / nhwb;
      cudaMemset(d_dp, 0, 2*n*sizeof(int));
      LU_SOL_DYNC_CSC_INIT<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_dp);
      cudaMemcpy(d_dp_saved, d_dp, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
      makeTopoCSC(n, d_ib, d_jb, d_db, d_dp, d_jlevL, d_jlevU);
    }
    cudaThreadSynchronize();
    tt = wall_timer() - tt;
    printf("  analysis time TOPOLOGY %f\n", tt/REPEAT);
    checktopo(n, ib, jb, db, d_jlevL, d_jlevU, d_dp_saved);
    exit(0);
  }
#endif

/*------------------- analysis */
  double ta2 = wall_timer();
  for (int j=0; j<REPEAT; j++) {
    int nhwb = BLOCKDIM / HALFWARP;
    int gDim = (n + nhwb-1) / nhwb;
    cudaMemset(d_dp, 0, 2*n*sizeof(int));
    LU_SOL_DYNC_CSC_INIT<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_dp);
    cudaMemcpy(d_dp_saved, d_dp, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
  }
  cudaThreadSynchronize();
  ta2 = wall_timer() - ta2;

  ta = wall_timer();
  for (int j=0; j<REPEAT; j++) {
    //makeLevelCSR(n, csr->ia, csr->ja, csr->di, &lev);
    int nhwb = BLOCKDIM / HALFWARP;
    int gDim = (n + nhwb-1) / nhwb;
    cudaMemset(d_dp, 0, 2*n*sizeof(int));
    LU_SOL_DYNC_CSC_INIT<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_dp);
    cudaMemcpy(d_dp_saved, d_dp, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
    if (!GPU_LEVEL) {

      cudaMemcpy(ib, d_ib, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(jb, d_jb, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(db, d_db, n*sizeof(int), cudaMemcpyDeviceToHost);

      makeLevelCSC(n, ib, jb, db, &lev);

      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    } else {
      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp,
                        d_jlevL, lev.ilevL, &lev.nlevL,
                        d_jlevU, lev.ilevU, &lev.nlevU);
    }
  }
  cudaThreadSynchronize();
  ta = wall_timer() - ta;


  d_dpl = d_dp;
  d_dpu = d_dp + n;

#if 0
/*------------------- GPU analysis */
  double ta3 = wall_timer();
  makeLevelCSC_SYNC_L(n, d_ib, d_jb, d_db, d_dpl, d_jlevL);
  cudaThreadSynchronize();
  ta3 = wall_timer() - ta3;
  printf("  analysis time         %f\n", ta/REPEAT);
  printf("  analysis time for GSF %f\n", ta2/REPEAT);
  printf("  analysis time for GPU %f\n", ta3);
  cuda_check_err();
  exit(0);
#endif

  int *tmp1 = (int*) malloc(n*sizeof(int));
  int *tmp2 = (int*) malloc(n*sizeof(int));

  t1 = wall_timer();
  for (j=0; j<REPEAT; j++) {
    int bDim = BLOCKDIM;
    /*-------- init dependent counter of L and U */
    //int gDim0 = (n + BLOCKDIM - 1) / BLOCKDIM;
    //LU_SOL_DYNC_INIT<<<gDim0, bDim>>>(n, d_ia, d_da, d_dpl, d_dpu,
    //                                 d_x, d_b);
    cudaMemcpy(d_dp, d_dp_saved, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_x, d_b, n*sizeof(REAL), cudaMemcpyDeviceToDevice);
    /*-------- num of warps per block */
    int nwb = BLOCKDIM / WARP;
    int gDim = (n + nwb-1) / nwb;
    // L-solve
    L_SOL_DYNC<<<gDim, bDim>>>(n, d_x, d_jb, d_ib, d_bb, d_db, d_dpl, d_jlevL);
    // U-solve
    U_SOL_DYNC<<<gDim, bDim>>>(n, d_x, d_jb, d_ib, d_bb, d_db, d_dpu, d_jlevU);
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  cudaMemcpy(tmp1, d_dpl, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp2, d_dpu, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    if (tmp1[i]) { printf("i=%d: %d\n", i, tmp1[i]); break; }
    if (tmp2[i]) { printf("i=%d: %d\n", i, tmp2[i]); break; }
  }

  if (print) {
    printf("[GPU] DYNC\n");
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
    printf("  analysis time for GSF %f ", ta2/REPEAT);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  free(ib);
  free(jb);
  free(bb);
  free(db);
  FreeLev(&lev);
  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_da);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ib);
  cudaFree(d_jb);
  cudaFree(d_db);
  cudaFree(d_bb);
  cudaFree(d_x);
  cudaFree(d_dp);
  cudaFree(d_dp_saved);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);

  free(tmp1);
  free(tmp2);
}


