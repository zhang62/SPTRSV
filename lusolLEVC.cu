#include "lusol.h"
//#include "cusparse.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || DOUBLEPRECISION == 0
#else
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

__global__ void LU_SOL_DYNC_CSC_INIT(int n, int *ib, int *jb, int *dp);

/*-------------------------------------------*/
/*     Column level scheduling Column kernel */
/*-------------------------------------------*/
__global__
void L_SOL_LEVC16(REAL *x, REAL *aa,
                int *ja, int *ia, int *di,
                int *jlevL, int l1, int l2) {
  int i,j,jj;

  // num of half-warps
  int nhw = gridDim.x * BLOCKDIM / HALFWARP;
  // half-warp id
  int hwid = (blockIdx.x * BLOCKDIM + threadIdx.x) / HALFWARP;
  // thread lane in each half-warp
  int lane = threadIdx.x & (HALFWARP-1);
  // local half-warp id
  int hwlane = threadIdx.x / HALFWARP;
  volatile __shared__ REAL s_xjj[BLOCKDIM / HALFWARP];

  for (i = l1+hwid; i < l2; i += nhw) {
    jj = jlevL[i-1]-1;

    int p1 = di[jj];
    int q1 = ia[jj+1];

    REAL xjj, dinv;

    if (lane == 0) {
      dinv = 1.0 / aa[p1-1];
      xjj = dinv * x[jj];
      s_xjj[hwlane] = xjj;
    }

    xjj = s_xjj[hwlane];

    for (j = p1+1+lane; j < q1; j += HALFWARP) {
      int k = ja[j-1] - 1;
      atomicAdd((REAL*)&x[k], -xjj * aa[j-1]);
    }

    if (lane == 0) {
      x[jj] = xjj;
    }
  }
}

__global__
void U_SOL_LEVC16(REAL *x, REAL *aa,
                int *ja, int *ia, int *di,
                int *jlevU, int l1, int l2) {
  int i,j,jj;

  // num of half-warps
  int nhw = gridDim.x * BLOCKDIM / HALFWARP;
  // warp id
  int hwid = (blockIdx.x * BLOCKDIM + threadIdx.x) / HALFWARP;
  // thread lane in each half-warp
  int lane = threadIdx.x & (HALFWARP-1);
  // local half-warp id
  int hwlane = threadIdx.x / HALFWARP;
  volatile __shared__ REAL s_xjj[BLOCKDIM / HALFWARP];

  for (i = l1+hwid; i < l2; i += nhw) {
    jj = jlevU[i-1]-1;

    int p1 = ia[jj];
    int q1 = di[jj];

    REAL xjj, dinv;

    if (lane == 0) {
      dinv = 1.0 / aa[q1-1];
      xjj = dinv * x[jj];
      s_xjj[hwlane] = xjj;
    }

    xjj = s_xjj[hwlane];

    for (j = p1+lane; j < q1; j += HALFWARP) {
      int k = ja[j-1] - 1;
      atomicAdd((REAL*)&x[k], -xjj * aa[j-1]);
    }

    if (lane == 0) {
      x[jj] = xjj;
    }
  }
}

//--------------------------------------------------------
void luSolvLevC16(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b,
                  int REPEAT, bool print)
{
  int *d_ib, *d_jb, *d_db, *d_jlevL, *d_jlevU;
  REAL *d_b, *d_x, *d_bb;
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
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_jb, nnz*sizeof(int));
  cudaMalloc((void **)&d_db, n*sizeof(int));
  cudaMalloc((void **)&d_bb, nnz*sizeof(REAL));

/*------------------- analysis */
  csrcsc(n, n, 1, 1, csr->a, csr->ja, csr->ia, bb, jb, ib);
  diagpos(n, ib, jb, bb, db);

/*------------------- Memcpy */
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


  ta = wall_timer();
  if (!GPU_LEVEL) {
    for (int j=0; j<REPEAT; j++) {

      cudaMemcpy(ib, d_ib, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(jb, d_jb, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(db, d_db, n*sizeof(int), cudaMemcpyDeviceToHost);

      //makeLevelCSR(n, csr->ia, csr->ja, &lev);
      makeLevelCSC(n, ib, jb, db, &lev);
      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    }
  } else {
    for (int j=0; j<REPEAT; j++) {
      int *d_dp;
      int nhwb = BLOCKDIM / HALFWARP;
      int gDim = (n + nhwb-1) / nhwb;
      cudaMalloc((void **)&d_dp, 2*n*sizeof(int));
      cudaMemset(d_dp, 0, 2*n*sizeof(int));
      LU_SOL_DYNC_CSC_INIT<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_dp);
      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp,
                        d_jlevL, lev.ilevL, &lev.nlevL,
                        d_jlevU, lev.ilevU, &lev.nlevU);
      cudaFree(d_dp);
    }
  }
  ta = wall_timer() - ta;

  int ilev_shift;
  ilev_shift = lev.ilevL[0] == 0; for (int i=0; i < lev.nlevL+1; i++) { lev.ilevL[i] += ilev_shift; }
  ilev_shift = lev.ilevU[0] == 0; for (int i=0; i < lev.nlevU+1; i++) { lev.ilevU[i] += ilev_shift; }

  t1 = wall_timer();
  for (int j=0; j<REPEAT; j++) {
    // copy b to x
    cudaMemcpy(d_x, d_b, n*sizeof(REAL), cudaMemcpyDeviceToDevice);
    // L-solve
    for (int i=0; i<lev.nlevL; i++) {
      int l1 = lev.ilevL[i];
      int l2 = lev.ilevL[i+1];
      int l_size = l2 - l1;
      int nthreads = min(l_size*HALFWARP, MAXTHREADS);
      int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
      int bDim = BLOCKDIM;
      L_SOL_LEVC16<<<gDim, bDim>>>(d_x, d_bb, d_jb, d_ib, d_db, d_jlevL, l1, l2);
    }
    // U-solve
    for (int i=0; i<lev.nlevU; i++) {
       int l1 = lev.ilevU[i];
       int l2 = lev.ilevU[i+1];
       int l_size = l2 - l1;
       int nthreads = min(l_size*HALFWARP, MAXTHREADS);
       int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
       int bDim = BLOCKDIM;
       U_SOL_LEVC16<<<gDim, bDim>>>(d_x, d_bb, d_jb, d_ib, d_db, d_jlevU, l1, l2);
    }
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] level-scheduling C16, #lev in L %d, #lev in U %d\n",
           lev.nlevL, lev.nlevU);
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  free(ib);
  free(jb);
  free(bb);
  free(db);
  FreeLev(&lev);
  cudaFree(d_b);
  cudaFree(d_ib);
  cudaFree(d_jb);
  cudaFree(d_db);
  cudaFree(d_bb);
  cudaFree(d_x);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);
}


// WARP
__global__
void L_SOL_LEVC32(REAL *x, REAL *aa,
                int *ja, int *ia, int *di,
                int *jlevL, int l1, int l2) {
  int i,j,jj;

  // num of warps
  int nw = gridDim.x * BLOCKDIM / WARP;
  // warp id
  int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // local warp id
  int wlane = threadIdx.x / WARP;
  volatile __shared__ REAL s_xjj[BLOCKDIM / WARP];

  for (i = l1+wid; i < l2; i += nw) {
    jj = jlevL[i-1]-1;

    int p1 = di[jj];
    int q1 = ia[jj+1];

    REAL xjj, dinv;

    if (lane == 0) {
      dinv = 1.0 / aa[p1-1];
      xjj = dinv * x[jj];
      s_xjj[wlane] = xjj;
    }

    xjj = s_xjj[wlane];

    for (j = p1+1+lane; j < q1; j += WARP) {
      int k = ja[j-1] - 1;
      atomicAdd((REAL*)&x[k], -xjj * aa[j-1]);
    }

    if (lane == 0) {
      x[jj] = xjj;
    }
  }
}

__global__
void U_SOL_LEVC32(REAL *x, REAL *aa,
                int *ja, int *ia, int *di,
                int *jlevU, int l1, int l2) {
  int i,j,jj;

  // num of warps
  int nw = gridDim.x * BLOCKDIM / WARP;
  // warp id
  int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // local warp id
  int wlane = threadIdx.x / WARP;
  volatile __shared__ REAL s_xjj[BLOCKDIM / WARP];

  for (i = l1+wid; i < l2; i += nw) {
    jj = jlevU[i-1]-1;

    int p1 = ia[jj];
    int q1 = di[jj];

    REAL xjj, dinv;

    if (lane == 0) {
      dinv = 1.0 / aa[q1-1];
      xjj = dinv * x[jj];
      s_xjj[wlane] = xjj;
    }

    xjj = s_xjj[wlane];

    for (j = p1+lane; j < q1; j += WARP) {
      int k = ja[j-1] - 1;
      atomicAdd((REAL*)&x[k], -xjj * aa[j-1]);
    }

    if (lane == 0) {
      x[jj] = xjj;
    }
  }
}

//--------------------------------------------------------
void luSolvLevC32(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b,
                  int REPEAT, bool print)
{
  int *d_ib, *d_jb, *d_db, *d_jlevL, *d_jlevU;
  REAL *d_b, *d_x, *d_bb;
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
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));

  // CSC
  csrcsc(n, n, 1, 1, csr->a, csr->ja, csr->ia, bb, jb, ib);
  diagpos(n, ib, jb, bb, db);

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_jb, nnz*sizeof(int));
  cudaMalloc((void **)&d_db, n*sizeof(int));
  cudaMalloc((void **)&d_bb, nnz*sizeof(REAL));
/*------------------- Memcpy */
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

/*------------------- analysis */
  ta = wall_timer();
  if (!GPU_LEVEL) {
    for (int j=0; j<REPEAT; j++) {

      cudaMemcpy(ib, d_ib, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(jb, d_jb, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(db, d_db, n*sizeof(int), cudaMemcpyDeviceToHost);

      //makeLevelCSR(n, csr->ia, csr->ja, &lev);
      makeLevelCSC(n, ib, jb, db, &lev);
      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    }
  } else {
    for (int j=0; j<REPEAT; j++) {
      int *d_dp;
      int nhwb = BLOCKDIM / HALFWARP;
      int gDim = (n + nhwb-1) / nhwb;
      cudaMalloc((void **)&d_dp, 2*n*sizeof(int));
      cudaMemset(d_dp, 0, 2*n*sizeof(int));
      LU_SOL_DYNC_CSC_INIT<<<gDim, BLOCKDIM>>>(n, d_ib, d_jb, d_dp);
      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp,
                        d_jlevL, lev.ilevL, &lev.nlevL,
                        d_jlevU, lev.ilevU, &lev.nlevU);
      cudaFree(d_dp);
    }
  }
  ta = wall_timer() - ta;

  int ilev_shift;
  ilev_shift = lev.ilevL[0] == 0; for (int i=0; i < lev.nlevL+1; i++) { lev.ilevL[i] += ilev_shift; }
  ilev_shift = lev.ilevU[0] == 0; for (int i=0; i < lev.nlevU+1; i++) { lev.ilevU[i] += ilev_shift; }

  t1 = wall_timer();
  for (int j=0; j<REPEAT; j++) {
    // copy b to x
    cudaMemcpy(d_x, d_b, n*sizeof(REAL), cudaMemcpyDeviceToDevice);
    // L-solve
    for (int i=0; i<lev.nlevL; i++) {
      int l1 = lev.ilevL[i];
      int l2 = lev.ilevL[i+1];
      int l_size = l2 - l1;
      int nthreads = min(l_size*WARP, MAXTHREADS);
      int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
      int bDim = BLOCKDIM;
      L_SOL_LEVC32<<<gDim, bDim>>>(d_x, d_bb, d_jb, d_ib, d_db, d_jlevL, l1, l2);
    }
    // U-solve
    for (int i=0; i<lev.nlevU; i++) {
       int l1 = lev.ilevU[i];
       int l2 = lev.ilevU[i+1];
       int l_size = l2 - l1;
       int nthreads = min(l_size*WARP, MAXTHREADS);
       int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
       int bDim = BLOCKDIM;
       U_SOL_LEVC32<<<gDim, bDim>>>(d_x, d_bb, d_jb, d_ib, d_db, d_jlevU, l1, l2);
    }
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] level-scheduling C32, #lev in L %d, #lev in U %d\n",
           lev.nlevL, lev.nlevU);
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  free(ib);
  free(jb);
  free(bb);
  free(db);
  FreeLev(&lev);
  cudaFree(d_b);
  cudaFree(d_ib);
  cudaFree(d_jb);
  cudaFree(d_db);
  cudaFree(d_bb);
  cudaFree(d_x);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);
}

