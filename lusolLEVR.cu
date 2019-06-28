#include "cusparse.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "lusol.h"

__global__
void LU_SOL_DYNR_INIT(int n, int *ia, int *da, int *dpl, int *dpu);
__global__ void DYNR_ANA_1(int n, int *ia, int *ii);
__global__ void DYNR_ANA_2(int n, int nnz, int *jb, int *jj, int *ib, int *db);

/*-------------------------------------------*/
/*     Row level scheduling kernel           */
/*-------------------------------------------*/

// HALF-WARP
__global__
void L_SOL_LEVR16(REAL *b, REAL *x, REAL *a,
               int *ja, int *ia, int *di,
               int *jlevL, int l1, int l2) {
  int i,k,jj;

  // num of half-warps
  int nhw = gridDim.x*BLOCKDIM/HALFWARP;
  // half warp id
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  // thread lane in each half warp
  int lane = threadIdx.x & (HALFWARP-1);
  // shared memory for patial result
  volatile __shared__ REAL r[BLOCKDIM+8];

  for (i=l1+hwid; i<l2; i+=nhw) {
    jj = jlevL[i-1]-1;
    int p1 = ia[jj];
    int q1 = di[jj];

    REAL sum = 0.0;
    for (k=p1+lane; k<q1; k+=HALFWARP)
      sum += a[k-1]*x[ja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0) {
      REAL t = 1.0 / a[q1-1];
      x[jj] = t*(b[jj] - r[threadIdx.x]);
    }
  }
}

/*----------------- x = U^{-1}*x */
__global__
void U_SOL_LEVR16(REAL *x, REAL *a,
               int *ja, int *ia, int *di,
               int *jlevU, int l1, int l2) {
  int i,k,jj;

  // num of half-warps
  int nhw = gridDim.x*BLOCKDIM/HALFWARP;
  // half warp id
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  // thread lane in each half warp
  int lane = threadIdx.x & (HALFWARP-1);
  // shared memory for patial result
  volatile __shared__ REAL r[BLOCKDIM+8];

  for (i=l1+hwid; i<l2; i+=nhw) {
    jj = jlevU[i-1]-1;
    int p1 = di[jj];
    int q1 = ia[jj+1];

    REAL sum = 0.0;
    for (k=p1+1+lane; k<q1; k+=HALFWARP)
      sum += a[k-1]*x[ja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0) {
      REAL t = 1.0 / a[p1-1];
      x[jj] = t*(x[jj] - r[threadIdx.x]);
    }
  }
}

//--------------------------------------------------------
void luSolvLevR16(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b,
                  int REPEAT, bool print)
{
  int i, j, *d_ia, *d_ja, *d_di, *d_jlevL, *d_jlevU;
  REAL *d_a, *d_b, *d_x;
  double t1, t2, ta;
  struct level_t lev;

  allocLevel(n, &lev);
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_di, n*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_di, csr->di, n*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL),
  cudaMemcpyHostToDevice);

/*------------------- analysis */
  ta = wall_timer();
  if (!GPU_LEVEL) {
    for (int j=0; j<REPEAT; j++) {

      cudaMemcpy(csr->ia, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->ja, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->di, d_di, n*sizeof(int), cudaMemcpyDeviceToHost);

      makeLevelCSR(n, csr->ia, csr->ja, csr->di, &lev);
      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    }
  } else {
    for (int j=0; j<REPEAT; j++) {
      int *d_dp, *d_ib, *d_jb, *d_db;
      cudaMalloc((void **)&d_dp, 2*n*sizeof(int));

      cudaMalloc((void **)&d_jb, nnz*sizeof(int));
      cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
      cudaMalloc((void **)&d_db, n*sizeof(int));

      int gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
      LU_SOL_DYNR_INIT<<<gDim, BLOCKDIM>>>(n, d_ia, d_di, d_dp, d_dp+n);

      int nhwb = BLOCKDIM / HALFWARP;  // number of half-warps per block
      gDim = (n + nhwb - 1) / nhwb;
      DYNR_ANA_1<<<gDim, BLOCKDIM>>>(n, d_ia, d_jb);
      //wrap raw pointer with a device_ptr to use with Thrust functions
      thrust::device_ptr<int> dev_data(d_jb);
      thrust::device_ptr<int> dev_keys(d_ja);
      thrust::stable_sort_by_key(dev_keys, dev_keys + nnz, dev_data);
      gDim = (nnz + BLOCKDIM-1) / BLOCKDIM;
      DYNR_ANA_2<<<gDim, BLOCKDIM>>>(n, nnz, d_jb, d_ja, d_ib, d_db);

      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp,
          d_jlevL, lev.ilevL, &lev.nlevL,
          d_jlevU, lev.ilevU, &lev.nlevU);

      // copy again since it was changed by the sorting
      cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int), cudaMemcpyHostToDevice);

      cudaFree(d_dp);
      cudaFree(d_ib);
      cudaFree(d_jb);
      cudaFree(d_db);
    }
  }
  ta = wall_timer() - ta;

  t1 = wall_timer();

  int ilev_shift;
  ilev_shift = lev.ilevL[0] == 0; for (int i=0; i < lev.nlevL+1; i++) { lev.ilevL[i] += ilev_shift; }
  ilev_shift = lev.ilevU[0] == 0; for (int i=0; i < lev.nlevU+1; i++) { lev.ilevU[i] += ilev_shift; }

  for (j=0; j<REPEAT; j++) {
    // L-solve
    for (i=0; i<lev.nlevL; i++) {
      int l1 = lev.ilevL[i];
      int l2 = lev.ilevL[i+1];
      int l_size = l2 - l1;
      int nthreads = min(l_size*HALFWARP, MAXTHREADS);
      int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
      int bDim = BLOCKDIM;
      L_SOL_LEVR16<<<gDim, bDim>>>(d_b, d_x, d_a, d_ja, d_ia, d_di, d_jlevL, l1, l2);
    }
    // U-solve
    for (i=0; i<lev.nlevU; i++) {
       int l1 = lev.ilevU[i];
       int l2 = lev.ilevU[i+1];
       int l_size = l2 - l1;
       int nthreads = min(l_size*HALFWARP, MAXTHREADS);
       int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
       int bDim = BLOCKDIM;
       U_SOL_LEVR16<<<gDim, bDim>>>(d_x, d_a, d_ja, d_ia, d_di, d_jlevU, l1, l2);
    }
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] level-scheduling R16, #lev in L %d, #lev in U %d\n",
           lev.nlevL, lev.nlevU);
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_di);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_x);
  FreeLev(&lev);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);
}

// WARP
__global__
void L_SOL_LEVR32(REAL *b, REAL *x, REAL *a,
               int *ja, int *ia, int *di,
               int *jlevL, int l1, int l2) {
  int i,k,jj;

  // num of warps
  int nw = gridDim.x*BLOCKDIM/WARP;
  // warp id
  int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // shared memory for patial result
  volatile __shared__ REAL r[BLOCKDIM+16];

  for (i=l1+wid; i<l2; i+=nw) {
    jj = jlevL[i-1]-1;
    int p1 = ia[jj];
    int q1 = di[jj];

    REAL sum = 0.0;
    for (k=p1+lane; k<q1; k+=WARP)
      sum += a[k-1]*x[ja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+16];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0) {
      REAL t = 1.0 / a[q1-1];
      x[jj] = t*(b[jj] - r[threadIdx.x]);
    }
  }
}

/*----------------- x = U^{-1}*x */
__global__
void U_SOL_LEVR32(REAL *x, REAL *a,
               int *ja, int *ia, int *di,
               int *jlevU, int l1, int l2) {
  int i,k,jj;

  // num of warps
  int nw = gridDim.x*BLOCKDIM/WARP;
  // warp id
  int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // shared memory for patial result
  volatile __shared__ REAL r[BLOCKDIM+16];

  for (i=l1+wid; i<l2; i+=nw) {
    jj = jlevU[i-1]-1;
    int p1 = di[jj];
    int q1 = ia[jj+1];

    REAL sum = 0.0;
    for (k=p1+1+lane; k<q1; k+=WARP)
      sum += a[k-1]*x[ja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+16];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0) {
      REAL t = 1.0 / a[p1-1];
      x[jj] = t*(x[jj] - r[threadIdx.x]);
    }
  }
}

//--------------------------------------------------------
void luSolvLevR32(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b,
                  int REPEAT, bool print)
{
  int i, j, *d_ia, *d_ja, *d_di, *d_jlevL, *d_jlevU;
  REAL *d_a, *d_b, *d_x;
  double t1, t2, ta;
  struct level_t lev;

  allocLevel(n, &lev);
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_di, n*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_di, csr->di, n*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL),
  cudaMemcpyHostToDevice);

/*------------------- analysis */
  ta = wall_timer();
  if (!GPU_LEVEL) {
    for (int j=0; j<REPEAT; j++) {

      cudaMemcpy(csr->ia, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->ja, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->di, d_di, n*sizeof(int), cudaMemcpyDeviceToHost);

      makeLevelCSR(n, csr->ia, csr->ja, csr->di, &lev);
      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    }
  } else {
    for (int j=0; j<REPEAT; j++) {
      int *d_dp, *d_ib, *d_jb, *d_db;
      cudaMalloc((void **)&d_dp, 2*n*sizeof(int));

      cudaMalloc((void **)&d_jb, nnz*sizeof(int));
      cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
      cudaMalloc((void **)&d_db, n*sizeof(int));

      int gDim = (n + BLOCKDIM - 1) / BLOCKDIM;
      LU_SOL_DYNR_INIT<<<gDim, BLOCKDIM>>>(n, d_ia, d_di, d_dp, d_dp+n);

      int nhwb = BLOCKDIM / HALFWARP;  // number of half-warps per block
      gDim = (n + nhwb - 1) / nhwb;
      DYNR_ANA_1<<<gDim, BLOCKDIM>>>(n, d_ia, d_jb);
      //wrap raw pointer with a device_ptr to use with Thrust functions
      thrust::device_ptr<int> dev_data(d_jb);
      thrust::device_ptr<int> dev_keys(d_ja);
      thrust::stable_sort_by_key(dev_keys, dev_keys + nnz, dev_data);
      gDim = (nnz + BLOCKDIM-1) / BLOCKDIM;
      DYNR_ANA_2<<<gDim, BLOCKDIM>>>(n, nnz, d_jb, d_ja, d_ib, d_db);

     // double tt1 = wall_timer();
      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp,
          d_jlevL, lev.ilevL, &lev.nlevL,
          d_jlevU, lev.ilevU, &lev.nlevU);

     // cudaThreadSynchronize();
     // tt1 = wall_timer() - tt1;
     // printf("GPU LEVEL TIME %f\n", tt1);

      // copy again since it was changed by the sorting
      cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int), cudaMemcpyHostToDevice);

      cudaFree(d_dp);
      cudaFree(d_ib);
      cudaFree(d_jb);
      cudaFree(d_db);
    }
  }
  ta = wall_timer() - ta;

  int ilev_shift;
  ilev_shift = lev.ilevL[0] == 0; for (int i=0; i < lev.nlevL+1; i++) { lev.ilevL[i] += ilev_shift; }
  ilev_shift = lev.ilevU[0] == 0; for (int i=0; i < lev.nlevU+1; i++) { lev.ilevU[i] += ilev_shift; }

  t1 = wall_timer();
  for (j=0; j<REPEAT; j++) {
    // L-solve
    for (i=0; i<lev.nlevL; i++) {
      int l1 = lev.ilevL[i];
      int l2 = lev.ilevL[i+1];
      int l_size = l2 - l1;
      int nthreads = min(l_size*WARP, MAXTHREADS);
      int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
      int bDim = BLOCKDIM;
      L_SOL_LEVR32<<<gDim, bDim>>>(d_b, d_x, d_a, d_ja, d_ia, d_di, d_jlevL, l1, l2);
    }
    // U-solve
    for (i=0; i<lev.nlevU; i++) {
       int l1 = lev.ilevU[i];
       int l2 = lev.ilevU[i+1];
       int l_size = l2 - l1;
       int nthreads = min(l_size*WARP, MAXTHREADS);
       int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
       int bDim = BLOCKDIM;
       U_SOL_LEVR32<<<gDim, bDim>>>(d_x, d_a, d_ja, d_ia, d_di, d_jlevU, l1, l2);
    }
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] level-scheduling R32, #lev in L %d, #lev in U %d\n",
           lev.nlevL, lev.nlevU);
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
  }
  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_di);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_x);
  FreeLev(&lev);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);
}


