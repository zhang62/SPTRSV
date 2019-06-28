#include "cusparse.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "lusol.h"

__global__
void LU_SOL_DYNR_INIT(int n, int *ia, int *da, int *dpl, int *dpu) {
  int gid = blockIdx.x * BLOCKDIM + threadIdx.x;
  if (gid < n) {
    int t = da[gid];
    dpl[gid] = t - ia[gid];
    dpu[gid] = ia[gid+1] - t - 1;
  }
}

__global__
void L_SOL_DYNR(int n, REAL *b, REAL *x, REAL *a, int *ja, int *ia, int *da,
                int *jb, int *ib, int *db, int *dp, int *jlev) {
  // num of warps in grid
  //int nw = gridDim.x * BLOCKDIM / WARP;
  // global warp id
  int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // first warp in this block
  //int fid = (blockIdx.x * BLOCKDIM) / WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // shared memory for patial result in reduction
  volatile __shared__ REAL r[BLOCKDIM + 16];
  // local warp id
  //int wlane = threadIdx.x / WARP;
  // make dp volatile to tell compiler do not use cached value
  volatile int *vdp = dp;
  
  if (wid >= n) {
    return;
  }

  // this warp works on row i
  int i, p1, q1, p2, q2;
  REAL dinv, bi, sum;
  i = jlev[wid] - 1;
  p1 = ia[i];
  q1 = da[i];
  p2 = db[i] + 1;
  q2 = ib[i+1];
  sum = 0.0;

  // busy waiting
  if (lane == 0) {
    dinv = 1.0 / a[q1-1];
    bi = b[i];
    while (1) {
      // check dependent counts of this row
      // if it is free to go
      if (vdp[i] == 0) {
        break;
      } 
    }
  }

  for (int k=p1+lane; k<q1; k+=WARP) {
    sum += a[k-1]*x[ja[k-1]-1];
  }

  // parallel reduction
  r[threadIdx.x] = sum;
  r[threadIdx.x] = sum = sum + r[threadIdx.x+16];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

  // save the result
  if (lane == 0) {
    x[i] = dinv * (bi - r[threadIdx.x]);
        
    __threadfence();
        
    // reset counter
    //dp[i] = q1 - p1;
  }

  /* remove i from other's dependents */
  for (int k=p2+lane; k<q2; k+=WARP) {
    int s1 = jb[k-1]-1;
    /*
    if (s1 < fid + BLOCKDIM / WARP && 0) {
      int *p = (int*) &spcount[s1 - fid];
      atomicAdd(p, 1);
    } else {
    */
      int *p = dp + s1;
      atomicSub(p, 1);
    //}
  }
}

__global__
void U_SOL_DYNR(int n, REAL *b, REAL *x, REAL *a, int *ja, int *ia, int *da,
                int *jb, int *ib, int *db, int *dp, int *jlev) {
  // num of warps in grid
  //int nw = gridDim.x*BLOCKDIM/WARP;
  // global warp id
  int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / WARP;
  // first warp in this block
  //int fid = (blockIdx.x * BLOCKDIM) / WARP;
  // thread lane in each warp
  int lane = threadIdx.x & (WARP-1);
  // shared memory for patial result
  volatile __shared__ REAL r[BLOCKDIM+16];
  // local warp id
  //int wlane = threadIdx.x / WARP;
  // make dp volatile to tell compiler do not use cached value
  volatile int *vdp = dp;
  volatile REAL *vb = b;

  if (wid >= n) {
    return;
  }
 
  // this warp works on row i
  int i, p1, q1, p2, q2;
  REAL dinv, bi, sum;
  i = jlev[wid] - 1;
  p1 = da[i];
  q1 = ia[i+1];
  p2 = ib[i];
  q2 = db[i];
  sum = 0.0;

  // busy waiting
  if (lane == 0) {
    dinv = 1.0 / a[p1-1];
    bi = vb[i];
    while (1) {
      // if it is free to go
      if (vdp[i] == 0) {
        break;
      } 
    }
  }

  for (int k=p1+1+lane; k<q1; k+=WARP) {
    sum += a[k-1]*x[ja[k-1]-1];
  }

  // parallel reduction
  r[threadIdx.x] = sum;
  r[threadIdx.x] = sum = sum + r[threadIdx.x+16];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
  r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

  // save the result
  if (lane == 0) {
    x[i] = dinv * (bi - r[threadIdx.x]);
        
    __threadfence();
        
    // reset counter
    //dp[i] = q1 - p1 - 1;
  }

  for (int k=p2+lane; k<q2; k+=WARP) {
    int s1 = jb[k-1]-1;
    /*
    if (n-1-s1 < fid + BLOCKDIM / WARP && 0) {
      int *p = (int *) &spcount[n-1-s1-fid];
      atomicAdd(p, 1);
    } else {
    */
      int *p = dp + s1;
      atomicSub(p, 1);
    //}
  }
}

/* kernel for the analysis phase of DYNR */
__global__ void DYNR_ANA_1(int n, int *ia, int *ii) {
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  int lane = threadIdx.x & (HALFWARP-1);
  if (hwid >= n) {
    return;
  }
  int p1 = ia[hwid];
  int q1 = ia[hwid+1];
  for (int i=p1+lane; i<q1; i+=HALFWARP) {
    ii[i-1] = hwid + 1;
  }
}

__global__ void DYNR_ANA_2(int n, int nnz, int *jb, int *jj, int *ib, int *db) {
  int tid = (blockIdx.x*BLOCKDIM+threadIdx.x);
  if (tid >= nnz) {
    return;
  }
  int i1 = jb[tid];
  int j1 = jj[tid];
  
  // diagonal entry
  if (i1 == j1) {
    db[i1-1] = tid + 1;
  }

  if (tid == 0) {
    ib[0] = 1;
    ib[n] = nnz + 1;
    return;
  }

  int j0 = jj[tid-1];

  if (j1 != j0) {
    ib[j1-1] = tid + 1;
  }
}


//--------------------------------------------------------
void luSolvDYNR(int n, int nnz, struct csr_t *csr, REAL *x, REAL *b, 
                int REPEAT, bool print)
{
  int j, *d_ia, *d_ja, *d_da, *d_ib, *d_jb, *d_db, *d_dp, *d_dp_saved, *d_dpl,
      *d_dpu, *d_jlevL, *d_jlevU;
  REAL *d_a, *d_b, *d_x;
  double t1, t2;
  struct level_t lev;
  double ta;
 
  cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_jb, nnz*sizeof(int));
  cudaMalloc((void **)&d_db, n*sizeof(int));

  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int), cudaMemcpyHostToDevice);

  allocLevel(n, &lev);
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));

  cudaMalloc((void **)&d_dp, 2*n*sizeof(int));
  cudaMalloc((void **)&d_dp_saved, 2*n*sizeof(int));
  d_dpl = d_dp;
  d_dpu = d_dp + n;

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_da, n*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));

/*------------------- Memcpy */
  cudaMemcpy(d_da, csr->di, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL), cudaMemcpyHostToDevice);

/*------------------- analysis */
#if 0
  int *ib, *jb, *db;
  ib = (int *) malloc((n+1)*sizeof(int));
  jb = (int *) malloc(nnz*sizeof(int));
  db = (int *) malloc(n*sizeof(int));
  
  ta = wall_timer();
  csrcsc(n, n, 0, 1, NULL, csr->ja, csr->ia, NULL, jb, ib);
  diagpos(n, ib, jb, db);
  makeLevel(csr, &lev);
  cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int),
  cudaMemcpyHostToDevice);
  ta = wall_timer() - ta;
  
  cudaMemcpy(d_ib, ib, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_jb, jb, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_db, db, n*sizeof(int), cudaMemcpyHostToDevice);
  free(ib);
  free(jb);
  free(db);
#else

  /*
  int *ib, *jb, *db;
  ib = (int *) malloc((n+1)*sizeof(int));
  jb = (int *) malloc(nnz*sizeof(int));
  db = (int *) malloc(n*sizeof(int));
  csrcsc(n, n, 0, 1, NULL, csr->ja, csr->ia, NULL, jb, ib);
  diagpos(n, ib, jb, db);
  */

  ta = wall_timer();
  for (int j=0; j<REPEAT; j++) {
    /*-------- init dependent counter of L and U */
    int gDim0 = (n + BLOCKDIM - 1) / BLOCKDIM;
    int bDim = BLOCKDIM;
    LU_SOL_DYNR_INIT<<<gDim0, bDim>>>(n, d_ia, d_da, d_dpl, d_dpu);
    cudaMemcpy(d_dp_saved, d_dp, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
    
    int nhwb = BLOCKDIM / HALFWARP;  // number of half-warps per block
    int gDim = (n + nhwb - 1) / nhwb;
    DYNR_ANA_1<<<gDim, BLOCKDIM>>>(n, d_ia, d_jb);
    //wrap raw pointer with a device_ptr to use with Thrust functions
    thrust::device_ptr<int> dev_data(d_jb);
    thrust::device_ptr<int> dev_keys(d_ja);
    thrust::stable_sort_by_key(dev_keys, dev_keys + nnz, dev_data);
    gDim = (nnz + BLOCKDIM-1) / BLOCKDIM;
    DYNR_ANA_2<<<gDim, BLOCKDIM>>>(n, nnz, d_jb, d_ja, d_ib, d_db);
    if (!GPU_LEVEL) {

      makeLevelCSR(n, csr->ia, csr->ja, csr->di, &lev);

      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
    } else {
      makeLevelCSC_SYNC(n, d_ib, d_jb, d_db, d_dp, 
                        d_jlevL, lev.ilevL, &lev.nlevL,
                        d_jlevU, lev.ilevU, &lev.nlevU);
    }
    // copy again since it was changed by the sorting
    cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int), cudaMemcpyHostToDevice);

    if (!GPU_LEVEL) { /* add the cost of memcpy */
      cudaMemcpy(csr->ia, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->ja, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->di, d_da, n*sizeof(int), cudaMemcpyDeviceToHost);
    }
  }
  ta = wall_timer() - ta;

  /*
  int *ib2, *jb2, *db2;
  ib2 = (int *) malloc((n+1)*sizeof(int));
  jb2 = (int *) malloc(nnz*sizeof(int));
  db2 = (int *) malloc(n*sizeof(int));
  cudaMemcpy(ib2, d_ib, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(jb2, d_jb, nnz*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(db2, d_db, n*sizeof(int), cudaMemcpyDeviceToHost);
  
  for (int i=0; i<n+1; i++) {
    if (ib[i] != ib2[i]) {
      printf("!!! ib: i %d, %d %d\n", i, ib[i], ib2[i]);
      exit(0);
    }
  }
  for (int i=0; i<nnz; i++) {
    if (jb[i] != jb2[i]) {
      printf("!!! jb: i %d, %d %d\n", i, jb[i], jb2[i]);
      exit(0);
    }
  }
  for (int i=0; i<n; i++) {
    if (db[i] != db2[i]) {
      printf("!!! db: i %d, %d %d\n", i, db[i], db2[i]);
      exit(0);
    }
  }
  free(ib2);
  free(jb2);
  free(db2);
  */
#endif


  int *tmp1 = (int*) malloc(n*sizeof(int));
  int *tmp2 = (int*) malloc(n*sizeof(int));

  t1 = wall_timer();
  for (j=0; j<REPEAT; j++) {
    int bDim = BLOCKDIM;
    /*-------- init dependent counter of L and U */
    //int gDim0 = (n + BLOCKDIM - 1) / BLOCKDIM;
    //LU_SOL_DYNR_INIT<<<gDim0, bDim>>>(n, d_ia, d_da, d_dpl, d_dpu);
    cudaMemcpy(d_dp, d_dp_saved, 2*n*sizeof(int), cudaMemcpyDeviceToDevice);
    /*-------- num of warps per block */
    int nwb = BLOCKDIM / WARP;
    int gDim = (n + nwb-1) / nwb;
    // L-solve
    L_SOL_DYNR<<<gDim, bDim>>>(n, d_b, d_x, d_a, d_ja, d_ia, d_da, 
                              d_jb, d_ib, d_db, d_dpl, d_jlevL);
    // U-solve
    U_SOL_DYNR<<<gDim, bDim>>>(n, d_x, d_x, d_a, d_ja, d_ia, d_da, 
                              d_jb, d_ib, d_db, d_dpu, d_jlevU);
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;
 
  cudaMemcpy(tmp1, d_dpl, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp2, d_dpu, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    if (tmp1[i]) printf("i=%d: %d\n", i, tmp1[i]);
    if (tmp2[i]) printf("i=%d: %d\n", i, tmp2[i]);
  }

  if (print) {
    printf("[GPU] DYNR\n");
    printf("  time(s)=%f, Gflops=%5.3f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta/REPEAT, ta/t2);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  FreeLev(&lev);
  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_da);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ib);
  cudaFree(d_jb);
  cudaFree(d_db);
  cudaFree(d_x);
  cudaFree(d_dp);
  cudaFree(d_dp_saved);
  cudaFree(d_jlevL);
  cudaFree(d_jlevU);

  free(tmp1);
  free(tmp2);
}

