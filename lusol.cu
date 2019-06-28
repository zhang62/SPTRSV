#include "lusol.h"
#include "cusparse.h"

/*-----------------------------------------------*/
void cuda_init(int argc, char **argv) {
  int deviceCount, dev;
  cudaGetDeviceCount(&deviceCount);
  printf("=========================================\n");
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        printf("There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
        printf("There is 1 device supporting CUDA\n");
      else
        printf("There are %d devices supporting CUDA\n", deviceCount);
    }
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("  Major revision number:          %d\n",
           deviceProp.major);
    printf("  Minor revision number:          %d\n",
           deviceProp.minor);
    printf("  Total amount of global memory:  %.2f GB\n",
           deviceProp.totalGlobalMem/1e9);
  }
  dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\nRunning on Device %d: \"%s\"\n", dev, deviceProp.name);
  printf("=========================================\n");
}

/*---------------------------------------------------*/
void cuda_check_err() {
  cudaError_t cudaerr = cudaGetLastError() ;
  if (cudaerr != cudaSuccess) 
    printf("error: %s\n",cudaGetErrorString(cudaerr));
}

void luSolv_cusparse1(struct csr_t *csr, REAL *b, REAL *x, 
                      int REPEAT, bool print) {
  int n = csr->n;
  int nnz = csr->nnz; 
  int *d_ia, *d_ja;
  REAL *d_a, *d_b, *d_x, *d_y;
  double t1, t2, ta;
  REAL done = 1.0;
/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_y, n*sizeof(REAL));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL),
  cudaMemcpyHostToDevice);
    
  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr_L=0, descr_U=0;

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library initialization failed\n");
     exit(1);
  }

  /* create and setup matrix descriptor for L */ 
  status= cusparseCreateMatDescr(&descr_L); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization L failed\n");
     exit(1);
  }

  t1 = wall_timer();

  cusparseSolveAnalysisInfo_t info_L = 0;
  cusparseCreateSolveAnalysisInfo(&info_L);
  cusparseSetMatType(descr_L,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

#if DOUBLEPRECISION
  status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                   descr_L, d_a, d_ia, d_ja, info_L);
#else
  status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                   descr_L, d_a, d_ia, d_ja, info_L);
#endif

  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse?csrsv_analysis L failed\n");
    exit(1);
  }

  /* create and setup matrix descriptor for U */ 
  status= cusparseCreateMatDescr(&descr_U); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization U failed\n");
     exit(1);
  }

  cusparseSolveAnalysisInfo_t info_U = 0;
  cusparseCreateSolveAnalysisInfo(&info_U);
  cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);


#if DOUBLEPRECISION
  status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                   descr_U, d_a, d_ia, d_ja, info_U);
#else
  status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                   descr_U, d_a, d_ia, d_ja, info_U);
#endif

  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse?csrsv_analysis L failed\n");
    exit(1);
  }
 
  //Barrier for GPU calls
  cudaThreadSynchronize();
  ta = wall_timer() - t1;

  t1 = wall_timer();
  for (int j=0; j<REPEAT; j++) {
#if DOUBLEPRECISION
    // L-solve
    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, 
                                  descr_L, d_a, d_ia, d_ja, info_L, d_b, d_y);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
    // U-solve
    status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, 
                                  descr_U, d_a, d_ia, d_ja, info_U, d_y, d_x);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
#else
    // L-solve
    status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, 
                                  descr_L, d_a, d_ia, d_ja, info_L, d_b, d_y);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
    // U-solve
    status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &done, 
                                  descr_U, d_a, d_ia, d_ja, info_U, d_y, d_x);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
#endif
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] CUSPARSE csrsv\n");
    printf("  time(s)=%f, Gflops=%5.2f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta, ta/t2*REPEAT);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_y);

  /* destroy matrix descriptor */ 
  status = cusparseDestroyMatDescr(descr_L); 
  descr_L = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor destruction failed\n");
    exit(1);
  }
  
  status = cusparseDestroyMatDescr(descr_U); 
  descr_U = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor destruction failed\n");
    exit(1);
  }

  status = cusparseDestroySolveAnalysisInfo(info_L);
  info_L = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("analysis info destruction failed\n");
    exit(1);
  }

  status = cusparseDestroySolveAnalysisInfo(info_U);
  info_U = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("analysis info destruction failed\n");
    exit(1);
  }

  /* destroy handle */
  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library release of resources failed\n");
     exit(1);
  }
}

void luSolv_cusparse2(struct csr_t *csr, REAL *b, REAL *x, 
                      int REPEAT, bool print) {
  int n = csr->n;
  int nnz = csr->nnz; 
  int *d_ia, *d_ja;
  REAL *d_a, *d_b, *d_x, *d_y;
  double t1, t2, ta;
  REAL done = 1.0;
/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_y, n*sizeof(REAL));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(REAL),
  cudaMemcpyHostToDevice);
    
  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr_L=0, descr_U=0;

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library initialization failed\n");
     exit(1);
  }

  /* create and setup matrix descriptor for L */ 
  status= cusparseCreateMatDescr(&descr_L); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization L failed\n");
     exit(1);
  }

  t1 = wall_timer();

  csrsv2Info_t info_L = 0;
  cusparseCreateCsrsv2Info(&info_L);
  cusparseSetMatType(descr_L,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

  int pBufferSize_L;
  void *pBuffer_L = 0;
#if DOUBLEPRECISION
  cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                             descr_L, d_a, d_ia, d_ja, info_L, &pBufferSize_L);
#else
  cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                             descr_L, d_a, d_ia, d_ja, info_L, &pBufferSize_L);
#endif
  
  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void**)&pBuffer_L, pBufferSize_L);

#if DOUBLEPRECISION
  status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                    descr_L, d_a, d_ia, d_ja, info_L, 
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);
#else
  status = cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                    descr_L, d_a, d_ia, d_ja, info_L,
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);

#endif

  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse?csrsv_analysis L failed\n");
    exit(1);
  }

  /* create and setup matrix descriptor for U */ 
  status= cusparseCreateMatDescr(&descr_U); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization U failed\n");
     exit(1);
  }

  csrsv2Info_t info_U = 0;
  cusparseCreateCsrsv2Info(&info_U);
  cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

  int pBufferSize_U;
  void *pBuffer_U = 0;
#if DOUBLEPRECISION
  cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                             descr_U, d_a, d_ia, d_ja, info_U, &pBufferSize_U);
#else
  cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                             descr_U, d_a, d_ia, d_ja, info_U, &pBufferSize_U);
#endif

  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void**)&pBuffer_U, pBufferSize_U);

#if DOUBLEPRECISION
  status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                    descr_U, d_a, d_ia, d_ja, info_U,
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
#else
  status = cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, 
                                    descr_U, d_a, d_ia, d_ja, info_U,
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
#endif

  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse?csrsv_analysis U failed\n");
    exit(1);
  }
 
  //Barrier for GPU calls
  cudaThreadSynchronize();
  ta = wall_timer() - t1;

  t1 = wall_timer();
  for (int j=0; j<REPEAT; j++) {
#if DOUBLEPRECISION
    // L-solve
    status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &done, 
                                   descr_L, d_a, d_ia, d_ja, info_L, d_b, d_y,
                                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
    // U-solve
    status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &done, 
                                   descr_U, d_a, d_ia, d_ja, info_U, d_y, d_x,
                                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
#else
    // L-solve
    status = cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &done, 
                                   descr_L, d_a, d_ia, d_ja, info_L, d_b, d_y,
                                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
    // U-solve
    status = cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &done, 
                                   descr_U, d_a, d_ia, d_ja, info_U, d_y, d_x,
                                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("cusparse?csrsv_solve L failed\n");
      exit(1);
    }
#endif
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;

  if (print) {
    printf("[GPU] CUSPARSE csrsv2\n");
    printf("  time(s)=%f, Gflops=%5.2f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);
    printf("  analysis time %f  (%f) ", ta, ta/t2*REPEAT);
  }

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(pBuffer_L);
  cudaFree(pBuffer_U);

  /* destroy matrix descriptor */ 
  status = cusparseDestroyMatDescr(descr_L); 
  descr_L = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor destruction failed\n");
    exit(1);
  }
  
  status = cusparseDestroyMatDescr(descr_U); 
  descr_U = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix descriptor destruction failed\n");
    exit(1);
  }

  status = cusparseDestroyCsrsv2Info(info_L);
  info_L = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("analysis info destruction failed\n");
    exit(1);
  }

  status = cusparseDestroyCsrsv2Info(info_U);
  info_U = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("analysis info destruction failed\n");
    exit(1);
  }

  /* destroy handle */
  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library release of resources failed\n");
     exit(1);
  }
}



#if 0
__global__
void LU_SOL_SF2_INIT(int n, int *ia, int *da, int *dpl, int *dpu,
                     int *hl, int *hu, int *d_tl, int tl, 
                     int *d_tu, int tu, int *lockL, int *lockU) {
  int gid = blockIdx.x * BLOCKDIM + threadIdx.x;
  if (gid < n) {
    int t = da[gid];
    dpl[gid] = t - ia[gid];
    dpu[gid] = ia[gid+1] - t - 1;
  }
  if (gid == 0) {
    *hl = 0;
    *hu = 0;
    *d_tl = tl;
    *d_tu = tu;
    *lockL = 0;
    *lockU = 0;
  }
}

/* lock: 0 is unlocked and 1 is locked */
__global__
void L_SOL_SF2(int n, REAL *b, REAL *x, REAL *a, int *ja, int *ia, int *da,
               int *jb, int *ib, int *db, int *dp, int *jlev, 
               volatile int *head, volatile int *tail, volatile int *lock) {
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
  int wlane = threadIdx.x / WARP;
  volatile __shared__ int buffer[BLOCKDIM/WARP];
  // make dp volatile to tell compiler do not use cached value
  //volatile int *vdp = dp;
  
  if (wid >= n) {
    return;
  }

  int i, p1, q1, p2, q2, hd, tl;
  REAL dinv, bi, sum;
  
  if (lane == 0) {
    while (1) {
      // try to lock
      while (atomicCAS((int*)lock, 0, 1) != 0);

      // locked, read the head and tail
      hd = *head;  tl = *tail;
      if (hd <= tl) {
        /* there is a row for me to work on, increase the head */
        (*head) ++;
        __threadfence();
      }

      // realease the lock
      atomicExch((int*)lock, 0);

      if (hd >= n) {
        buffer[wlane] = -1;
        break;
      } else if (hd <= tl) {
        int tt = jlev[hd] - 1;
        buffer[wlane] = tt;
        break;
      }
    }
  }

  // this warp get a row to work on
  i = buffer[wlane];

  //if (lane == 0) {
  //  printf("hd %d tl %d row i = %d\n", hd, tl, i);
  //}

  if (i < 0) {
    return;
  }

  p1 = ia[i];
  q1 = da[i];
  p2 = db[i] + 1;
  q2 = ib[i+1];
  sum = 0.0;
  if (lane == 0) {
    dinv = 1.0 / a[q1-1];
    bi = b[i];
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

    /* remove i from other's dependents */
    for (int k=p2; k<q2; k++) {
      int s1 = jb[k-1]-1;
      int *p = dp + s1;
      int old = atomicSub(p, 1);
      if (old == 1) {
        while (atomicCAS((int*)lock, 0, 1) != 0);
        (*tail)++;
        jlev[*tail] = s1 + 1;
        __threadfence();
        atomicExch((int*)lock, 0);
      }
    }
  }
}


//--------------------------------------------------------
void luSolvSF2(int n, int nnz, struct csr_t *csr,
               struct syncfree_t *syncf, REAL *x, REAL *b)
{
  int j, *d_ia, *d_ja, *d_da, *d_ib, *d_jb, *d_db, *d_dpl,
      *d_dpu, *d_jlevL, *d_jlevU, *lockL, *lockU, *headL, *headU, 
      *tailL, *tailU;
  REAL *d_a, *d_b, *d_x;
  double t1, t2;

/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_da, n*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_b, n*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
  cudaMalloc((void **)&d_jb, nnz*sizeof(int));
  cudaMalloc((void **)&d_db, n*sizeof(int));
  cudaMalloc((void **)&d_dpl, n*sizeof(int));
  cudaMalloc((void **)&d_dpu, n*sizeof(int));
  cudaMalloc((void **)&d_jlevL, n*sizeof(int));
  cudaMalloc((void **)&d_jlevU, n*sizeof(int));
  cudaMalloc((void **)&lockL, sizeof(int));
  cudaMalloc((void **)&lockU, sizeof(int));
  cudaMalloc((void **)&headL, sizeof(int));
  cudaMalloc((void **)&headU, sizeof(int));
  cudaMalloc((void **)&tailL, sizeof(int));
  cudaMalloc((void **)&tailU, sizeof(int));
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
  cudaMemcpy(d_ib, syncf->AT.ia, (n+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_jb, syncf->AT.ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_db, syncf->AT.di, n*sizeof(int),
  cudaMemcpyHostToDevice);
  //cudaMemcpy(d_dpl, syncf->dpL, n*sizeof(int),
  //cudaMemcpyHostToDevice);
  //cudaMemcpy(d_dpu, syncf->dpU, n*sizeof(int),
  //cudaMemcpyHostToDevice);
  cudaMemcpy(d_jlevL, syncf->lev.jlevL, n*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_jlevU, syncf->lev.jlevU, n*sizeof(int),
  cudaMemcpyHostToDevice);

  int *tmp1 = (int*) malloc(n*sizeof(int));
  int *tmp2 = (int*) malloc(n*sizeof(int));

  // number of free rows (level == 0)
  int tl = syncf->lev.ilevL[1] - syncf->lev.ilevL[0];
  int tu = syncf->lev.ilevU[1] - syncf->lev.ilevU[0];

  t1 = wall_timer();
  for (j=0; j<REPEAT; j++) {
    int bDim = BLOCKDIM;
    /*-------- init dependent counter of L and U */
    int gDim0 = (n + BLOCKDIM - 1) / BLOCKDIM;
    LU_SOL_SF2_INIT<<<gDim0, bDim>>>(n, d_ia, d_da, d_dpl, d_dpu, headL, headU, 
                                     tailL, tl-1, tailU, tu-1, lockL, lockU);
    /*-------- num of warps per block */
    int nwb = BLOCKDIM / WARP;
    int gDim = (n + nwb-1) / nwb;
    // L-solve
    L_SOL_SF2<<<gDim, bDim>>>(n, d_b, d_x, d_a, d_ja, d_ia, d_da, 
                              d_jb, d_ib, d_db, d_dpl, d_jlevL, headL, tailL, lockL);
    // U-solve
    //U_SOL_SF1<<<gDim, bDim>>>(n, d_x, d_x, d_a, d_ja, d_ia, d_da, 
    //                          d_jb, d_ib, d_db, d_dpu, d_jlevU);
    break;
  }

  //Barrier for GPU calls
  cudaThreadSynchronize();
  t2 = wall_timer() - t1;
 
  cudaMemcpy(tmp1, d_dpl, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp2, d_dpu, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    //if (tmp1[i] != syncf->dpL[i]) printf("i=%d: %d %d\n", i, tmp1[i], syncf->dpL[i]);
    //if (tmp2[i] != syncf->dpU[i]) printf("i=%d: %d %d\n", i, tmp2[i], syncf->dpU[i]);
    if (tmp1[i]) printf("i=%d: %d\n", i, tmp1[i]);
    //if (tmp2[i]) printf("i=%d: %d\n", i, tmp2[i]);
  }

  printf("[GPU] SyncFree v-1\n");
  printf("  time(s)=%f, Gflops=%5.2f", t2/REPEAT, REPEAT*2*((nnz)/1e9)/t2);

  /*-------- copy x to host mem */
  cudaMemcpy(x, d_x, n*sizeof(REAL),
  cudaMemcpyDeviceToHost);

  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_da);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ib);
  cudaFree(d_jb);
  cudaFree(d_db);
  cudaFree(d_x);
  cudaFree(d_dpl);
  cudaFree(d_dpu);
  free(tmp1);
  free(tmp2);
}
#endif


