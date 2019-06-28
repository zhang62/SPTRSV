#include "lusol.h"



/*----------------------------------------------------------*/
void cudaMallocCSR(int n, int nnz, struct csr_t *d_csr)
{
  cudaMalloc((void **)&d_csr->ia, (size_t)(n+1)*sizeof(int));
  cudaMalloc((void **)&d_csr->ja, (size_t)(nnz)*sizeof(int));
  cudaMalloc((void **)&d_csr->a,  (size_t)(nnz)*sizeof(REAL));
}

/*----------------------------------------------------------*/
void CSRHost2Device(struct csr_t *h_csr, struct csr_t *d_csr)
{
  int n = h_csr->n;
  int nnz = h_csr->nnz;

  cudaMemcpy(d_csr->ia, h_csr->ia, (size_t)(n+1)*sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr->ja, h_csr->ja, (size_t)(nnz)*sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr->a,  h_csr->a,  (size_t)(nnz)*sizeof(REAL),
             cudaMemcpyHostToDevice);
}

/*-------------------------------------------------------------*/
void cudaMallocLU(int n, int nnzl, int nnzu, struct lu_t *d_lu)
{
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->l.ia, (size_t)(n+1)*sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->l.ja, (size_t)(nnzl)*sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->l.a,  (size_t)(nnzl)*sizeof(REAL)));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->u.ia, (size_t)(n+1)*sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->u.ja, (size_t)(nnzu)*sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_lu->u.a,  (size_t)(nnzu)*sizeof(REAL)));
}

/*------------------------------------------------------------*/
void LUHost2Device(struct lu_t *h_lu, struct lu_t *d_lu)
{
  // L
  int n   = h_lu->l.n;
  int nnz = h_lu->l.nnz;
  CUDA_SAFE_CALL(cudaMemcpy(d_lu->l.ia, h_lu->l.ia, (size_t)(n+1)*sizeof(int),
             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_lu->l.ja, h_lu->l.ja, (size_t)(nnz)*sizeof(int),
             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_lu->l.a,  h_lu->l.a,  (size_t)(nnz)*sizeof(REAL),
             cudaMemcpyHostToDevice));

  // U
  n   = h_lu->u.n;
  nnz = h_lu->u.nnz;

  CUDA_SAFE_CALL(cudaMemcpy(d_lu->u.ia, h_lu->u.ia, (size_t)(n+1)*sizeof(int),
             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_lu->u.ja, h_lu->u.ja, (size_t)(nnz)*sizeof(int),
             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_lu->u.a,  h_lu->u.a,  (size_t)(nnz)*sizeof(REAL),
             cudaMemcpyHostToDevice));
}

/*---------------------------------*/
void CudaFreeCSR(struct csr_t *d_csr)
{
  cudaFree(d_csr->ia);
  cudaFree(d_csr->ja);
  cudaFree(d_csr->a);
}

/*------------------------------*/
void CudaFreeLU(struct lu_t *d_lu)
{
  cudaFree(d_lu->l.ia);
  cudaFree(d_lu->l.ja);
  cudaFree(d_lu->l.a);
  cudaFree(d_lu->u.ia);
  cudaFree(d_lu->u.ja);
  cudaFree(d_lu->u.a);
}

/*-----------------------------------*/
void CudaFreeLev(struct level_t *d_lev)
{
  cudaFree(d_lev->jlevL);
  cudaFree(d_lev->ilevL);
  cudaFree(d_lev->jlevU);
  cudaFree(d_lev->ilevU);
}

