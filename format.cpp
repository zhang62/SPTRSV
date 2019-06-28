#include "lusol.h"

/*--------------------------------------------------*/
void COO2CSR(struct coo_t *coo, struct csr_t *csr) {
  //Allocate CSR
  csr->n = coo->n;
  csr->nnz = coo->nnz;
  csr->ia = (int *) malloc((csr->n+1)*sizeof(int));
  csr->ja = (int *) malloc(csr->nnz*sizeof(int));
  csr->a = (REAL *) malloc(csr->nnz*sizeof(REAL));
  // COO -> CSR
  FORT(coocsr)(&coo->n, &coo->nnz, coo->val, coo->ir, coo->jc, 
               csr->a, csr->ja, csr->ia);
}

