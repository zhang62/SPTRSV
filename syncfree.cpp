#if 0
#include "lusol.h"

void CreateSyncfree(struct csr_t *csr, struct syncfree_t *syncf) {
  int n = csr->n;
  int nnz = csr->nnz;

  syncf->AT.n = n;
  syncf->AT.nnz = nnz;
  syncf->AT.ia = (int *) malloc((n+1)*sizeof(int));
  syncf->AT.ja = (int *) malloc(nnz*sizeof(int));
  syncf->AT.a = (REAL *) malloc(nnz*sizeof(REAL));
  csrcsc(n, n, 1, 1, csr->a, csr->ja, csr->ia, syncf->AT.a, syncf->AT.ja, syncf->AT.ia);
  finddiag(&syncf->AT);

  syncf->dpL = (int *) malloc(n*sizeof(int));
  syncf->dpU = (int *) malloc(n*sizeof(int));

  for (int i = 0; i < n; i++) {
    syncf->dpL[i] = csr->di[i] - csr->ia[i];
    syncf->dpU[i] = csr->ia[i+1] - csr->di[i] - 1;
  }

  makeLevel(csr, &syncf->lev);
}

void FreeSyncfree(struct syncfree_t *syncf) {
  free(syncf->AT.a);
  free(syncf->AT.ia);
  free(syncf->AT.ja);
  free(syncf->dpL);
  free(syncf->dpU);
}
#endif
