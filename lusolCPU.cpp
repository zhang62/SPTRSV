#include "lusol.h"
#include "cusparse.h"

/*-------------------------------------------------------*/
void luSolCPU(int n, int nnz, REAL *b, REAL *x, csr_t *csr, 
              int REPEAT, bool print)
{
  int i,k,i1,i2,ii;
  
  int *ia = csr->ia;
  int *ja = csr->ja;
  REAL *a = csr->a;
  int *di = csr->di;

  double t1 = wall_timer();
  
#if 0
  for (ii=0; ii<REPEAT; ii++) {
    /* Forward solve. Solve L*x = b */
    for (i=0; i<n; i++) {
      x[i] = b[i];
      i1 = ia[i];
      i2 = di[i];
      double rr = 0.0;
//#pragma omp parallel for reduction(+:rr)
      for (k=i1; k<i2; k++)
        rr += a[k-1]*x[ja[k-1]-1];
      REAL t = 1.0 / a[i2-1];
      x[i] = t * (x[i] - rr);
    }
    /* Backward slove. Solve x = U^{-1}*x */
    for (i=n-1; i>=0; i--) {
      i1 = di[i];
      i2 = ia[i+1];
      double rr = 0.0;
//#pragma omp parallel for reduction(+:rr)
      for (k=i1+1; k<i2; k++)
        rr += a[k-1]*x[ja[k-1]-1];
      REAL t = 1.0 / a[i1-1];
      x[i] = t * (x[i] - rr);
    }
  }
#else
  for (ii=0; ii<REPEAT; ii++) {
    /* Forward solve. Solve L*x = b */
    for (i=0; i<n; i++) {
      x[i] = b[i];
      i1 = ia[i];
      i2 = di[i];
      for (k=i1; k<i2; k++)
        x[i] -= a[k-1]*x[ja[k-1]-1];
      REAL t = 1.0 / a[i2-1];
      x[i] *= t;
    }
    /* Backward slove. Solve x = U^{-1}*x */
    for (i=n-1; i>=0; i--) {
      i1 = di[i];
      i2 = ia[i+1];
      for (k=i1+1; k<i2; k++)
        x[i] -= a[k-1]*x[ja[k-1]-1];
      REAL t = 1.0 / a[i1-1];
      x[i] *= t;
    }
  }
#endif

  double t2 = wall_timer() - t1;
  if (print) {
    printf("[CPU]\n");
    printf("  time(s)=%f, Gflops=%5.2f\n\n", t2/REPEAT, 
           REPEAT*2*((nnz)/1e9)/t2);
  }
}

