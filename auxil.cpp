#include "lusol.h"
#include <getopt.h>
/*------------------------------------------*/
double wall_timer() {
  struct timeval tim;
  gettimeofday(&tim, NULL);
  double t = tim.tv_sec + tim.tv_usec/1e6;
  return(t);
}
/*------------------------------*/
void FreeCOO(struct coo_t *coo) {
  free(coo->ir);
  free(coo->jc);
  free(coo->val);
}

/*---------------------------*/
void FreeCSR(struct csr_t *csr)
{
  free(csr->a);
  free(csr->ia);
  free(csr->ja);
  free(csr->di);
}

/*-------------------------------------------------*/
void csrcsc(int n, int n2, int job, int ipos, 
            REAL *a, int *ja, int *ia, 
	    REAL *ao, int *jao, int *iao)
{
  int i,j,k,next;
  
  /*compute lengths of rows of A' */
  for (i=1; i<=n2+1; i++)
    iao[i-1] = 0;
    
  for (i=1; i<=n; i++)
    for (k=ia[i-1]; k<=ia[i]-1; k++)
    {
      j = ja[k-1]+1;
      iao[j-1] ++;
    }
  
  /* compute pointers from lengths */  
  iao[0] = ipos;
  for (i=1; i<=n2; i++)
    iao[i] += iao[i-1];
    
  /* now do the actual copying */
  for (i=1; i<=n; i++)
    for (k=ia[i-1]; k<=ia[i]-1; k++)
    {
      j = ja[k-1];
      next = iao[j-1];
      if (job == 1)
        ao[next-1] = a[k-1];
      jao[next-1] = i;
      iao[j-1] = next + 1;
    }
    
  /* reshift iao and leave */
  for (i=n2; i>=1; i--)
    iao[i] = iao[i-1];
  iao[0] = ipos;
}

/*-------------------------------------------*
 * Sort each row by increasing column order
 * By double transposition
 *-------------------------------------------*/
void sortrow(int n, int *ia, int *ja, REAL *a)
{
  int nnz = ia[n] - 1;
  // work array
  REAL *b = (REAL *) malloc(nnz*sizeof(REAL));
  int *jb = (int *) malloc(nnz*sizeof(int));
  int *ib = (int *) malloc((n+1)*sizeof(int));

  // double transposition
  csrcsc(n, n, 1, 1, a, ja, ia, b, jb, ib);
  csrcsc(n, n, 1, 1, b, jb, ib, a, ja, ia);

  free(b);
  free(jb);
  free(ib);
}

void diagpos(int n, int *ia, int *ja, REAL *a, int *di) {
  for (int i = 0; i < n; i++) {
    di[i] = -1;
    double vi = 0.0;
    for (int j = ia[i]; j < ia[i+1]; j++) {
      int k = ja[j-1]-1;
      if (i == k) {
        di[i] = j;
        vi = a[j-1];
      }
    }
    if (di[i] == -1 || fabs(vi) < 1e-15) {
      printf("error: zero diag %d, %e\n", i, vi);
      exit(-1);
    }
  }
}

void finddiag(struct csr_t *csr) {
  csr->di = (int *) malloc(csr->n*sizeof(int));
  diagpos(csr->n, csr->ia, csr->ja, csr->a, csr->di);
}

/*-----------------------------------------*/
double error_norm(REAL *x, REAL *y, int n) {
  int i;
  double t, normz, normx;
  normx = normz = 0.0;
  for (i=0; i<n; i++) {
    t = x[i]-y[i];
    normz += t*t;
    normx += x[i]*x[i];
  }
  return (sqrt(normz/normx));
}

