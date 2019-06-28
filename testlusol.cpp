#include "lusol.h"

using namespace std;

/* use a global variable to switch between CPU and GPU level construction */
int GPU_LEVEL = 0;

int main(int argc, char *argv[]) {
/*----------------------------------- *
 *  Driver program for GPU L/U solve  *
 *  x = U^{-1} * L^{-1} * b           *
 *  Kernels provided:                 *
 *  CPU L/U solve                     *
 *  GPU L/U solve w/ level-scheduling *
 *  GPU L/U solve w/ sync-free        *
 *----------------------------------- */
  int i,n,nnz,ret, nx=32, ny=32, nz=32, npts=7, flg=0, mm=1, dotest=0;
  REAL *h_b,*d_b,*d_x,*h_x0,*h_x1,*h_x2,*h_x3,*h_x4,*h_x5,*h_x6;
  struct coo_t h_coo;
  struct csr_t h_csr;
  double e1,e2,e3,e4,e5,e6;
  double t1;
  char fname[2048];
/*-----------------------------------------*/
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg) {
    printf("Usage: ./testL.ex -nx [int] -ny [int] -nz [int] -npts [int] -mat fname -mm [int] -dotest -gpulev [int]\n");
    return 0;
  }
  //srand (SEED);
  srand(time(NULL));
/*---------- Init GPU */
  cuda_init(argc, argv);
/*--------------------------------------*/
  if (DOUBLEPRECISION)
    printf("L/U Solv DOUBLE precision\n");
  else
    printf("L/U Solv SINGLE precision\n");
/*---------- cmd line arg */
  findarg("nx", INT, &nx, argc, argv);
  findarg("ny", INT, &ny, argc, argv);
  findarg("nz", INT, &nz, argc, argv);
  findarg("npts", INT, &npts, argc, argv);
  flg = findarg("mat", STR, fname, argc, argv);
  findarg("mm", INT, &mm, argc, argv);
  dotest = findarg("dotest", NA, &dotest, argc, argv);
  findarg("gpulev", INT, &GPU_LEVEL, argc, argv);
  //printf("gpulev %d\n", GPU_LEVEL);
/*---------- Read from Martrix Market file */
  if (flg == 1) {
    read_coo_MM(&h_coo, fname, mm);
  } else {
    lapgen(nx, ny, nz, &h_coo, npts);
  }
  n = h_coo.n;
  nnz = h_coo.nnz;
/*---------- COO -> CSR */
  COO2CSR(&h_coo, &h_csr);
/*---------- sort each row by increasing col idx */
  sortrow(n, h_csr.ia, h_csr.ja, h_csr.a);
/*---------- mark diag */
  finddiag(&h_csr);
/*--------------------- vector b */
  h_b = (REAL *) malloc(n*sizeof(REAL));
  for (i=0; i<n; i++)
    h_b[i] = rand() / (RAND_MAX + 1.0);

/*------------------------------------------------ */
/*------------- Start testing kernels ------------ */
/*------------------------------------------------ */
  int NTESTS = 100;
  bool PRINT = false;
  double err;
  
  if (!dotest) goto bench;

/*------------- CPU L/U Sol */
  h_x0 = (REAL *) malloc(n*sizeof(REAL));
  luSolCPU(n, nnz, h_b, h_x0, &h_csr, 1, PRINT);
/*------------ GPU L/U Solv w/ Lev-Sched R32 */
  err = 0.0;
  h_x1 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] LEVR32\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvLevR32(n, nnz, &h_csr, h_x1, h_b, 1, PRINT);
    e1=error_norm(h_x0, h_x1, n);
    err = max(e1, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x1);
/*------------ GPU L/U Solv w/ Lev-Sched R16 */
  err = 0.0;
  h_x1 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] LEVR16\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvLevR16(n, nnz, &h_csr, h_x1, h_b, 1, PRINT);
    e1=error_norm(h_x0, h_x1, n);
    err = max(e1, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x1);
/*------------ GPU L/U Solv w/ Lev-Sched C32 */
  err = 0.0;
  h_x6 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] LEVC32\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvLevC32(n, nnz, &h_csr, h_x6, h_b, 1, PRINT);
    e6=error_norm(h_x0, h_x6, n);
    err = max(e6, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x6);
/*------------ GPU L/U Solv w/ Lev-Sched C16 */
  err = 0.0;
  h_x6 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] LEVC16\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvLevC16(n, nnz, &h_csr, h_x6, h_b, 1, PRINT);
    e6=error_norm(h_x0, h_x6, n);
    err = max(e6, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x6);
/*----------- DYNR */
  err = 0.0;
  h_x4 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] DYNR\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvDYNR(n, nnz, &h_csr, h_x4, h_b, 1, PRINT);
    e4=error_norm(h_x0, h_x4, n);
    err = max(e4, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x4);
/*----------- DYNC */
  err = 0.0;
  h_x5 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] DYNC\n");
  for (int i=0; i<NTESTS; i++) {
    luSolvDYNC(n, nnz, &h_csr, h_x5, h_b, 1, PRINT);
    e5=error_norm(h_x0, h_x5, n);
    err = max(e5, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x5);
/*----------- CUSPARSE-1 */
  err = 0.0;
  h_x2 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] CUSPARSE csrsv1\n");
  for (int i=0; i<NTESTS; i++) {
    luSolv_cusparse1(&h_csr, h_b, h_x2, 1, PRINT);
    e2=error_norm(h_x0, h_x2, n);
    err = max(e2, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x2);
/*----------- CUSPARSE-2 */
  err = 0.0;
  h_x3 = (REAL *) malloc(n*sizeof(REAL));
  printf("[GPU] CUSPARSE csrsv2\n");
  for (int i=0; i<NTESTS; i++) {
    luSolv_cusparse2(&h_csr, h_b, h_x3, 1, PRINT);
    e3=error_norm(h_x0, h_x3, n);
    err = max(e3, err);
  }
  printf("err norm %.2e\n", err);
  free(h_x3);

  printf("\n\n\n");

bench:
/*------------------------------------------------ */
/*------------- Start benchmarking kernels ------- */
/*------------------------------------------------ */
  int REPEAT = 10;
/*------------- CPU L/U Sol */
  h_x0 = (REAL *) malloc(n*sizeof(REAL));
  luSolCPU(n, nnz, h_b, h_x0, &h_csr, REPEAT, true);
/*------------ GPU L/U Solv w/ Lev-Sched R32 */
  h_x1 = (REAL *) malloc(n*sizeof(REAL));
  luSolvLevR32(n, nnz, &h_csr, h_x1, h_b, REPEAT, true);
  e1=error_norm(h_x0, h_x1, n);
  printf("err norm %.2e\n", e1);
  free(h_x1);
/*------------ GPU L/U Solv w/ Lev-Sched R16 */
  h_x1 = (REAL *) malloc(n*sizeof(REAL));
  luSolvLevR16(n, nnz, &h_csr, h_x1, h_b, REPEAT, true);
  e1=error_norm(h_x0, h_x1, n);
  printf("err norm %.2e\n", e1);
  free(h_x1);
/*------------ GPU L/U Solv w/ Lev-Sched C32 */
  h_x6 = (REAL *) malloc(n*sizeof(REAL));
  luSolvLevC32(n, nnz, &h_csr, h_x6, h_b, REPEAT, true);
  e6=error_norm(h_x0, h_x6, n);
  printf("err norm %.2e\n", e6);
  free(h_x6);
/*------------ GPU L/U Solv w/ Lev-Sched C16 */
  h_x6 = (REAL *) malloc(n*sizeof(REAL));
  luSolvLevC16(n, nnz, &h_csr, h_x6, h_b, REPEAT, true);
  e6=error_norm(h_x0, h_x6, n);
  printf("err norm %.2e\n", e6);
  free(h_x6);
/*----------- DYNR */
  h_x4 = (REAL *) malloc(n*sizeof(REAL));
  luSolvDYNR(n, nnz, &h_csr, h_x4, h_b, REPEAT, true);
  e4=error_norm(h_x0, h_x4, n);
  printf("err norm %.2e\n", e4);
  free(h_x4);
/*----------- DYNC */
  h_x5 = (REAL *) malloc(n*sizeof(REAL));
  luSolvDYNC(n, nnz, &h_csr, h_x5, h_b, REPEAT, true);
  e5=error_norm(h_x0, h_x5, n);
  printf("err norm %.2e\n", e5);
  free(h_x5);
/*----------- CUSPARSE-1 */
  h_x2 = (REAL *) malloc(n*sizeof(REAL));
  luSolv_cusparse1(&h_csr, h_b, h_x2, REPEAT, true);
  e2=error_norm(h_x0, h_x2, n); e2 = max(e2, 0.0);
  printf("err norm %.2e\n", e2);
  free(h_x2);
/*----------- CUSPARSE-2 */
  h_x3 = (REAL *) malloc(n*sizeof(REAL));
  luSolv_cusparse2(&h_csr, h_b, h_x3, REPEAT, true);
  e3=error_norm(h_x0, h_x3, n);
  printf("err norm %.2e\n", e3);
  free(h_x3);
/*----------- Done free */
  free(h_b);  free(h_x0);    
  FreeCOO(&h_coo); FreeCSR(&h_csr);
/*---------- check error */
  cuda_check_err();
}

