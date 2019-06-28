#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>

#if DOUBLEPRECISION
#define REAL double
#else
#define REAL float
#endif

#define FORT(name) name ## _
//#define FORT(name) name

/*
#define CUDA_SAFE_CALL_NO_SYNC( call) {                               \
    cudaError err = call;                                             \
    if( cudaSuccess != err) {                                         \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString( err) );       \
        exit(EXIT_FAILURE);                                           \
    } }
#define CUDA_SAFE_CALL( call)    CUDA_SAFE_CALL_NO_SYNC(call);
*/

#define WARP 32
#define HALFWARP 16
#define BLOCKDIM 512
// number of half warp per block
#define NHW (BLOCKDIM/HALFWARP)
// number of warp per block
#define NW (BLOCKDIM/WARP)
#define MAXTHREADS (30 * 1024 * 60)
#define SEED 200

// COO format type
struct coo_t {
  int n;
  int nnz;
  int *ir;
  int *jc;
  REAL *val;
};

// CSR format type
struct csr_t {
  int n;
  int nnz;
  int *ia;
  int *ja;
  int *di;
  REAL *a;
};

struct level_t {
  // L
  int nlevL;
  int *jlevL;
  int *ilevL;
  // U
  int nlevU;
  int *jlevU;
  int *ilevU;
  // level
  int *levL;
  int *levU;
};

/*
struct syncfree_t {
  struct csr_t AT;
  struct level_t lev;
  int *dpL, *dpU;
};
*/

/* types of user command-line input */
typedef enum {
  INT,
  DOUBLE,
  STR,
  NA
} ARG_TYPE;

#include <protos.h>

extern int GPU_LEVEL;
