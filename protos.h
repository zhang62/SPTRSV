void cuda_init(int, char **);
void cuda_check_err();
int get_arg(int, char**, int*, char**);
int read_coo_MM(struct coo_t *coo, char *matfile, int);
void COO2CSR(struct coo_t *, struct csr_t *);
void cudaMallocCSR(int n, int nnz, struct csr_t *d_csr);
void CSRHost2Device(struct csr_t *, struct csr_t *);
void cudaMallocLU(int n, int nnzl, int nnzu, struct lu_t *d_lu);
void LUHost2Device(struct lu_t *h_lu, struct lu_t *d_lu);
void FreeCOO(struct coo_t *coo);
void FreeCSR(struct csr_t *csr);
void FreeLU(struct lu_t *h_lu);
void FreeLev(struct level_t *h_lev);
void CudaFreeCSR(struct csr_t *d_csr);
void CudaFreeLU(struct lu_t *d_lu);
void CudaFreeLev(struct level_t *h_lev);
extern "C" {
void FORT(coocsr)(int *, int *, REAL *,
                  int *, int *, REAL *, int *, int *);
}
int ilu0(struct csr_t *csr, struct lu_t *lu);
void sortrow(int n, int *ia, int *ja, REAL *a);
void luSolv(int n, int nnz, struct lu_t *d_lu, REAL *d_x, REAL *d_b);
void makeLevel(struct csr_t *, struct level_t *);
void makeLevelCSR(int n, int *ia, int *ja, int *, struct level_t *h_lev);
void makeLevelCSC(int n, int *ia, int *ja, int *, struct level_t *h_lev);

void luSolCPU(int n, int nnz, REAL *b, REAL *x, csr_t *csr, int, bool);

void luSolvLevR16(int n, int nnz, struct csr_t *, REAL *d_x, REAL *d_b, int, bool);
void luSolvLevR32(int n, int nnz, struct csr_t *, REAL *d_x, REAL *d_b, int, bool);
void luSolvLevC16(int n, int nnz, struct csr_t *, REAL *d_x, REAL *d_b, int, bool);
void luSolvLevC32(int n, int nnz, struct csr_t *, REAL *d_x, REAL *d_b, int, bool);

void csrcsc(int n, int n2, int job, int ipos, 
            REAL *a, int *ja, int *ia, 
	    REAL *ao, int *jao, int *iao);
void Reorder_MMD(struct csr_t *A, int **perm, int **iperm);
void RowColPermMat(struct csr_t *A, int *perm);
double wall_timer();
double error_norm(REAL *x, REAL *y, int n);

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);
int lapgen(int nx, int ny, int nz, struct coo_t *Acoo, int npts);

void diagpos(int n, int *ia, int *ja, REAL *a, int *di);
void finddiag(struct csr_t *csr);

void luSolv_cusparse1(struct csr_t *csr, REAL *b, REAL *x, int, bool);
void luSolv_cusparse2(struct csr_t *csr, REAL *b, REAL *x, int, bool);

void CreateSyncfree(struct csr_t *csr, struct syncfree_t *syncf);
void FreeSyncfree(struct syncfree_t *syncf);
void luSolvDYNR(int n, int nnz, struct csr_t *csr,
               REAL *x, REAL *b, int, bool);

void luSolvDYNC(int n, int nnz, struct csr_t *csr,
               REAL *x, REAL *b, int, bool);

void allocLevel(int n, struct level_t *lev);

void makeLevelCSC_SYNC(int n, int *d_ib, int *d_jb, int *d_db, 
                       int *d_dp, int *d_jlevL, int *ilevL, int *,
                       int *d_jlevU, int *ilevU, int*);

void makeTopoCSC(int n, int *d_ib, int *d_jb, int *d_db, 
                 int *d_dp, int *d_jlevL, int *d_jlevU);

void checktopo(int n, int *ib, int *jb, int *db, int *d_jlevL, 
               int *d_jlevU, int *d_dp);
