#include "lusol.h"

using namespace std;

void allocLevel(int n, struct level_t *lev) {
  lev->nlevL = 0;
  lev->jlevL = (int *) malloc(n*sizeof(int));
  lev->ilevL = (int *) malloc(n*sizeof(int));
  lev->nlevU = 0;
  lev->jlevU = (int *) malloc(n*sizeof(int));
  lev->ilevU = (int *) malloc(n*sizeof(int));
  lev->levL = (int *) malloc(n*sizeof(int));
  lev->levU = (int *) malloc(n*sizeof(int));
}

/*-------------------------------------------------------------*/
/* arrays in h_lev should be allocated before calling */
void makeLevelCSR(int n, int *ia, int *ja, int *da, struct level_t *h_lev) 
{
  int *level;

  //double tt, ta2, t3;
  
  memset(h_lev->ilevL, 0, n*sizeof(int));
  memset(h_lev->ilevU, 0, n*sizeof(int));

  // L
  level = h_lev->levL;

  h_lev->ilevL[0] = 1;
  

  //tt = wall_timer();
  //for (int i = 0; i < n; i++) {
    //for (int j = ia[i]; j < da[i]; j++) {
      //int k = ja[j-1]-1;
      //level[k] = i;
    //}
    //level[i] = 0;
  //}
  //t3 = wall_timer() - tt;
  //printf("t3 %f\n", t3);

  //tt = wall_timer();
  
  for (int i = 0; i < n; i++)
  {
    int l = 0;
    for (int j = ia[i]; j < da[i]; j++) {
      //int k = ja[j-1]-1;
      //l = max(l, level[k]);
      l = max(l, level[ja[j-1]-1]);
    }
    level[i] = l+1;
    // used as counters, (note the shift by 1)
    h_lev->ilevL[l+1] ++;
    h_lev->nlevL = max(h_lev->nlevL, l+1);
  }

  //ta2 = wall_timer() - tt;
  //printf("ta2 %f\n", ta2);



  for (int i=1; i<=h_lev->nlevL; i++)
    h_lev->ilevL[i] += h_lev->ilevL[i-1];

  for (int i=0; i<n; i++)
  {
    int *k = &h_lev->ilevL[level[i]-1];
    h_lev->jlevL[(*k)-1] = i+1;
    (*k)++;
  }

  for (int i=h_lev->nlevL-1; i>0; i--)
    h_lev->ilevL[i] = h_lev->ilevL[i-1];

  h_lev->ilevL[0] = 1;

  // U
  level = h_lev->levU;

  h_lev->ilevU[0] = 1;
  
  //tt = wall_timer();
  
  for (int i=n-1; i>=0; i--)
  {
    int l = 0;
    for (int j = da[i]+1; j < ia[i+1]; j++) {
      //int k = ja[j-1]-1;
      //l = max(l, level[k]);
      l = max(l, level[ja[j-1]-1]);
    }

    level[i] = l+1;
    h_lev->ilevU[l+1] ++;
    h_lev->nlevU = max(h_lev->nlevU, l+1);
  }

  //ta2 += wall_timer() - tt;
  
  //printf("ta2 %f\n", ta2);
  
  for (int i=1; i<=h_lev->nlevU; i++)
    h_lev->ilevU[i] += h_lev->ilevU[i-1];

  for (int i=0; i<n; i++)
  {
    int *k = &h_lev->ilevU[level[i]-1];
    h_lev->jlevU[(*k)-1] = i+1;
    (*k)++;
  }

  for (int i=h_lev->nlevU-1; i>0; i--)
    h_lev->ilevU[i] = h_lev->ilevU[i-1];

  h_lev->ilevU[0] = 1;
}

/*-------------------------------------------------------------*/
/* ia, ja : CSC
 * arrays in h_lev should be allocated before calling */
void makeLevelCSC(int n, int *ia, int *ja, int *da, struct level_t *h_lev) 
{
  int *level;

  memset(h_lev->ilevL, 0, n*sizeof(int));
  memset(h_lev->ilevU, 0, n*sizeof(int));

  // L
  level = h_lev->levL;
  memset(level, 0, n*sizeof(int));

  h_lev->ilevL[0] = 1;
  for (int i = 0; i < n; i++)
  {
    int l = level[i];
//#pragma omp parallel for schedule(static)
    for (int j = da[i]+1; j < ia[i+1]; j++) {
      int k = ja[j-1]-1;
      level[k] = max(level[k], l+1);
    }
    // used as counters, (note the shift by 1)
    h_lev->ilevL[l+1] ++;
    h_lev->nlevL = max(h_lev->nlevL, l+1);
  }

  for (int i=1; i<=h_lev->nlevL; i++)
    h_lev->ilevL[i] += h_lev->ilevL[i-1];

  for (int i=0; i<n; i++)
  {
    int *k = &h_lev->ilevL[level[i]];
    h_lev->jlevL[(*k)-1] = i+1;
    (*k)++;
  }

  for (int i=h_lev->nlevL-1; i>0; i--)
    h_lev->ilevL[i] = h_lev->ilevL[i-1];

  h_lev->ilevL[0] = 1;

  // U
  level = h_lev->levU;
  memset(level, 0, n*sizeof(int));

  h_lev->ilevU[0] = 1;
  for (int i=n-1; i>=0; i--)
  {
    int l = level[i];
    for (int j = ia[i]; j < da[i]; j++) {
      int k = ja[j-1]-1;
      level[k] = max(level[k], l+1);
    }
    h_lev->ilevU[l+1] ++;
    h_lev->nlevU = max(h_lev->nlevU, l+1);
  }

  for (int i=1; i<=h_lev->nlevU; i++)
    h_lev->ilevU[i] += h_lev->ilevU[i-1];

  for (int i=0; i<n; i++)
  {
    int *k = &h_lev->ilevU[level[i]];
    h_lev->jlevU[(*k)-1] = i+1;
    (*k)++;
  }

  for (int i=h_lev->nlevU-1; i>0; i--)
    h_lev->ilevU[i] = h_lev->ilevU[i-1];

  h_lev->ilevU[0] = 1;
}

/*-------------------------------*/
void FreeLev(struct level_t *h_lev)
{
  free(h_lev->jlevL);
  free(h_lev->ilevL);
  free(h_lev->jlevU);
  free(h_lev->ilevU);
  free(h_lev->levL);
  free(h_lev->levU);
}
