#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <mm_malloc.h>
#include <sstream>
#include <string>

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    printf("size: %d\n", size);

//  int Nstates = 1000000 / size;
  int Nstates = 1;
  int Nrep = 20;
  int offset = __NEKRK_NSPECIES__ + 1;

  // Initialize variables
  double T = 1000;
  double p = 1e5;
  double Y[__NEKRK_NSPECIES__];
  for (int k = 0; k < __NEKRK_NSPECIES__; k++)
    Y[k] = 1.0 / __NEKRK_NSPECIES__;

  double R;
  R = 1.380649e-23 * 6.02214076e23;

  cfloat *ydot = (cfloat *)(_mm_malloc(Nstates * offset * sizeof(double), 64));
  cfloat wdot[__NEKRK_NSPECIES__];
  cfloat h_RT[__NEKRK_NSPECIES__];
  cfloat sc[__NEKRK_NSPECIES__];


  /*** Throughput ***/
  for(int ii=0; ii<4; ii++) {
//    if(ii==0) Nstates = 100/size;
//    if(ii==1) Nstates = 1000000/size;
//    if(ii==2) Nstates = 100000/size;
//    if(ii==3) Nstates = 10000/size;
    Nstates = 1;

    for (int i = 0; i < Nrep; i++) {
  
      MPI_Barrier(MPI_COMM_WORLD);
      const auto startTime = MPI_Wtime();
  
      for (int n = 0; n < Nstates; n++) {
        const cfloat rcpT = 1 / T;
        const cfloat logT = log(T);
        const cfloat T2 = T * T;
        const cfloat T3 = T * T * T;
        const cfloat T4 = T * T * T * T;
  
        cfloat W;
        cfloat wrk1[__NEKRK_NSPECIES__];
        {
          cfloat iW = 0;
          for (int k = 0; k < __NEKRK_NSPECIES__; k++) {
            wrk1[k] = Y[k] * nekrk_rcp_molar_mass[k];
            iW += wrk1[k];
          }
          W = 1 / iW;
        }
        cfloat rho = p * W / (R * T);
        for (int k = 0; k < __NEKRK_NSPECIES__; k++)
          sc[k] = wrk1[k] * rho;
        nekrk_species_rates(logT,T, T2, T3, T4, rcpT, sc, wdot);
        for (int k = 0; k < __NEKRK_NSPECIES__; k++)
          ydot[n + k * Nstates] = wdot[k] * nekrk_molar_mass[k];
  
        nekrk_enthalpy_RT(logT, T, T2, T3, T4, rcpT, h_RT);
        cfloat sum_h_RT = 0;
        for (int k = 0; k < __NEKRK_NSPECIES__; k++)
          sum_h_RT += wdot[k] * h_RT[k];
        cfloat ratesFactorEnergy = -R * T;
        ydot[n] = ratesFactorEnergy * sum_h_RT;
      }
  
      MPI_Barrier(MPI_COMM_WORLD);
      const auto elapsedTime = (MPI_Wtime() - startTime);

      if(rank == 0 && ii)
        printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
               (size * (double)(Nstates * offset)) / elapsedTime / 1e9,
               size * Nstates);
    }
  } 

#if 1
  for (int i=0; i< Nstates * offset; i++)
    printf("rates[%d]: %.8f \n", i, ydot[i]);
#endif

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
