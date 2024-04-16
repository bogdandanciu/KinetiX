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

  int Nstates = 1 / size;
  int Nrep = 1;
  bool transport = true;
  int offset = __NEKRK_NSPECIES__ + 1;


  // Initialize variables
  double T = 1000;
  double p = 1e5;
  double Y[__NEKRK_NSPECIES__];
  for (int k = 0; k < __NEKRK_NSPECIES__; k++)
    Y[k] = 1.0 / __NEKRK_NSPECIES__;
 
  if (rank == 0){
    printf("T: %.1f K \n", T);
    printf("p: %.1f Pa \n", p);
  }
  std::istringstream iss(species_names);
  std::vector<std::string> species_names_arr;
  for (std::string species_name; iss >> species_name;) {
    species_names_arr.push_back(species_name);
  }
  if (rank == 0){
    for (int k = 0; k < __NEKRK_NSPECIES__; k++) {
      if (k == __NEKRK_NSPECIES__ - 1)
        printf("%s = %.5f \n", species_names_arr[k].c_str(), Y[k]);
      else
        printf("%s = %.5f, ", species_names_arr[k].c_str(), Y[k]);
    }
  }


  double R;
  R = 1.380649e-23 * 6.02214076e23;

  /*** Throughput ***/
  { // Chemistry
    cfloat *ydot = (cfloat *)(_mm_malloc(Nstates * offset * sizeof(double), 64));
    cfloat wdot[__NEKRK_NSPECIES__];
    cfloat h_RT[__NEKRK_NSPECIES__];
    cfloat sc[__NEKRK_NSPECIES__];

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int i = 0; i < Nrep; i++) {
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
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    if(rank == 0)
      printf("--- Chemistry ---\n");
      printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
             (size * (double)(Nstates * (__NEKRK_NSPECIES__ + 1) * Nrep)) / elapsedTime / 1e9,
             size * Nstates);

#if 0
    for (int i=0; i< Nstates * (__NEKRK_NSPECIES__ + 1); i++)
      printf("rates[%d]: %.8f \n", i, ydot[i]);
#endif

  } 

  if (transport){ //Transport 
    cfloat *cond = (cfloat *)(_mm_malloc(Nstates * sizeof(double), 64));
    cfloat *visc = (cfloat *)(_mm_malloc(Nstates * sizeof(double), 64));
    cfloat *rhoD = (cfloat *)(_mm_malloc(Nstates * __NEKRK_NSPECIES__ * sizeof(double), 64));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for(int i=0; i<Nrep; i++) {
      for(int n=0; n<Nstates; n++) {
	const cfloat T_nondim = T/T;
        const cfloat lnT = log(T_nondim);
        const cfloat lnT2 = lnT * lnT;
        const cfloat lnT3 = lnT * lnT * lnT;
        const cfloat lnT4 = lnT * lnT * lnT * lnT;
        const cfloat sqrT = sqrt(T_nondim);
        
	cfloat wrk1[__NEKRK_NSPECIES__];
        
	cfloat iW = 0;
	{
          for (int k = 0; k < __NEKRK_NSPECIES__; k++) {
            wrk1[k] = Y[k] * nekrk_rcp_molar_mass[k];
            iW += wrk1[k];
          }
	}

        cond[n] = nekrk_conductivity(iW, lnT, lnT2, lnT3, lnT4, wrk1) * sqrT;
        visc[n] = nekrk_viscosity(lnT, lnT2, lnT3, lnT4, wrk1) * sqrT;
        nekrk_density_diffusivity(n, sqrT, lnT, lnT2, lnT3, lnT4, wrk1, rhoD, Nstates);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    if(rank == 0){
      printf("--- Transport ---\n");
      printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
             (size * (double)(Nstates*(__NEKRK_NSPECIES__ + 1) * Nrep))/ elapsedTime /1e9,
             size*Nstates);
    }

#if 0
    for (int i = 0; i < Nstates; i++){
      printf("cond[%d]: %.8f \n", i, cond[i]);
      printf("visc[%d]: %.8f \n", i, visc[i]);
    }
    for (int i = 0; i < Nstates * __NEKRK_NSPECIES__; i++)
      printf("rhoD[%d]: %.8f \n", i, rhoD[i]);
#endif
 
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}
