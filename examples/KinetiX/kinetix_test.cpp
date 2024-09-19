#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <mm_malloc.h>
#include <sstream>
#include <string>
#include <vector>

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
  int offset = __KINETIX_NSPECIES__ + 1;

  // Initialize variables
  double T = 1000;
  double p = 1e5;
  double Y[__KINETIX_NSPECIES__];
  for (int k = 0; k < __KINETIX_NSPECIES__; k++)
    Y[k] = 1.0 / __KINETIX_NSPECIES__;
 
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
    for (int k = 0; k < __KINETIX_NSPECIES__; k++) {
      if (k == __KINETIX_NSPECIES__ - 1)
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
    cfloat wdot[__KINETIX_NSPECIES__];
    cfloat h_RT[__KINETIX_NSPECIES__];
    cfloat sc[__KINETIX_NSPECIES__];

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
        cfloat wrk1[__KINETIX_NSPECIES__];
        {
          cfloat iW = 0;
          for (int k = 0; k < __KINETIX_NSPECIES__; k++) {
            wrk1[k] = Y[k] * kinetix_rcp_molar_mass[k];
            iW += wrk1[k];
          }
          W = 1 / iW;
        }
        cfloat rho = p * W / (R * T);
        for (int k = 0; k < __KINETIX_NSPECIES__; k++)
          sc[k] = wrk1[k] * rho;
        kinetix_species_rates(logT,T, T2, T3, T4, rcpT, sc, wdot);
        for (int k = 0; k < __KINETIX_NSPECIES__; k++)
          ydot[n + k * Nstates] = wdot[k] * kinetix_molar_mass[k];
    
        kinetix_enthalpy_RT(T, T2, T3, T4, rcpT, h_RT);
        cfloat sum_h_RT = 0;
        for (int k = 0; k < __KINETIX_NSPECIES__; k++)
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
             (size * (double)(Nstates * (__KINETIX_NSPECIES__ + 1) * Nrep)) / elapsedTime / 1e9,
             size * Nstates);

#if 0
    for (int i=0; i< Nstates * (__KINETIX_NSPECIES__ + 1); i++)
      printf("rates[%d]: %.9e \n", i, ydot[i]);
#endif

  } 

  if (transport){ //Transport 
    cfloat *cond = (cfloat *)(_mm_malloc(Nstates * sizeof(double), 64));
    cfloat *visc = (cfloat *)(_mm_malloc(Nstates * sizeof(double), 64));
    cfloat *Di = (cfloat *)(_mm_malloc(Nstates * __KINETIX_NSPECIES__ * sizeof(double), 64));
    cfloat *rhoDi = (cfloat *)(_mm_malloc(Nstates * __KINETIX_NSPECIES__ * sizeof(double), 64));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for(int i=0; i<Nrep; i++) {
      for(int n=0; n<Nstates; n++) {
        const cfloat lnT = log(T);
        const cfloat lnT2 = lnT * lnT;
        const cfloat lnT3 = lnT * lnT * lnT;
        const cfloat lnT4 = lnT * lnT * lnT * lnT;
        const cfloat sqrT = sqrt(T);
        
	cfloat wrk1[__KINETIX_NSPECIES__];
        
	cfloat iW = 0;
	{
          for (int k = 0; k < __KINETIX_NSPECIES__; k++) {
            wrk1[k] = Y[k] * kinetix_rcp_molar_mass[k];
            iW += wrk1[k];
          }
	}
	cfloat W = 1/iW; 

        cond[n] = kinetix_conductivity(iW, lnT, lnT2, lnT3, lnT4, wrk1) * sqrT;
        visc[n] = kinetix_viscosity(lnT, lnT2, lnT3, lnT4, wrk1) * sqrT;
        kinetix_diffusivity(W, p, T*sqrT, lnT, lnT2, lnT3, lnT4, wrk1, Di);
        
	cfloat rho = p * W / (R * T);
	for(int k = 0; k < __KINETIX_NSPECIES__; k++) {
          unsigned int idx = k*Nstates+n;
          rhoDi[idx] = rho * Di[k];
        }

      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    if(rank == 0){
      printf("--- Transport ---\n");
      printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
             (size * (double)(Nstates*(__KINETIX_NSPECIES__ + 1) * Nrep))/ elapsedTime /1e9,
             size*Nstates);
    }

#if 0
    for (int i = 0; i < Nstates; i++){
      printf("cond[%d]: %.9e \n", i, cond[i]);
      printf("visc[%d]: %.9e \n", i, visc[i]);
    }
    for (int i = 0; i < Nstates * __KINETIX_NSPECIES__; i++)
      printf("rhoDi[%d]: %.9e \n", i, rhoDi[i]);
#endif
 
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}
