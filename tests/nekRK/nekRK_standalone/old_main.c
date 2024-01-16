#include "nekRK.h" 
#include <cstdio>
#include <cmath>
#include <mm_malloc.h>
#include "mpi.h"

#define ALIGN_WIDTH 64

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool chemistry = true;
  bool transport = true; 
  int Nstates = 1000000/size;
  int Nrep = 100;

  int offset = __NEKRK_NSPECIES__+1;

//  cfloat *rates = (cfloat*)( _mm_malloc(offset * sizeof(cfloat), ALIGN_WIDTH) );
//  cfloat *Ci = (cfloat*)( _mm_malloc(offset  * sizeof(cfloat), ALIGN_WIDTH) );
//  cfloat *h_RT = (cfloat*)( _mm_malloc(offset  * sizeof(cfloat), ALIGN_WIDTH) );
//  cfloat *cp_R = (cfloat*)( _mm_malloc(offset  * sizeof(cfloat), ALIGN_WIDTH) );
//  rates = static_cast<cfloat*>(__builtin_assume_aligned(rates, ALIGN_WIDTH));
//  Ci  = static_cast<cfloat*>(__builtin_assume_aligned(Ci, ALIGN_WIDTH));
//  h_RT = static_cast<cfloat*>(__builtin_assume_aligned(h_RT, ALIGN_WIDTH));
//  cp_R = static_cast<cfloat*>(__builtin_assume_aligned(cp_R, ALIGN_WIDTH));

  // big arrays should be called outside of the main loops
  int id = 1;
  alignas(ALIGN_WIDTH) double density_diffusion_packed[Nstates*__NEKRK_NSPECIES__+id];


  cfloat rates[__NEKRK_NSPECIES__ + 1];
  cfloat Ci[__NEKRK_NSPECIES__ + 1];
  cfloat h_RT[__NEKRK_NSPECIES__ + 1];
  cfloat cp_R[__NEKRK_NSPECIES__ + 1];

  for (int k=0; k<__NEKRK_NSPECIES__+1;k++)
  {
    Ci[k] = 0.0188679;
  }

  if (chemistry){
    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();
  
    for(int i=0; i<Nrep; i++) {
      for(int n=0; n<Nstates; n++) {
        const cfloat T = 300 + 700*(Nstates%3);
	rates[54] = T;
//	h_RT[53] = T;
//	cp_R[53] = T;
        nekrk_species_rates(Ci, rates);
//	nekrk_enthalpy_RT(h_RT);
//	nekrk_molar_heat_capacity_R(cp_R);
      }
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);
  
    if(rank == 0){
      printf("NekRK standalone chemistry functions. \n");
      printf("avg elapsed time: %.5f s\n", elapsedTime);
      printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
             (size * (double)(Nstates*offset*Nrep))/elapsedTime/1e9,
             size*Nstates);
    }
  }

  if (transport){
    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();
  
    for(int i=0; i<Nrep; i++) {
      for(int n=0; n<Nstates; n++) {
        const cfloat T = 300 + 700*(Nstates%3);
        const cfloat lnT = log(T);
        const cfloat lnT2 = lnT * lnT;
        const cfloat lnT3 = lnT * lnT * lnT;
        const cfloat lnT4 = lnT * lnT * lnT * lnT;
        const cfloat sqrT = sqrt(T);

	int id = 1;
	cfloat rcpMbar = 1;

	alignas(ALIGN_WIDTH) cfloat wrk1[__NEKRK_NSPECIES__];

          
        cfloat cond = nekrk_conductivityNIVT(rcpMbar, lnT, lnT2, lnT3, lnT4, wrk1) * sqrT;
	cfloat vis  = nekrk_viscosityIVT(lnT, lnT2, lnT3, lnT4, wrk1);
//	nekrk_density_diffusivity(id, sqrT, sqrT, lnT, lnT2, lnT3, lnT4, wrk1, density_diffusion_packed, n);
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);
  
    if(rank == 0){
      printf("NekRK standalone transport functions. \n");
      printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
             (size * (double)(Nstates*offset*Nrep))/elapsedTime/1e9,
             size*Nstates);
    }
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}

