#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <sstream>
#include <string>

#include "cantera/base/global.h"
#include "cantera/core.h"
#include "cantera/kinetics/Reaction.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/IdealGasPhase.h"
#include "cantera/transport/MixTransport.h"
#include <iostream>

using namespace Cantera;

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    printf("size: %d\n", size);
  int err = 0;
  int n_states = 10;
  int n_rep = 1;
  bool transport = false;
  bool print_states = false;
  std::string mech;

  while (1) {
    static struct option long_options[] = {
        {"n-states", required_argument, 0, 's'},
        {"n-repetitions", required_argument, 0, 'r'},
        {"mechanism", required_argument, 0, 'm'},
        {"transport", no_argument, 0, 't'},
        {"print-states", no_argument, 0, 'p'},
    };

    int option_index = 0;
    int args = getopt_long(argc, argv, "s:", long_options, &option_index);

    if (args == -1)
      break;

    switch (args) {
    case 's':
      n_states = std::stoi(optarg);
      break;
    case 'r':
      n_rep = std::stoi(optarg);
      break;
    case 'm':
      mech.assign(optarg);
      break;
    case 't':
      transport = true;
      break;
    case 'd':
      print_states = true;
      break;

    default:
      err++;
    }
  }

  if (mech.size() < 1)
    err++;

  if (err > 0) {
    if (rank == 0)
      printf("Usage: ./cantera_test  --n-states n --n-repetitions n --mechanism f "
             "[--transport] [--print-states] \n");
    exit(EXIT_FAILURE);
  }

  int nStates = n_states / size;
  int nRep = n_rep;

  // Initialize reaction mechanism
  auto sol = newSolution(mech);
  auto gas = sol->thermo();
  auto trans = sol->transport();
  double T = 1000.0;
  double p = 1e5;
  int nSpecies = gas->nSpecies();
  double Y[nSpecies];
  for (int k = 0; k < nSpecies; k++) {
    Y[k] = 1.0 / nSpecies;
  }
  gas->setState_TPY(T, p, Y);
  if (rank == 0){
    printf("T: %.1f K \n", T);
    printf("p: %.1f Pa \n", p);
    for (int k = 0; k < nSpecies; k++) {
      if (k == nSpecies - 1)
        printf("%s = %.5f \n", gas->speciesName(k).c_str(), Y[k]);
      else
        printf("%s = %.5f, ", gas->speciesName(k).c_str(), Y[k]);
    }
  }

  // Initialize reaction kinetics
  auto kin = sol->kinetics();
  int nReactions = kin->nReactions();

  // Initialize states vector
  int offset = nSpecies + 1;

  /*** Throughput ***/
  { //Chemistry
    double *ydot = (double *)(_mm_malloc(nStates * offset * sizeof(double), 64));
    
    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int i = 0; i < nRep; i++) {
      for (int n = 0; n < nStates; n++) {
        double wdot[nSpecies];
        double h_RT[nSpecies];
        gas->setState_TPY(T, p, Y);

        kin->getNetProductionRates(wdot);
        for (int k = 0; k < nSpecies; k++)
          ydot[n + (k + 1) * nStates] = wdot[k] * gas->molecularWeight(k);

        gas->getEnthalpy_RT(h_RT);
        double sum_h_RT = 0;
        for (int k = 0; k < nSpecies; k++)
          sum_h_RT += wdot[k] * h_RT[k];
        double ratesFactorEnergy = -GasConstant * T;
        ydot[n] = ratesFactorEnergy * sum_h_RT;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    if (rank == 0){
      printf("--- Chemistry ---\n");
      printf("avg elapsed time: %.5f s\n", elapsedTime);
      printf("avg aggregated throughput: %.3f GRXN/s (nStates = %d)\n",
             (size * (double)(nStates * nReactions) * nRep) / elapsedTime / 1e9,
             size * nStates);
    }

    if (print_states) {
      for (int i = 0; i < nStates * offset; i++)
        printf("ydot[%d]: %.9e \n", i, ydot[i]);
    }
  }

  if (transport) { //Transport
    double *cond = (double *)(_mm_malloc(nStates * sizeof(double), 64));
    double *visc = (double *)(_mm_malloc(nStates * sizeof(double), 64));
    double *rhoD = (double *)(_mm_malloc(nStates * nSpecies * sizeof(double), 64));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int i = 0; i < nRep; i++) {
      for (int n = 0; n < nStates; n++) {
        gas->setState_TPY(T, p, Y);

        visc[n] = trans->viscosity();
        cond[n] = trans->thermalConductivity();

	double wrk1[nSpecies];
        trans->getMixDiffCoeffs(wrk1);
	double rho = gas->density();
        for (int k = 0; k < nSpecies; k++){
	  unsigned int idx = k*nStates+n;
	  rhoD[idx] = rho * wrk1[k];
	}

      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    if (rank == 0){
      printf("--- Transport properties ---\n");
      printf("avg elapsed time: %.5f s\n", elapsedTime); printf("avg aggregated throughput: %.3f GDOF/s (nStates = %d)\n",
             (size * (double)(nStates * (nSpecies + 2)) * nRep) / elapsedTime / 1e9,
             size * nStates);
    }
    
    if (print_states){
      for (int i = 0; i < nStates; i++){
        printf("cond[%d]: %.9e \n", i, cond[i]);
        printf("visc[%d]: %.9e \n", i, visc[i]);
      }
      for (int i = 0; i < nStates * nSpecies; i++)
        printf("rhoD[%d]: %.9e \n", i, rhoD[i]);
    }
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
