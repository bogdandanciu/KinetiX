#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <sstream>
#include <string>

#include "cantera/base/global.h" // provides Cantera::writelog
#include "cantera/core.h"
#include "cantera/kinetics/Reaction.h"
#include <iostream>

using namespace Cantera;

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("size: %d\n", size);

  int err = 0;
  int n_states = 100000;
  int n_rep = 20;
  std::string mech;

  while (1) {
    static struct option long_options[] = {
        {"n-states", required_argument, 0, 's'},
        {"n-repetitions", required_argument, 0, 'r'},
        {"mechanism", required_argument, 0, 'm'},
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

    default:
      err++;
    }
  }

  if (mech.size() < 1)
    err++;

  if (err > 0) {
    if (rank == 0)
      printf("Usage: ./cantera_test  [--n-states n] [--n-repetitions n] "
             "[--mechanism f] \n");
    exit(EXIT_FAILURE);
  }

  int Nstates = n_states / size;
  int Nrep = n_rep;

  // Initialize reaction mechanism
  auto sol = newSolution(mech);
  auto gas = sol->thermo();
  double temp = 2800.0;
  double pres = OneAtm;
  gas->setState_TPX(temp, pres, "H2:1, O2:1, AR:2");

  // Reaction information
  auto kin = sol->kinetics();
  int nSpecies = gas->nSpecies();
  int offset = nSpecies + 1;

  // Initialize species production rates
  std::vector<double> netProductionRates(nSpecies, 0.0);

  for (int i = 0; i < Nrep; i++) {

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int n = 0; n < Nstates; n++) {
      // Get species net production rates
      kin->getNetProductionRates(netProductionRates.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
           (size * (double)(Nstates * offset)) / elapsedTime / 1e9,
           size * Nstates);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
