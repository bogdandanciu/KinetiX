#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <mm_malloc.h>
#include <sstream>
#include <string>

#include "mechanism.H"

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
  int offset = 9;

  // Initialize variables
  double *wdot = (double *)(_mm_malloc(offset * sizeof(double), 64));
  double *sc = (double *)(_mm_malloc(offset * sizeof(double), 64));
  double T = 1000;

  for (int i = 0; i < offset; i++)
    sc[i] = 1;

  for (int i = 0; i < Nrep; i++) {

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int n = 0; n < Nstates; n++) {
      // Get species net production rates
      productionRate(wdot, sc, T);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
           (size * (double)(Nstates * offset)) / elapsedTime / 1e9,
           size * Nstates);
  }

#if 0
  for (int i=0; i<offset; i++)
    printf("wdot[%d]: %.3f \n", i, wdot[i]);
#endif

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
