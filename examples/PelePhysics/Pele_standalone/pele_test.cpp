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
  std::srand(std::time(0));

  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    printf("size: %d\n", size);

  int err = 0;
  int n_states = 1000000;
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
  int offset = NUM_SPECIES + 1;

  // Initialize variables
  double T = 1000;
  double p = 1e5;
  double Y[NUM_SPECIES];
  for (int k = 0; k < NUM_SPECIES; k++)
    Y[k] = 1.0 / NUM_SPECIES;
  if (rank == 0){
    printf("T: %.1f K \n", T);
    printf("p: %.1f Pa \n", p);
  }
  amrex::Vector<std::string> species_names;
  CKSYMS_STR(species_names);
  if (rank == 0){
    for (int k = 0; k < NUM_SPECIES; k++) {
      if (k == NUM_SPECIES - 1)
        printf("%s = %.5f \n", species_names[k].c_str(), Y[k]);
      else
        printf("%s = %.5f, ", species_names[k].c_str(), Y[k]);
    }
  }

  double R, Rc, Patm;
  CKRP(R, Rc, Patm);
  R /= 1e7;

  double Wk[NUM_SPECIES], iWk[NUM_SPECIES];
  get_mw(Wk);
  get_imw(iWk);

  double *ydot = (double *)(_mm_malloc(Nstates * offset * sizeof(double), 64));
  double *states = (double *)(_mm_malloc(Nstates * offset * sizeof(double), 64));
  for (int id = 0; id < Nstates; id++) {
    states[id + 0*Nstates] = T;
    for (int k = 0; k < NUM_SPECIES; k++) {
      states[id + (k+1)*Nstates] = Y[k];
    }
  }


  /*** Throughput ***/
  MPI_Barrier(MPI_COMM_WORLD);
  const auto startTime = MPI_Wtime();
   
  for (int i = 0; i < Nrep; i++) {

    for (int n = 0; n < Nstates; n++) {
      T = states[n];

      double wdot[NUM_SPECIES];
      double h_RT[NUM_SPECIES];
      double sc[NUM_SPECIES];
    
      double W;
      double wrk1[NUM_SPECIES];
      {
        double iW = 0;
        for (int k = 0; k < NUM_SPECIES; k++) {
          wrk1[k] = Y[k] * iWk[k];
          iW += wrk1[k];
        }
        W = 1 / iW;
      }
      double rho = p * W / (R * T);
      for (int k = 0; k < NUM_SPECIES; k++)
        sc[k] = wrk1[k] * rho;
      productionRate(wdot, sc, T);
      for (int k = 0; k < NUM_SPECIES; k++)
        ydot[n + (k + 1) * Nstates] = wdot[k] * Wk[k];

      speciesEnthalpy(h_RT, T);
      double sum_h_RT = 0;
      for (int k = 0; k < NUM_SPECIES; k++)
        sum_h_RT += wdot[k] * h_RT[k];
      double ratesFactorEnergy = -R * T;
      ydot[n] = ratesFactorEnergy * sum_h_RT;
    }

  }

  MPI_Barrier(MPI_COMM_WORLD);
  const auto elapsedTime = (MPI_Wtime() - startTime);

  if (rank == 0)
    printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
           (size * (double)(Nstates * offset) * Nrep) / elapsedTime / 1e9,
           size * Nstates);

#if 0
  for (int i=0; i< Nstates * offset; i++)
    printf("rates[%d]: %.9e \n", i, ydot[i]);
#endif

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
