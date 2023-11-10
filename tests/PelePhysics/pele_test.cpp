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
  int offset = NUM_SPECIES + 1;

  // Initialize variables
  double T = 1000;
  double p = 1e5;
  double Y[NUM_SPECIES];
  for (int k = 0; k< NUM_SPECIES; k++)
    Y[k] = 1.0/NUM_SPECIES;
  printf("T: %.1f K \n", T);
  printf("p: %.1f Pa \n", p);
//  for (int i = 0; i < nSpecies; i++) {
//    if (i == nSpecies - 1)
//      printf("%s = %.5f \n", gas->speciesName(i).c_str(), Y[i]);
//    else
//      printf("%s = %.5f, ", gas->speciesName(i).c_str(), Y[i]);
//  }


  double R, Rc, Patm;
  CKRP(R, Rc, Patm);
  R /= 1e7;
  printf("R: %.5f \n", R);

  double Wk[NUM_SPECIES], iWk[NUM_SPECIES];
  get_mw(Wk);
  get_imw(iWk);
  for (int k = 0; k< NUM_SPECIES; k++)
    printf("Wk[%d]: %.8f, iWk[%d]: %.8f \n", k, Wk[k], k, iWk[k]);

  double *ydot = (double *)(_mm_malloc(Nstates * offset * sizeof(double), 64));
  double wdot[NUM_SPECIES];
  double h_RT[NUM_SPECIES];
  double sc[NUM_SPECIES];

  /*** Throughput ***/
  for (int i = 0; i < Nrep; i++) {

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for (int n = 0; n < Nstates; n++) {
      double W;
      double wrk1[NUM_SPECIES];
      {
        double iW = 0;
        for (int k = 0; k < NUM_SPECIES; k++) {
          wrk1[k] = Y[k] * iWk[k];
          iW += wrk1[k];
        }
        for (int k = 0; k < NUM_SPECIES; k++)
 	  printf("Y[%d]: %.8f \n", k, Y[k]);
        W = 1/iW; 
        printf("W: %.5f \n", W);
        printf("iW: %.5f \n", iW);
      }
      double rho = p * W / (R * T);
      printf("rho: %.5f \n", rho);
      for (int k = 0; k < NUM_SPECIES; k++)
	sc[k] = wrk1[k] * rho;
      for (int k = 0; k < NUM_SPECIES; k++)
	printf("sc[%d]: %.8f \n", k, sc[k]);
      productionRate(wdot, sc, T);
      for (int k = 0; k < NUM_SPECIES; k++)
	printf("wdot[%d]: %.8f \n", k, wdot[k]);
      for (int k = 0; k < NUM_SPECIES; k++)
        ydot[n + (k + 1) * Nstates] = wdot[k] * Wk[k];

      speciesEnthalpy(h_RT, T);
      for (int k = 0; k < NUM_SPECIES; k++)
	printf("h_RT[%d]: %.8f \n", k, h_RT[k]);
      double sum_h_RT = 0;
      for (int k = 0; k < NUM_SPECIES; k++)
        sum_h_RT += wdot[k] * h_RT[k];
      double ratesFactorEnergy = -R*T;
      ydot[n] = ratesFactorEnergy * sum_h_RT; 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
           (size * (double)(Nstates * offset)) / elapsedTime / 1e9,
           size * Nstates);
  }

#if 1
  for (int i=0; i< Nstates * offset; i++)
    printf("rates[%d]: %.8f \n", i, ydot[i]);
#endif

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
