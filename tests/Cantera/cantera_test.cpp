#include "mpi.h"
#include <cmath>

#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstddef>

#include "cantera/core.h"
#include "cantera/kinetics/Reaction.h"
#include "cantera/base/global.h" // provides Cantera::writelog
#include <iostream>

using namespace Cantera;

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("size: %d\n", size);
  int Nstates = 1000000/size;
  constexpr int Nrep = 5;

  int offset = 9;
  
  // Initialize reaction mechanism 
  auto sol = newSolution("h2o2.yaml", "ohmech");
  auto gas = sol->thermo();
  double temp = 1200.0;
  double pres = OneAtm;
  gas->setState_TPX(temp, pres, "H2:1, O2:1, AR:2");

  // Reaction information
  auto kin = sol->kinetics();
  int irxns = kin->nReactions();
  vector<double> q(irxns);


  for(int i=0; i<Nrep; i++) {

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();

    for(int n=0; n<Nstates; n++) {
      kin->getNetRatesOfProgress(&q[0]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = (MPI_Wtime() - startTime);

    printf("avg aggregated throughput: %.3f GDOF/s (Nstates = %d)\n",
           (size * (double)(Nstates*offset))/elapsedTime/1e9,
           size*Nstates);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}
                       
