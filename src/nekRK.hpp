#ifndef NEK_RK_H
#define NEK_RK_H

#include <occa.hpp>
#include <functional>
#include "mpi.h"

namespace nekRK 
{

using nekRKBuildKernel_t = std::function<occa::kernel(const std::string &fileName,
                                   const std::string &kernelName,
                                   const occa::properties &props)>;

bool isInitialized();

void init(
  const std::string& yamlPath,
  occa::device device,
  occa::properties kernel_properties,
  int blockSize,
  bool unroll_loops,
  bool unroll_loops_transport,
  bool pragma_unroll_loops,
  bool loop_gibbsexp,
  bool nonsymDij,
  bool fit_rcpDiffCoeffs,
  int align_width,
  const std::string& target,
  bool useFP64Transport,
  MPI_Comm comm,
  bool verbose = false 
  );

void init(
  const std::string& yamlPath,
  occa::device device,
  occa::properties kernel_properties,
  int blockSize,
  bool unroll_loops,
  bool unroll_loops_transport,
  bool pragma_unroll_loops,
  bool loop_gibbsexp,
  bool nonsymDij,
  bool fir_rcpDiffCoeffs,
  int align_width,
  const std::string& target,
  bool useFP64Transport,
  MPI_Comm comm,
  nekRKBuildKernel_t buildKernel,
  bool verbose = false 
  );

void build(
  double refPressure,
  double refTemperature,
  double refLength,
  double refVelocity,
  double refMass_fractions[],
  bool transport = true
);

void productionRates(
  int n_states,
  int offsetT,
  int offset,
  bool normalize,
  bool excessSpecies,
  double pressure,
  const occa::memory& o_state,
  occa::memory& o_rates,
  bool fp32 = false
);

void mixtureAvgTransportProps(
  int nStates,
  int offsetT,
  int offset,
  double pressure,
  const occa::memory& o_state,
  occa::memory& o_viscosity,
  occa::memory& o_thermalConductivity,
  occa::memory& o_densityDiffCoeffs
);
    
void thermodynamicProps(
  int n_states,
  int offsetT,
  int offset,
  double pressure,
  const occa::memory& o_state,
  occa::memory& o_rho,
  occa::memory& o_cpi,
  occa::memory& o_rhoCp,
  occa::memory& o_meanMolecularWeight
);

const std::vector<double> molecularWeights();

double refPressure();
double refTemperature();
const std::vector<double> refMassFractions();
double refDensity();
double refTime();
double refLength();
double refVelocity();
double refMeanMolecularWeight();

double refCp();
double refCv();
double refViscosity();
double refThermalConductivity();
const std::vector<double> refRhoDiffCoeffs();
const std::vector<double> refDiffCoeffs();

double Re();
double Pr();
const std::vector<double> Le();

int nSpecies();

int nActiveSpecies();

const std::vector<std::string> speciesNames();

int speciesIndex(const std::string& name);

// Only use these functions for debugging and testing purposes
namespace debugging {
}
}

#endif