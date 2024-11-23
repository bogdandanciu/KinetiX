#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

#include "mpi.h"

#include "occa.hpp"
#include "kinetix.hpp"

#define DEBUG

using dfloat = double;

occa::device device;
bool debug = false;
int n_states;
int n_species;
int n_reactions;
int n_active_species;
int n_inert_species;

double *ref_mole_fractions; 
double *ref_mass_fractions;
double ref_molar_mass;
double ref_pressure; 
double ref_temperature;

std::vector<double> ci_rho;
std::vector<double> ci_cp_mean;
std::vector<double> ci_hrr;
std::vector<double> ci_conductivity;
std::vector<double> ci_viscosity;
std::vector<std::vector<double>> ci_cp_i;
std::vector<std::vector<double>> ci_rates;
std::vector<std::vector<double>> ci_rhoD;


void checkValue(dfloat value)
{
  if(std::isnan(value) || std::isinf(value)) {
    printf("Detected invalid value: %e \n", value);
    //exit(EXIT_FAILURE);
  }
}

dfloat relErr(dfloat a, dfloat b)
{
  checkValue(a);
  checkValue(b);
  return std::abs((a-b)/b);
}

std::vector<std::string> split (std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
    token = s.substr (pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back (token);
  }

  res.push_back (s.substr (pos_start));
  return res;
}

std::string read(const std::string& path)
{
  std::ifstream input_file(path);
  return std::string((std::istreambuf_iterator<char>(input_file)),
                     std::istreambuf_iterator<char>());
}

bool checkThermo(occa::memory& o_rho,
                 occa::memory& o_cp_i,
                 occa::memory& o_rhoCp)
{
  std::vector<dfloat> rho(n_states);
  o_rho.copyTo(rho.data());
  std::vector<dfloat> cp_i(n_states * n_species); 
  o_cp_i.copyTo(cp_i.data());
  std::vector<dfloat> rhoCp(n_states * n_species); 
  o_rhoCp.copyTo(rhoCp.data());

  for (int id = 0; id < n_states; id++) {
    const auto rho_SI = rho[id];
    const auto ciVal = ci_rho[id];
    if(debug)
      printf("rho: Cantera %.15f KinetiX %.15f relative error %e\n",
             ciVal,
             rho_SI,
             relErr(rho_SI, ciVal));
  }

  for (int k = 0; k < n_species; k++) {
    for (int id = 0; id < n_states; id++) {
      const auto cpi_SI = cp_i[k * n_states + id];
      const auto ciVal = ci_cp_i[id][k];
      if(debug)
        printf("cp[%d]: Cantera %e KinetiX %e relative error %e\n",
               k,
               ciVal,
               cpi_SI,
               relErr(cpi_SI, ciVal));
    }
  }

  for (int id = 0; id < n_states; id++) {
    const auto ciVal = ci_rho[id] * ci_cp_mean[id];
    const auto rhoCp_SI = rhoCp[id];
    if(debug)
      printf("rhoCp: Cantera %e KinetiX %e relative error %e\n",
             ciVal, 
             rhoCp_SI,
             relErr(rhoCp_SI, ciVal));
  }

  auto allPassed = true;

  for (int id = 0; id < n_states; id++) {
    std::vector<double> e;
    auto e_rho = relErr(rho[id], ci_rho[id]);
    e.push_back(e_rho);
    auto e_rhoCp = relErr(rhoCp[id], ci_rho[id] * ci_cp_mean[id]); 
    e.push_back(e_rhoCp);
    for(int k = 0; k < n_species; k++) {
      const auto ciVal = ci_cp_i[id][k];
      auto e_cpi = relErr(cp_i[k * n_states + id], ciVal);
      e.push_back(e_cpi);
    }

    auto errInf = *(std::max_element(e.begin(), e.end()));
    const auto rtol = 5e-7;
    const auto passed = (errInf < rtol);
    allPassed &= passed;
    printf("thermoCoeffs error_inf: %e < %e (%s)\n",
           errInf,
           rtol,
           (passed) ? "passed" : "failed");
  }

  return allPassed;
}

bool checkRates(occa::memory& o_rates,
                bool single_precision)
{
  std::vector<dfloat> rates(n_states * (n_species + 1));
  o_rates.copyTo(rates.data());

  auto allPassed = true;
  for (int id = 0; id < n_states; id++) {
    std::vector<double> errors;

    const auto ciVal= ci_hrr[id];
    const auto hrrSI = rates[id];
    const auto e = relErr(hrrSI, ciVal);
    errors.push_back(e); 
    if(debug)
      printf("HRR    Cantera %+.15e KinetiX %+.15e relative error %e\n",
             ciVal,
             hrrSI,
             e);

    for (int k = 0; k < n_active_species; k++) {
        const auto ciVal = ci_rates[id][k];
        const auto mass_production_rate = rates[id + (k+1) * n_states];
        const auto molar_rate =  
          mass_production_rate / (kinetix::molecularWeights()[k] * kinetix::refMeanMolecularWeight());
 
        const auto e = (std::abs(ciVal) > 1e-50) ? 
                       relErr(molar_rate, ciVal) : std::abs(molar_rate);
        errors.push_back(e); 
        if(debug) 
          printf("%-6s Cantera %+.15e KinetiX %+.15e relative error %e\n",
                 kinetix::speciesNames()[k].c_str(),
                 ciVal,
                 molar_rate,
                 e);
    }

    const auto errInf = *(std::max_element(errors.begin(), errors.end()));
    auto rtol = (single_precision) ? 0.02 : 2e-08;
    if(id == 2 || id == 3) rtol = (single_precision) ? 0.02 : 5e-5;
    const auto passed = (errInf < rtol);
    allPassed &= passed;
    printf("rates error_inf: %e < %e (%s)\n",
           errInf,
           rtol,
           (passed) ? "passed" : "failed");
  }

  return allPassed;
}

bool checkTransport(occa::memory& o_conductivity,
                    occa::memory& o_viscosity,
                    occa::memory& o_rhoD)
{
  auto conductivity = std::vector<dfloat>(n_states);
  o_conductivity.copyTo(conductivity.data());
  auto viscosity = std::vector<dfloat>(n_states);
  o_viscosity.copyTo(viscosity.data());
  auto rhoD = std::vector<dfloat>(n_species * n_states);
  o_rhoD.copyTo(rhoD.data());

  auto allPassed = true;
  for (int id = 0; id < n_states; id++) {
    std::vector<double> errors;
    {
      auto e = relErr(conductivity[id], ci_conductivity[id]);
      errors.push_back(e);
      if(debug)
        printf("conductivity: Cantera %e KinetiX %e relative error %e\n",
               ci_conductivity[id],
               conductivity[id],
               e);
    }

    {
      auto e = relErr(viscosity[id], ci_viscosity[id]);
      errors.push_back(e);
      if(debug)
        printf("viscosity: Cantera %e KinetiX %e relative error %e\n",
               ci_viscosity[id],
               viscosity[id],
               e);
    }

    {
      for(int k = 0; k < n_species; k++) {
        const auto ci_rhoDi = ci_rhoD[id][k];
        const auto rhoDi = rhoD[k * n_states + id];
        const auto e = relErr(rhoDi, ci_rhoDi);
        if(debug)
          printf("rhoD(%d): Cantera %e KinetiX %e relative error %e\n",
                 k, ci_rhoDi, rhoDi, e);
        errors.push_back(e);
      }
    }

    const auto errInf = *(std::max_element(errors.begin(), errors.end()));
    const auto rtol = 1e-3;
    const auto passed = (errInf < rtol);
    allPassed &= passed;
    printf("transport error_inf: %e < %e (%s)\n",
           errInf,
           rtol,
           (passed) ? "passed" : "failed");

  }

  return allPassed;
}

void loadCiStateFromFile(const std::string& ciState,
                         const std::string& mech, 
                         double& pressure_Pa, 
                         double& temperature_K, 
                         std::vector<double>& mole_fractions,
                         std::vector<double>& molar_masses) 
{
  const auto data = std::string(
                      std::string(getenv("KINETIX_PATH")) +
                      "/kinetix/ci_data/" + mech + "." + ciState + ".cantera");

  std::cout << "Reading ci data from " << data;
  std::vector<std::string> lines = split(read(data),"\n");
  if (lines.size() < 13) {
    std::cout << "\ndata file does not exist or is corrupt!" << std::endl;
    exit(EXIT_FAILURE);
  }

  molar_masses.clear();

  // check if species ordering matches 
  {
    auto value = split(lines[1]," ");
    for(int k = 0; k < n_species; k++) {
      molar_masses.push_back(std::stod(value[k]));
      const auto w = molar_masses[k];
      const auto wC = kinetix::molecularWeights()[k] * kinetix::refMeanMolecularWeight();
      auto e = abs(w - wC)/wC ;
      if(e > 1e-7) {
        printf("\nmolar mass mismatch [%d]: w %e cantera %e relative error %e\n", 
               k, w, wC, e);    
        exit(EXIT_FAILURE);
      }
    }
  }

  // check if number of species match
  if(molar_masses.size() != n_species) { 
    std::cerr << "\nNumber of species does not match!\n"; 
    exit(EXIT_FAILURE);
  }

  temperature_K = std::stod(lines[2]);
  pressure_Pa = std::stod(lines[3]);

  mole_fractions.clear();
  {
    const auto Xi = split(lines[4]," ");
    for(int k = 0; k < n_species; k++)
      mole_fractions.push_back(std::stod(Xi[k]));
    double sum = 0.;
    for(int k = 0; k < n_species; k++)
      sum += mole_fractions[k]; 
    for (int k = 0; k < n_species; k++) 
      mole_fractions[k] /= sum;
  }

  ci_rho.push_back(std::stod(lines[5]));

  double mean_molar_mass = 0;
  for (int k = 0; k < n_species; k++) 
    mean_molar_mass += mole_fractions[k] * molar_masses[k];

  ci_cp_mean.push_back(std::stod(lines[6]) / mean_molar_mass);
  ci_cp_i.push_back((
    {
      std::vector<double> ci_cp_i;
      const auto value = split(lines[7]," ");
      for(int k = 0; k < n_species; k++)
        ci_cp_i.push_back(std::stod(value[k]) / molar_masses[k]);
      ci_cp_i;
    }
  ));

  ci_rates.push_back((
    {
      std::vector<double> ci_rates;
      {
        const auto value = split(lines[8]," ");
        for(int k = 0; k < n_species; k++)
          ci_rates.push_back(std::stod(value[k]));
      }
      ci_rates;
    }
  ));

  ci_hrr.push_back(std::stod(lines[9]));
  ci_conductivity.push_back(std::stod(lines[10]));
  ci_viscosity.push_back(std::stod(lines[11]));

  ci_rhoD.push_back((
    {
      std::vector<double> ci_rhoD;
      {
        const auto value = split(lines[12]," ");
        for(int k = 0; k < n_species; k++)
          ci_rhoD.push_back(std::stod(value[k]));
      }
    ci_rhoD;
    }
  ));

  std::cout << " ... done" << std::endl;
}

int main(int argc, char** argv)
{
  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int err = 0;
  std::string threadModel;
  n_states = 100000;
  int mode = 0;
  std::string tool = "KinetiX";
  int blockSize = 512;
  int nRep = 50;
  bool single_precision = false;
  int ci = 0;
  int deviceId = 0;
  int deviceIdFlag = 0;
  int unroll_loops = false;
  bool loop_gibbsexp = false;
  bool group_rxnUnroll = false;
  bool group_vis = false;
  bool nonsymDij = false;
  bool fit_rcpDiffCoeffs = false;
  std::string mech;

  debug = false;

  while(1) {
    static struct option long_options[] =
    {
      {"mode", required_argument, 0, 'e'},
      {"backend", required_argument, 0, 'd'},
      {"tool", required_argument, 0, 't'},
      {"n-states", required_argument, 0, 'n'},
      {"block-size", required_argument, 0, 'b'},
      {"n-repetitions", required_argument, 0, 'r'},
      {"single-precision", no_argument, 0, 'p'},
      {"debug", no_argument, 0, 'g'},
      {"cimode", required_argument, 0, 'c'},
      {"yaml-file", required_argument, 0, 'f'},
      {"device-id", required_argument, 0, 'i'},
      {"unroll-loops", no_argument, 0, 'u'},
      {"group-rxnUnroll", no_argument, 0, 'a'},
      {"loop-gibbsexp", no_argument, 0, 'x'},
      {"group-vis", no_argument, 0, 'v'},
      {"nonsymDij", no_argument, 0, 's'},
      {"fit-rcpDiffCoeffs", no_argument, 0, 'o'},
      {0, 0, 0, 0}
    };

    int option_index = 0;
    int c = getopt_long (argc, argv, "s:", long_options, &option_index);

    if (c == -1)
      break;

    switch(c) {
    case 'e':
      mode = std::stoi(optarg);
      break;
    case 't':
      tool.assign(optarg);
      break;
    case 'd':
      threadModel.assign(strdup(optarg));
      break;
    case 'n':
      n_states = std::stoi(optarg) / size;
      break;
    case 'b':
      blockSize = std::stoi(optarg);
      break;
    case 'r':
      nRep = std::stoi(optarg);
      break;
    case 'p':
      single_precision = true;
      break;
    case 'c':
      ci = std::stoi(optarg);
      break;
    case 'g':
      debug = true;
      break;
    case 'f':
      mech.assign(optarg);
      break;
    case 'i':
      deviceId = std::stoi(optarg);
      deviceIdFlag = 1;
      break;
    case 'u':
      unroll_loops = true;
      break;
    case 'x':
      loop_gibbsexp = true;
      break;
    case 'a':
      group_rxnUnroll = true;
      break;
    case 'v':
      group_vis = true;
      break;
    case 's':
      nonsymDij = true;
      break;
    case 'o':
      fit_rcpDiffCoeffs = true;
      break;

    default:
      err++;
    }
  }

  if(threadModel.size() < 1) err++;
  if(tool == "KinetiX" && mech.size() < 1) err++;

  if(err > 0) {
    if(rank == 0)
      printf("Usage: ./bk --backend SERIAL|CUDA|HIP|DPCPP --n-states n --yaml-file s"
             "[--mode 1|2] [--tool s] [--n-repetitions n] [--single-precision] [--cimode n] [--debug] "
	     "[--block-size  n] [--device-id  n] [--unroll-loops] [--loop-gibbsexp] "
	     "[--group-rxnUnroll] [--group-vis] [--nonsymDij] [--fit-rcpDiffCoeffs] \n");
    exit(EXIT_FAILURE);
  }

  std::vector<std::string> ciStates;
  if(ci) {
    if(size != 1) {
      printf("Running ci mode requires single MPI rank!");
      exit(EXIT_FAILURE);
    }

    if(ci == 1) {
      ciStates.push_back("initial");
      ciStates.push_back("ignition");
      ciStates.push_back("final");
    } 
    if (ci == 2 && mech.find("gri30") != std::string::npos){
      ciStates.push_back("ignition.highP");
    }
    n_states = ciStates.size();
    nRep = 0;
  } 

  // setup device
  if(!deviceIdFlag) {
    deviceId = 0;
    long int hostId = gethostid();
    long int* hostIds = (long int*) calloc(size, sizeof(long int));
    MPI_Allgather(&hostId, 1, MPI_LONG, hostIds,1, MPI_LONG, MPI_COMM_WORLD);
    for (int r = 0; r < rank; r++) {
      if (hostIds[r] == hostId) deviceId++;
    }
  }

  {
    char deviceConfig[BUFSIZ];
    const int platformId = 0;
    if(strstr(threadModel.c_str(), "CUDA"))
      sprintf(deviceConfig, "{mode: 'CUDA', device_id: %d}",deviceId);
    else if(strstr(threadModel.c_str(),  "HIP"))
      sprintf(deviceConfig, "{mode: 'HIP', device_id: %d}",deviceId);
    else if(strstr(threadModel.c_str(),  "DPCPP"))
      sprintf(deviceConfig, "{mode: 'dpcpp', device_id: %d, platform_id: %d}", deviceId, platformId);
    else
      sprintf(deviceConfig, "{mode: 'Serial'}");
 
    device.setup(std::string(deviceConfig));

    char toolConfig[BUFSIZ];
    if(strstr(tool.c_str(), "Pele"))
      sprintf(toolConfig, "{Using: Pele routines}");
    else
      sprintf(toolConfig, "{Using: KinetiX routines}");

    if (!getenv("OCCA_CACHE_DIR")) {
      char buf[4096];
      char * ret = getcwd(buf, sizeof(buf));
      const std::string cache_dir = std::string(buf) + "/.cache";
      const std::string occa_cache_dir = cache_dir + "/occa/";
      occa::env::OCCA_CACHE_DIR = occa_cache_dir;
      setenv("OCCA_CACHE_DIR", occa_cache_dir.c_str(), 1);
    }
  }

  if(rank == 0) {
    std::cout << "number of states: " << n_states << '\n';
    std::cout << "number of repetitions: " << nRep << '\n';
  }

  const std::string target = (device.mode() == "Serial") ? "c++17" : device.mode();
  const int align_width = (device.mode() == "Serial") ? 64 : 0;
  if (single_precision)
    using dfloat = float;

  occa::properties kernel_properties;
  kinetix::init(mech,
                device,
                kernel_properties,
	        tool,
                blockSize,
	        single_precision,
	        unroll_loops,
                loop_gibbsexp,
                group_rxnUnroll,
                group_vis,
                nonsymDij,
                fit_rcpDiffCoeffs,
	        align_width,
	        target,
                false, /* useFP64Transport */
                MPI_COMM_WORLD,
                debug);

  n_species = kinetix::nSpecies();
  n_reactions = kinetix::nReactions();
  n_active_species = kinetix::nActiveSpecies();
  n_inert_species = n_species - n_active_species;

  if(debug) {
    std::cout << "species: \n";
    for (int k = 0; k < n_species; k++) {
      std::cout << kinetix::speciesNames()[k];
      if (k < n_species - 1) std::cout << ' ';
    }
    std::cout << '\n';
  }

  // setup state vector
  dfloat pressure;
  std::vector<dfloat> states((n_species+1) * n_states);
  occa::memory o_states = device.malloc<dfloat>((n_species+1) * n_states);

  for (int id = 0; id < n_states; id++) {
    double pressure_Pa = 1e5;
    double temperature_K = 1000;
    std::vector<double> mole_fractions(n_species, 1.0/n_species);
    std::vector<double> molar_masses(n_species, 1.0);

    if(ci)
      loadCiStateFromFile(ciStates[id], fs::path(mech).stem(), pressure_Pa, temperature_K, mole_fractions, molar_masses);

    auto molar_mass = 0.0;
    for (int k = 0; k < n_species; k++) 
      molar_mass += mole_fractions[k] * molar_masses[k]; 

    std::vector<double> mass_fractions(n_species, 1.0/n_species);
    for(int k = 0; k < n_species; k++)
      mass_fractions[k] = 
        mole_fractions[k] * molar_masses[k] / molar_mass;

    if(id == 0) { // Reference is first state
      ref_mole_fractions = new double[n_species];
      ref_mass_fractions = new double[n_species];

      for (int i = 0; i < n_species; i++) {
        ref_mole_fractions[i] = mole_fractions[i];
        ref_mass_fractions[i] = mass_fractions[i];
      }

      ref_pressure = pressure_Pa;
      if(ci == 2) ref_pressure = 101325; // use an arbitrary but different reference pressure
      ref_temperature = temperature_K;
      ref_molar_mass = molar_mass;
    }

    pressure = pressure_Pa / ref_pressure; 
    states[id + 0*n_states] = temperature_K / ref_temperature; 
    for (int i = 0; i < n_species; i++) {
      states[id + (i+1)*n_states] = mass_fractions[i]; 
    }

    if(debug) {
      printf("Yi: "); 
      for (int i = 0; i < n_species; i++) printf("%g ", states[id + (i+1)*n_states]);
      printf("\n");
      printf("T: %g\n", states[id + 0*n_states]);
      printf("p: %g\n", pressure);
    }

  }
 
  o_states.copyFrom(states.data());

  kinetix::build(ref_pressure,
                 ref_temperature,
                 ref_mass_fractions,
                 mode == 0 || mode == 2   /* enable transport */
                 );

  //////////////////////////////////////////////////////////////////////////////

  bool pass = true;

  {
    auto o_rho = device.malloc<dfloat>(n_states);
    auto o_cp_i = device.malloc<dfloat>(n_species * n_states);
    auto o_rhoCp = device.malloc<dfloat>(n_states);

    kinetix::thermodynamicProps(n_states,
                                n_states /* offsetT */,
                                n_states /* offset */,
                                pressure,
                                o_states,
                                o_rho,
                                o_cp_i,
                                o_rhoCp);
 
    if(ci && rank == 0 && mode == 0)
      pass &= checkThermo(o_rho,
                          o_cp_i,
                          o_rhoCp);
  }

  //////////////////////////////////////////////////////////////////////////////

  if (mode == 0 || mode == 1) {
    auto o_rates = device.malloc<dfloat>((n_species + 1) * n_states);

    kinetix::productionRates(n_states,
                             n_states /* offsetT */,
                             n_states /* offset */,
                             pressure,
                             o_states,
                             o_rates);

    device.finish();
    MPI_Barrier(MPI_COMM_WORLD);

    const auto startTime = MPI_Wtime();
    for(int i = 0; i < nRep; i++) {
      kinetix::productionRates(n_states,
                               n_states /* offsetT */,
                               n_states /* offset */,
                               pressure,
                               o_states,
                               o_rates);
    }
    device.finish();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = MPI_Wtime() - startTime;

    if(ci && rank == 0)
      pass &= checkRates(o_rates,
                         single_precision);

    if(!ci && rank == 0) {
      printf("BK1 (reaction rates) results:\n");
      printf("avg elapsed time: %.5f s\n", elapsedTime);
      printf("avg aggregated throughput: %.2f GRXN/s\n",
              (size * (double)(n_states * n_reactions) * nRep) / elapsedTime / 1e9);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  if (mode == 0 || mode == 2) {
    auto o_conductivity = device.malloc<dfloat>(n_states);
    auto o_viscosity = device.malloc<dfloat>(n_states);
    auto o_rhoD = device.malloc<dfloat>(n_species * n_states);

    kinetix::mixtureAvgTransportProps(n_states,
                                      n_states /* offsetT */,
                                      n_states /* offset */,
                                      pressure,
                                      o_states,
                                      o_viscosity,
                                      o_conductivity,
                                      o_rhoD);
    
    device.finish();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto startTime = MPI_Wtime();
    for(int i = 0; i < nRep; i++) {
      kinetix::mixtureAvgTransportProps(n_states,
                                        n_states /* offsetT */,
                                        n_states /* offset */,
                                        pressure,
                                        o_states,
                                        o_viscosity,
                                        o_conductivity,
                                        o_rhoD);
    }
    device.finish();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsedTime = MPI_Wtime() - startTime;

    if(ci && rank == 0)
      pass &= checkTransport(o_conductivity,
                             o_viscosity,
                             o_rhoD);

    if(!ci && rank == 0) {
      printf("BK2 (transport) results:\n");
      printf("avg elapsed time: %.5f s\n", elapsedTime);
      printf("avg aggregated throughput: %.2f GDOF/s\n",
              (size * (double)(n_states * (n_species + 2)) * nRep) / elapsedTime / 1e9);
    }
  }

  MPI_Finalize();
  if(pass && ci) printf("all tests passed!\n");
  exit(pass ? EXIT_SUCCESS : EXIT_FAILURE);
}
