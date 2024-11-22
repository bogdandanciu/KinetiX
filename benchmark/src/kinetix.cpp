#include <algorithm>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <float.h>
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <occa.hpp>

#include "kinetix.hpp"

namespace fs = std::filesystem;

namespace {

occa::kernel production_rates_kernel, production_rates_fpmix_kernel, production_rates_fp32_kernel;
occa::kernel transport_kernel, transport_fpmix_kernel, transport_fp32_kernel;
occa::kernel thermoCoeffs_kernel, thermoCoeffs_fpmix_kernel, thermoCoeffs_fp32_kernel;

kinetix::kinetixBuildKernel_t buildKernel;

occa::device device;
std::string occaCacheDir0;

double ref_pressure;
double ref_temperature;
std::vector<double> ref_mass_fractions;
double ref_meanMolarMass;

const double R = 1.380649e-23 * 6.02214076e23;
int n_species = -1;
int n_reactions = -1;
int n_active_species = -1;
std::vector<double> m_molar;

MPI_Comm comm;
int initialized = 0;

std::vector<std::string> species_names;

std::string yamlPath;
std::string cacheDir;
occa::properties kernel_properties, kernel_properties_fpmix, kernel_properties_fp32;
std::string tool;
int group_size;
bool verbose;
bool single_precision;
bool unroll_loops;
bool loop_gibbsexp;
bool group_rxnUnroll;
bool group_vis;
bool nonsymDij;
bool fit_rcpDiffCoeffs;
int align_width;
std::string target;
bool useFP64Transport;

} // namespace

bool kinetix::isInitialized() { return initialized; }

static void fileSync(const char *file)
{
  std::string dir;
  {
    const int len = std::char_traits<char>::length(file);
    char *tmp = (char *)malloc((len + 1) * sizeof(char));
    strncpy(tmp, file, len);
    dir.assign(dirname(tmp));
  }

  int fd;
  fd = open(file, O_RDONLY);
  fsync(fd);
  close(fd);

  fd = open(dir.c_str(), O_RDONLY);
  fsync(fd);
  close(fd);
}

// std::to_string might be not accurate enough
static std::string to_string_f(double a)
{
  std::stringstream s;
  constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
  s << std::setprecision(maxPrecision) << std::scientific << a;
  return s.str();
}

namespace kinetix{
static std::string to_string_f(double a)
{
  std::stringstream s;
#if 0
  constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
  s << std::setprecision(maxPrecision) << std::scientific << a;
#else
  s << std::scientific << a;
#endif
  return s.str();
}

unsigned long hash(const std::string& str)
{
  unsigned int hash = 1315423911;

  for(std::size_t i = 0; i < str.length(); i++)
  {
      hash ^= ((hash << 5) + str[i] + (hash >> 2));
  }

  return (hash & 0x7FFFFFFF);
}

}

static bool mkDir(const fs::path &file_path)
{
  size_t pos = 0;
  bool ret_val = true;

  std::string dir_path(file_path);
  if (!fs::is_directory((file_path)))
    dir_path = file_path.parent_path();

  while (ret_val && pos != std::string::npos) {
    pos = dir_path.find('/', pos + 1);
    const auto dir = fs::path(dir_path.substr(0, pos));
    if (!fs::exists(dir)) {
      ret_val = fs::create_directory(dir);
    }
  }

  return ret_val;
}

static std::vector<std::string> split(std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

static occa::properties addOccaCompilerFlags(const occa::properties &a, const occa::properties &b)
{   
  occa::properties fused = b;
  for (const auto& key : a.keys()) {
    if (key == "compiler_flags") {
      if (fused.has("compiler_flags")) {
        fused["compiler_flags"] = fused["compiler_flags"].string() + " " + a[key].string();
      } else {
        fused["compiler_flags"] = a[key];
      }
    } else {
      if (!fused.has(key)) {
        fused[key] = a[key];
      }
    }
  }

  return fused;
}

static void setupKernelProperties()
{
  kernel_properties["compiler_flags"] += " -include " + cacheDir + "/mech.h";

  if (verbose && 0) {
    kernel_properties["verbose"] = "true";
    kernel_properties["compiler_flags"] += " -DDEBUG";
  }

  // workaround to bypass occa parser
  kernel_properties["okl/strict_headers"] = false;
  const auto incStatement = " -I" + cacheDir;
  kernel_properties["compiler_flags"] += incStatement;
  kernel_properties["defines/p_R"] += R;

  if (device.mode() == "CUDA") {
    setenv("OCCA_CXXFLAGS", incStatement.c_str(), 1); // required for launcher
    kernel_properties["compiler_flags"] += " -D__KINETIX_DEVICE__=__device__";
    kernel_properties["compiler_flags"] += " -D__KINETIX_CONST__=__constant__";
    kernel_properties["compiler_flags"] += " -D__KINETIX_INLINE__='__forceinline__ static'";
    kernel_properties["compiler_flags"] += " --expt-relaxed-constexpr"; //required for amrex
  }
  else if (device.mode() == "HIP") {
    setenv("OCCA_CXXFLAGS", incStatement.c_str(), 1); // required for launcher
    kernel_properties["compiler_flags"] += " -D__KINETIX_DEVICE__=__device__";
    kernel_properties["compiler_flags"] += " -D__KINETIX_CONST__=__constant__";
    kernel_properties["compiler_flags"] += " -D__KINETIX_INLINE__='__forceinline static'";
  }
  else if (device.mode() == "dpcpp") {
    setenv("OCCA_CXXFLAGS", incStatement.c_str(), 1); // required for launcher
    kernel_properties["compiler_flags"] += " -D__KINETIX_DEVICE__=SYCL_EXTERNAL";
    kernel_properties["compiler_flags"] += " -D__KINETIX_CONST__=const";
    kernel_properties["compiler_flags"] += " -D__KINETIX_INLINE__='__forceinline'";
  }
  else {
    std::string OCCA_CXXFLAGS;
    if (getenv("OCCA_CXXFLAGS"))
      OCCA_CXXFLAGS.assign(getenv("OCCA_CXXFLAGS"));
    kernel_properties["compiler_flags"] += OCCA_CXXFLAGS;
    kernel_properties["compiler_flags"] += " -D__KINETIX_DEVICE__=";
    kernel_properties["compiler_flags"] += " -D__KINETIX_CONST__=const";
    kernel_properties["compiler_flags"] += " -D__KINETIX_INLINE__='static inline'";
    kernel_properties["compiler_flags"] += " -include cmath";
    kernel_properties["compiler_flags"] += " -include cstdio";
    group_size = 1;
  }

  kernel_properties["defines/p_BLOCKSIZE"] = std::to_string(group_size);
  if (tool == "Pele"){
    kernel_properties["defines/p_PELETOOL"] = std::to_string(1);
  }else{
    kernel_properties["defines/p_PELETOOL"] = std::to_string(0);
  }

  kernel_properties_fp32 = kernel_properties;

  {
    const auto dfloatType = std::string("double");
    const auto cfloatType = std::string("double");
    kernel_properties["defines/dfloat"] = dfloatType;
    kernel_properties["defines/cfloat"] = cfloatType;
    kernel_properties["compiler_flags"] += " -Ddfloat=" + dfloatType;
    kernel_properties["compiler_flags"] += " -Dcfloat=" + cfloatType;
    kernel_properties["compiler_flags"] += " -DCFLOAT_MAX=1e300";
    kernel_properties["compiler_flags"] += " -DCFLOAT_MIN=1e-300";
    kernel_properties["compiler_flags"] += " -D__KINETIX_EXP__=exp";
    kernel_properties["compiler_flags"] += " -D__KINETIX_LOG10__=log10";
    kernel_properties["compiler_flags"] += " -D__KINETIX_LOG__=log";
    kernel_properties["compiler_flags"] += " -D__KINETIX_POW__=pow";
    kernel_properties["compiler_flags"] += " -D__KINETIX_MIN_CFLOAT=fmin";
    kernel_properties["compiler_flags"] += " -D__KINETIX_MAX=fmax";
  }

  {
    const auto dfloatType = std::string("float");
    const auto cfloatType = std::string("float");
    kernel_properties_fp32["compiler_flags"] += " -DCFLOAT_MAX=1e37f";
    kernel_properties_fp32["compiler_flags"] += " -DCFLOAT_MIN=1e-37f";
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_EXP__=expf";
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_LOG10__=log10f";
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_LOG__=logf";
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_POW__=powf";
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_MIN_CFLOAT=fminf";

    {
      const auto dfloatType = std::string("double");
      const auto cfloatType = std::string("float");
      kernel_properties_fpmix = kernel_properties_fp32;
      kernel_properties_fpmix["defines/dfloat"] = dfloatType;
      kernel_properties_fpmix["defines/cfloat"] = cfloatType;
      kernel_properties_fpmix["compiler_flags"] += " -Ddfloat=" + dfloatType;
      kernel_properties_fpmix["compiler_flags"] += " -Dcfloat=" + cfloatType;
      kernel_properties_fpmix["compiler_flags"] += " -D__KINETIX_MAX=fmax";
    }

    kernel_properties_fp32["defines/dfloat"] = dfloatType;
    kernel_properties_fp32["defines/cfloat"] = cfloatType;
    kernel_properties_fp32["compiler_flags"] += " -Ddfloat=" + dfloatType;
    kernel_properties_fp32["compiler_flags"] += " -Dcfloat=" + cfloatType;
    kernel_properties_fp32["compiler_flags"] += " -D__KINETIX_MAX=fmaxf";
  }

}

static occa::kernel _buildKernel(const std::string &path, const std::string &fileName, occa::properties prop)
{
  occa::kernel kernel;
  const auto okl_path = std::string(getenv("KINETIX_PATH")) + "/okl/";

  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  for (int r = 0; r < 2; r++) {
    if ((r == 0 && rank == 0) || (r == 1 && rank > 0))
      kernel = device.buildKernel(okl_path + path, fileName, prop);
    MPI_Barrier(comm);
  }

  return kernel;
}

static void setup()
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  occaCacheDir0 = getenv("OCCA_CACHE_DIR");
  
  if (!getenv("KINETIX_PATH")) {
    std::string path = std::string(getenv("HOME")) + "/.local/kinetix";
    setenv("KINETIX_PATH", path.c_str(), 0);
  }
  if (!getenv("OCCA_DIR")) {
    occa::env::OCCA_DIR = std::string(getenv("KINETIX_PATH")) + "/";
  }

  const auto installDir = std::string(getenv("KINETIX_PATH"));

  {
    const std::string yamlName = fs::path(yamlPath).stem();
    cacheDir = ".cache/KINETIX/" + yamlName;
    cacheDir = std::string(fs::absolute(cacheDir));

    if (rank == 0) {
        mkDir(cacheDir + "/mech.h");
        fileSync(cacheDir.c_str());
    }

    const std::string occaCacheDir = cacheDir + "/.occa/";
    occa::env::OCCA_CACHE_DIR = occaCacheDir;
    setenv("OCCA_CACHE_DIR", occaCacheDir.c_str(), 1);
    const auto bckCache =  cacheDir + "/.occa.bck"; 
    if (fs::exists(bckCache)) 
      fs::remove_all(bckCache);
  }

  if (rank == 0) {
    std::string cmdline = installDir + "/kinetix/__main__.py" + " --header-only" + " --mechanism " +
                          yamlPath + " --output " + cacheDir;
    if (verbose)
      std::cout << cmdline << std::endl;
    if (system(cmdline.c_str())) {
      std::cout << "Error while running code generator!\n";
      MPI_Abort(comm, 1);
    }
    fileSync(std::string(cacheDir + "/mech.h").c_str());
  }

  MPI_Barrier(comm);

  setupKernelProperties();

  {
    const auto oklpath = std::string(getenv("KINETIX_PATH")) + "/okl/";
    occa::kernel nSpeciesKernel = buildKernel("mech.okl", "nSpecies", kernel_properties);
    occa::kernel nReactionsKernel = buildKernel("mech.okl", "nReactions", kernel_properties);
    occa::kernel mMolarKernel = buildKernel("mech.okl", "mMolar", kernel_properties);
    occa::kernel speciesNamesLengthKernel = buildKernel("mech.okl", "speciesNamesLength", kernel_properties);
    occa::kernel speciesStringKernel = buildKernel("mech.okl", "speciesString", kernel_properties);
 
    {
      auto tmp = (int *)calloc(2, sizeof(int));
      auto o_nSpecies = device.malloc(2 * sizeof(int));
      nSpeciesKernel(o_nSpecies);
      o_nSpecies.copyTo(tmp);
      n_species = tmp[0];
      n_active_species = tmp[1];
      free(tmp);
    }

    {
      auto tmp = (int *)calloc(1, sizeof(int));
      auto o_nReactions = device.malloc(1 * sizeof(int));
      nReactionsKernel(o_nReactions);
      o_nReactions.copyTo(tmp);
      n_reactions = tmp[0];
      free(tmp);
    }
 
    {
      auto tmp = (double *)calloc(n_species, sizeof(double));
      auto o_tmp = device.malloc(n_species * sizeof(double));
      mMolarKernel(o_tmp);
      o_tmp.copyTo(tmp);
      for (int k = 0; k < n_species; k++) {
        m_molar.push_back(tmp[k]);
      }
      free(tmp);
    }
 
    {
      auto speciesNamesLength = 0;
      auto o_speciesNamesLength = device.malloc(sizeof(int));
      speciesNamesLengthKernel(o_speciesNamesLength);
      o_speciesNamesLength.copyTo(&speciesNamesLength);
 
      auto tmp = (char *)calloc(speciesNamesLength, sizeof(char));
      auto o_tmp = device.malloc(speciesNamesLength * sizeof(char));
      speciesStringKernel(o_tmp);
      o_tmp.copyTo(tmp);
      species_names = split(tmp, " ");
      free(tmp);
    }
  }

  occa::env::OCCA_CACHE_DIR = occaCacheDir0;
  setenv("OCCA_CACHE_DIR", occaCacheDir0.c_str(), 1);
}

static void buildMechKernels(bool transport)
{
  {// Thermo coeffs kernel
    occa::properties includeProp;
    if (single_precision) {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/fheat_capacity_R.cpp";
      thermoCoeffs_fp32_kernel = buildKernel("thermoCoeffs.okl", 
          	                             "thermoCoeffs",
          			             addOccaCompilerFlags(includeProp, kernel_properties_fp32));
      thermoCoeffs_fpmix_kernel = buildKernel("thermoCoeffs.okl", 
          	                              "thermoCoeffs",
          			              addOccaCompilerFlags(includeProp, kernel_properties_fpmix));
    } else {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/heat_capacity_R.cpp";
      thermoCoeffs_kernel = buildKernel("thermoCoeffs.okl", 
          	                        "thermoCoeffs",
          			        addOccaCompilerFlags(includeProp, kernel_properties));
    }
  }

  {// Rates kernel 
    occa::properties includeProp;
    includeProp["kernel/include_occa"] = true; //required for amrex
    includeProp["kernel/link_occa"] = true;
    includeProp["compiler_flags"] += " -include " + cacheDir + "/mechanism.H";
    if (single_precision) {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/fenthalpy_RT.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/frates.cpp";
      production_rates_fp32_kernel = buildKernel("productionRates.okl", 
          	    			         "productionRates",
          				         addOccaCompilerFlags(includeProp, kernel_properties_fp32));
      production_rates_fpmix_kernel = buildKernel("productionRates.okl", 
          	    			          "productionRates",
          				          addOccaCompilerFlags(includeProp, kernel_properties_fpmix));
    } else {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/enthalpy_RT.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/rates.cpp";
      production_rates_kernel = buildKernel("productionRates.okl",
                                            "productionRates",
                                            addOccaCompilerFlags(includeProp, kernel_properties));
    }
  }

  // Transport props kernel
  if (transport) {

    occa::properties includeProp;
    if (single_precision) {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/fconductivity.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/fviscosity.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/fdiffusivity.cpp";
      transport_fp32_kernel = buildKernel("transportProps.okl", 
		                          "transport",
		                          addOccaCompilerFlags(includeProp, kernel_properties_fp32));
      transport_fpmix_kernel = buildKernel("transportProps.okl", 
		                           "transport",
					   addOccaCompilerFlags(includeProp, kernel_properties_fpmix));
    } else {
      includeProp["compiler_flags"] += " -include " + cacheDir + "/conductivity.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/viscosity.cpp";
      includeProp["compiler_flags"] += " -include " + cacheDir + "/diffusivity.cpp";
      transport_kernel = buildKernel("transportProps.okl", 
		                     "transport",
				     addOccaCompilerFlags(includeProp, kernel_properties));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  API
///////////////////////////////////////////////////////////////////////////////

void kinetix::init(const std::string &model_path,
                   occa::device _device,
                   occa::properties _props,
                   const std::string &_tool,
                   int _group_size,
                   bool _single_precision,
                   bool _unroll_loops,
                   bool _loop_gibbsexp,
                   bool _group_rxnUnroll,
		   bool _group_vis,
		   bool _nonsymDij,
                   bool _fit_rcpDiffCoeffs,
                   int _align_width,
                   const std::string &_target,
                   bool _useFP64Transport,
                   MPI_Comm _comm,
                   bool _verbose)
{
  kinetix::kinetixBuildKernel_t build = [&](const std::string &path, const std::string &fileName, occa::properties prop)
  {
    return _buildKernel(path, fileName, prop);
  };
  kinetix::init(model_path,
                _device,
                _props,
	        _tool,
                _group_size,
	        _single_precision,
                _unroll_loops,
	        _loop_gibbsexp,
	        _group_rxnUnroll,
	        _group_vis,
                _nonsymDij,
                _fit_rcpDiffCoeffs,
                _align_width,
                _target,
                _useFP64Transport,
                _comm,
                build,
                _verbose);
}

void kinetix::init(const std::string &model_path,
                   occa::device _device,
                   occa::properties _props,
                   const std::string &_tool,
                   int _group_size,
                   bool _single_precision,
                   bool _unroll_loops,
                   bool _loop_gibbsexp,
                   bool _group_rxnUnroll,
		   bool _group_vis,
		   bool _nonsymDij,
                   bool _fit_rcpDiffCoeffs,
                   int _align_width,
                   const std::string &_target,
                   bool _useFP64Transport,
                   MPI_Comm _comm,
                   kinetixBuildKernel_t build,
                   bool _verbose)
{
  buildKernel = build;

  verbose = _verbose;
  comm = _comm;
  device = _device;
  tool = _tool;

  if (!_props.isInitialized()) {
    if (device.mode() == "CUDA") {
      kernel_properties["compiler_flags"] += " -O3 ";
      kernel_properties["compiler_flags"] += " --use_fast_math";
    }
    else if (device.mode() == "HIP") {
      kernel_properties["compiler_flags"] += " -O3 ";
      kernel_properties["compiler_flags"] += " -ffp-contract=fast ";
      kernel_properties["compiler_flags"] += " -funsafe-math-optimizations";
      kernel_properties["compiler_flags"] += " -ffast-math";
    }
    else if (device.mode() == "dpcpp") {
      kernel_properties["compiler_flags"] += " -O3 ";
      kernel_properties["compiler_flags"] += " -fsycl ";
      kernel_properties["compiler_flags"] += " -ffp-contract=fast ";
      kernel_properties["compiler_flags"] += " -funsafe-math-optimizations";
      kernel_properties["compiler_flags"] += " -ffast-math";
    }
    else {
      std::string OCCA_CXXFLAGS;
      if (getenv("OCCA_CXXFLAGS"))
        OCCA_CXXFLAGS.assign(getenv("OCCA_CXXFLAGS"));
      kernel_properties["compiler_flags"] += OCCA_CXXFLAGS;
      kernel_properties["compiler_flags"] += " -O3 ";
      kernel_properties["compiler_flags"] += " -march=native";
      kernel_properties["compiler_flags"] += " -mtune=native";
      kernel_properties["compiler_flags"] += " -ffast-math";
      kernel_properties["compiler_flags"] += " -include cmath";
      kernel_properties["compiler_flags"] += " -include cstdio";
    }
  }
  else {
    kernel_properties = _props;
  }

  yamlPath = fs::path(model_path);
  group_size = std::max(_group_size, 32);
  single_precision = _single_precision;
  unroll_loops = _unroll_loops;
  loop_gibbsexp = _loop_gibbsexp;
  group_rxnUnroll = _group_rxnUnroll;
  group_vis = _group_vis;
  nonsymDij = _nonsymDij;
  fit_rcpDiffCoeffs = _fit_rcpDiffCoeffs;
  align_width = _align_width;
  target = _target;
  useFP64Transport = _useFP64Transport;

  setup();
  MPI_Barrier(comm);
}

double kinetix::refPressure() { return ref_pressure; }

double kinetix::refTemperature() { return ref_temperature; }

const std::vector<double> kinetix::refMassFractions() { return ref_mass_fractions; }

double kinetix::refMeanMolecularWeight() { return ref_meanMolarMass; }

void kinetix::build(double _ref_pressure,
                    double _ref_temperature,
                    double _ref_mass_fractions[],
                    bool transport)
{
  initialized = 1;
  const std::string occaCacheDir = cacheDir + "/.occa/";
  occa::env::OCCA_CACHE_DIR = occaCacheDir;
  setenv("OCCA_CACHE_DIR", occaCacheDir.c_str(), 1);

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  ref_pressure = _ref_pressure;
  ref_temperature = _ref_temperature;

  double sum = 0.;
  for (int k = 0; k < nSpecies(); k++) {
    sum += _ref_mass_fractions[k] / m_molar[k];
    ref_mass_fractions.push_back(_ref_mass_fractions[k]);
  }
  ref_meanMolarMass = 1. / sum;

  auto ref_mole_fractions = new double[nSpecies()];
  std::string ref_mole_fractions_string;
  for (int k = 0; k < nSpecies(); k++) {
    ref_mole_fractions[k] = 1. / m_molar[k] * ref_meanMolarMass * _ref_mass_fractions[k];
    ref_mole_fractions_string += kinetix::to_string_f(ref_mole_fractions[k]);
    if (k < nSpecies() - 1)
      ref_mole_fractions_string += ',';
  }

  const auto installDir = std::string(getenv("KINETIX_PATH") ?: ".");
  if (rank == 0) {
    std::string cmdline_pele;
    // Copy Pele mechanisms files to cache
    const std::string yamlName = fs::path(yamlPath).stem();
    cmdline_pele = "cp " + installDir + "/kinetix/mechanisms/Pele/" + yamlName + "/mechanism.*" + " " + cacheDir;
    if (verbose)
      std::cout << cmdline_pele << std::endl;
    if (system(cmdline_pele.c_str())) {
      std::cout << "Error while copying pele mechanisms files to cache!\n";
      MPI_Abort(comm, 1);
    }

    // Run generator
    std::string cmdline;
    cmdline = installDir +
              "/kinetix/__main__.py" +
              " --mechanism " + yamlPath + 
              " --output " + cacheDir +
              " --align-width " + std::to_string(align_width) +
              " --target " + target;
    if (single_precision)
      cmdline.append(" --single-precision");
    if (unroll_loops)
      cmdline.append(" --unroll-loops");
    if (loop_gibbsexp)
      cmdline.append(" --loop-gibbsexp");
    if (group_rxnUnroll)
      cmdline.append(" --group-rxnunroll");
    if (group_vis)
       cmdline.append(" --group-vis");
    if (nonsymDij)
      cmdline.append(" --nonsymDij");
    if (fit_rcpDiffCoeffs)
      cmdline.append(" --fit-rcpdiffcoeffs");

    const auto currentHash = hash(cmdline);
    auto runGenerator = [&]
    {
      unsigned long oldHash = 0;
      std::ifstream f(cacheDir + "/.hash");
      if(f.is_open()) {
        f >> oldHash;
        f.close();
      }
      return oldHash != currentHash; 
    }();

    if (runGenerator) {
      if (verbose)
        std::cout << cmdline << std::endl;
      if (system(cmdline.c_str())) {
        std::cout << "Error while running code generator!\n";
        MPI_Abort(comm, 1);
      }

      std::ofstream f(cacheDir + "/.hash");
      f << currentHash;
      f.close();
 
      fileSync(std::string(cacheDir + "/mechanism.cpp").c_str());
      fileSync(std::string(cacheDir + "/mechanism.H").c_str());
      if (single_precision) {
        fileSync(std::string(cacheDir + "/fconductivity.cpp").c_str());
        fileSync(std::string(cacheDir + "/fdiffusivity.cpp").c_str());
        fileSync(std::string(cacheDir + "/fenthalpy_RT.cpp").c_str());
        fileSync(std::string(cacheDir + "/fheat_capacity_R.cpp").c_str());
        fileSync(std::string(cacheDir + "/fviscosity.cpp").c_str());
        fileSync(std::string(cacheDir + "/frates.cpp").c_str());
        fileSync(std::string(cacheDir + "/frates.cpp").c_str());
      } else {
        fileSync(std::string(cacheDir + "/conductivity.cpp").c_str());
        fileSync(std::string(cacheDir + "/diffusivity.cpp").c_str());
        fileSync(std::string(cacheDir + "/enthalpy_RT.cpp").c_str());
        fileSync(std::string(cacheDir + "/heat_capacity_R.cpp").c_str());
        fileSync(std::string(cacheDir + "/viscosity.cpp").c_str());
        fileSync(std::string(cacheDir + "/rates.cpp").c_str());
      }
      // inc files are passed to compiler so occa doesn't know if they have changed
      // force occa to recompile by renaming old cache dir
      // removing might not work as some files are still open
   
      const auto bckCache =  cacheDir + "/.occa.bck"; 
      if (fs::exists(bckCache)) 
        fs::rename(occaCacheDir, bckCache);
    }
    fflush(stdout);
  }
  MPI_Barrier(comm);

  buildMechKernels(transport);

  delete[] ref_mole_fractions;

  if (rank == 0) {
    if (tool == "Pele")
      std::cout   << "\n================= KinetiX with Pele mechanism =================\n";
    else 
      std::cout   << "\n================= KinetiX =================\n";
    std::cout << "active occa mode: " << device.mode() << "\n";
    if (device.mode() != "Serial")
      std::cout << "blockSize: " << group_size << "\n";
    std::cout   << "cache: " << cacheDir << "\n"
                << "yaml-file: " << yamlPath << "\n"
                << "nSpecies: " << n_species << "\n"
                << "TRef: " << ref_temperature << " K\n"
                << "pRef: " << ref_pressure << " Pa\n"
                << "YRef: ";
    bool first = true;
    for (auto &&species : kinetix::speciesNames()) {
      auto fraction = ref_mass_fractions[speciesIndex(species)];
      if (fraction > 0) {
        if (!first)
          std::cout << ", ";
        std::cout << species << "=" << fraction;
        first = false;
      }
    }
    std::cout << "\n";

    auto printVec = [&](const std::string& prefix, const std::vector<double>& vec, const std::string& unit = "") {
      std::cout << prefix;
      bool first = true;
      for (auto &&species : kinetix::speciesNames()) {
        auto entry = vec[speciesIndex(species)];
        if (entry > 0) {
          if (!first)
            std::cout << ", ";
          std::cout << species << "=" << entry;
          first = false;
        }
      }
      std::cout << " " << unit << std::endl;
      return;
    };

    fflush(stdout);
  }

  occa::env::OCCA_CACHE_DIR = occaCacheDir0;
  setenv("OCCA_CACHE_DIR", occaCacheDir0.c_str(), 1);

  MPI_Barrier(comm);
}

void kinetix::productionRates(int n_states,
                              int offsetT,
                              int offset,
                              double pressure,
                              const occa::memory &o_state,
                              occa::memory &o_rates)
{
  assert(initialized);

  occa::kernel kernel = production_rates_kernel;
  if (o_state.dtype() != occa::dtype::float_ && single_precision)
    kernel = production_rates_fpmix_kernel;
  else if (single_precision)
    kernel = production_rates_fp32_kernel;


  const double pressure_R = pressure * ref_pressure / R;
  const double pressure_ = pressure * ref_pressure;

  kernel(n_states,
         offsetT,
         offset,
         pressure_R,
         pressure_,
         o_state,
         o_rates,
         ref_temperature);
}

void kinetix::mixtureAvgTransportProps(int n_states,
                                       int offsetT,
                                       int offset,
                                       double pressure,
                                       const occa::memory &o_state,
                                       occa::memory &o_viscosity,
                                       occa::memory &o_conductivity,
                                       occa::memory &o_density_diffusivity)
{
  assert(initialized);

  occa::kernel kernel = transport_kernel;
  if (o_state.dtype() != occa::dtype::float_ && single_precision)
    kernel = transport_fpmix_kernel;
  else if (single_precision)
    kernel = transport_fp32_kernel;

  kernel(n_states, 
         offsetT, 
         offset, 
         pressure, 
         o_state, 
         o_conductivity, 
         o_viscosity,
         o_density_diffusivity,
         ref_temperature);
}

void kinetix::thermodynamicProps(int n_states,
                                 int offsetT,
                                 int offset,
                                 double pressure,
                                 const occa::memory &o_state,
                                 occa::memory &o_rho,
                                 occa::memory &o_cp,
                                 occa::memory &o_rhocp)
{
  assert(initialized);

  occa::kernel kernel = thermoCoeffs_kernel;
  if (o_state.dtype() != occa::dtype::float_ && single_precision)
    kernel = thermoCoeffs_fpmix_kernel;
  else if (single_precision)
    kernel = thermoCoeffs_fp32_kernel;

  const double pressure_R = pressure * ref_pressure / R;

  kernel(n_states,
         offsetT,
         offset,
         pressure_R,
         o_state,
         o_rho,
         o_cp,
         o_rhocp,
         ref_temperature);
}

int kinetix::nSpecies()
{ 
  return n_species; 
}

int kinetix::nReactions()
{ 
  return n_reactions; 
}

int kinetix::nActiveSpecies()
{ 
  return n_active_species; 
}

const std::vector<double> kinetix::molecularWeights()
{
  std::vector<double> tmp;
  for (int k = 0; k < n_species; k++) {
    tmp.push_back(m_molar[k] / ref_meanMolarMass);
  }
  return tmp;
}

const std::vector<std::string> kinetix::speciesNames()
{ 
  return ::species_names; 
}

int kinetix::speciesIndex(const std::string &name)
{
  auto it = find(::species_names.begin(), ::species_names.end(), name);
  if (it != ::species_names.end())
    return it - ::species_names.begin();
  return -1;
}
