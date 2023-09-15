#include <vector>
#include <string>
#include <float.h>
#include <cassert>
#include <unistd.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <fcntl.h>
#include <libgen.h>

#include <mpi.h>
#include <occa.hpp>

#include "_nekCRF.hpp"

namespace fs = std::filesystem;

namespace {

occa::kernel production_rates_kernel, production_rates_fpmix_kernel, production_rates_fp32_kernel;
occa::kernel transport_fpmix_kernel, transport_fp32_kernel;
occa::kernel thermoCoeffs_fpmix_kernel, thermoCoeffs_fp32_kernel;

_nekCRF::nekCRFBuildKernel_t buildKernel;

occa::device device;
std::string occaCacheDir0;

double ref_time;
double ref_length;
double ref_velocity;
double ref_pressure;
double ref_temperature;
std::vector<double> ref_mass_fractions;
double ref_viscosity;
double ref_conductivity;
double ref_density;
double ref_cp;
double ref_cv;
std::vector<double> ref_rhoDiffCoeffs;
double ref_meanMolarMass;
double ref_mass_rate;
double ref_volumetric_energy_rate;

const double R = 1.380649e-23 * 6.02214076e23;
int n_species = -1;
int n_active_species = -1;
std::vector<double> m_molar;

MPI_Comm comm;
int initialized = 0;

std::vector<std::string> species_names;

std::string yamlPath;
std::string cacheDir;
occa::properties kernel_properties, kernel_properties_fp32, kernel_properties_mixed;
int group_size;
bool verbose;
bool unroll_loops;
bool unroll_loops_transport;
bool pragma_unroll_loops;
bool loop_gibbsexp;
bool nonsymDij;
bool fit_rcpDiffCoeffs;
int align_width;
std::string target;
bool useFP64Transport;

} // namespace

bool _nekCRF::isInitialized() { return initialized; }

static bool isStandalone(){
#ifdef STANDALONE
  return true;
#else
  return false;
#endif
}

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

#ifdef STANDALONE
// std::to_string might be not accurate enough
static std::string to_string_f(double a)
{
  std::stringstream s;
  constexpr auto maxPrecision{std::numeric_limits<double>::digits10 + 1};
  s << std::setprecision(maxPrecision) << std::scientific << a;
  return s.str();
}
#endif

namespace _nekCRF{
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

static occa::properties addOccaCompilerFlags(occa::properties &a, occa::properties &b)
{
  occa::properties fused = b;
  std::string flags = a.get<std::string>("compiler_flags");
  fused["compiler_flags"] += flags;
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
    kernel_properties["compiler_flags"] += " -D__NEKRK_DEVICE__=__device__";
    kernel_properties["compiler_flags"] += " -D__NEKRK_CONST__=__constant__";
    kernel_properties["compiler_flags"] += " -D__NEKRK_INLINE__='__forceinline__ static'";
  }
  else if (device.mode() == "HIP") {
    setenv("OCCA_CXXFLAGS", incStatement.c_str(), 1); // required for launcher
    kernel_properties["compiler_flags"] += " -D__NEKRK_DEVICE__=__device__";
    kernel_properties["compiler_flags"] += " -D__NEKRK_CONST__=__constant__";
    kernel_properties["compiler_flags"] += " -D__NEKRK_INLINE__='__forceinline static'";
  }
  else if (device.mode() == "dpcpp") {
    setenv("OCCA_CXXFLAGS", incStatement.c_str(), 1); // required for launcher
    kernel_properties["compiler_flags"] += " -D__NEKRK_DEVICE__=SYCL_EXTERNAL";
    kernel_properties["compiler_flags"] += " -D__NEKRK_CONST__=const";
    kernel_properties["compiler_flags"] += " -D__NEKRK_INLINE__='__forceinline'";
  }
  else {
    std::string OCCA_CXXFLAGS;
    if (getenv("OCCA_CXXFLAGS"))
      OCCA_CXXFLAGS.assign(getenv("OCCA_CXXFLAGS"));
    kernel_properties["compiler_flags"] += OCCA_CXXFLAGS;
    kernel_properties["compiler_flags"] += " -D__NEKRK_DEVICE__=";
    kernel_properties["compiler_flags"] += " -D__NEKRK_CONST__=const";
    kernel_properties["compiler_flags"] += " -D__NEKRK_INLINE__='static inline'";
    kernel_properties["compiler_flags"] += " -include cmath";
    kernel_properties["compiler_flags"] += " -include cstdio";
    group_size = 1;
  }

  kernel_properties["defines/p_BLOCKSIZE"] = std::to_string(group_size);

  kernel_properties_fp32 = kernel_properties;

  {
    const auto dfloatType = std::string("double");
    const auto cfloatType = std::string("double");
    kernel_properties["defines/dfloat"] = dfloatType;
    kernel_properties["defines/cfloat"] = cfloatType;
    kernel_properties["compiler_flags"] += " -Ddfloat=" + dfloatType;
    kernel_properties["compiler_flags"] += " -Dcfloat=" + cfloatType;
    kernel_properties["compiler_flags"] += " -D__NEKRK_EXP__=exp";
    kernel_properties["compiler_flags"] += " -D__NEKRK_LOG10__=log10";
    kernel_properties["compiler_flags"] += " -D__NEKRK_LOG__=log";
    kernel_properties["compiler_flags"] += " -D__NEKRK_POW__=pow";
    kernel_properties["compiler_flags"] += " -DCFLOAT_MAX=1e300";
    kernel_properties["compiler_flags"] += " -DCFLOAT_MIN=1e-300";
    kernel_properties["compiler_flags"] += " -D__NEKRK_MIN_CFLOAT=fmin";
    kernel_properties["compiler_flags"] += " -D__NEKRK_MAX=fmax";
  }

  {
    const auto dfloatType = std::string("float");
    const auto cfloatType = std::string("float");
    kernel_properties_fp32["compiler_flags"] += " -DCFLOAT_MAX=1e37f";
    kernel_properties_fp32["compiler_flags"] += " -DCFLOAT_MIN=1e-37f";
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_MIN_CFLOAT=fminf";
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_EXP__=expf";
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_LOG10__=log10f";
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_LOG__=logf";
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_POW__=powf";

    {
      const auto dfloatType = std::string("double");
      const auto cfloatType = std::string("float");
      kernel_properties_mixed = kernel_properties_fp32;
      kernel_properties_mixed["defines/dfloat"] = dfloatType;
      kernel_properties_mixed["defines/cfloat"] = cfloatType;
      kernel_properties_mixed["compiler_flags"] += " -Ddfloat=" + dfloatType;
      kernel_properties_mixed["compiler_flags"] += " -Dcfloat=" + cfloatType;
      kernel_properties_mixed["compiler_flags"] += " -D__NEKRK_MAX=fmax";
    }

    kernel_properties_fp32["defines/dfloat"] = dfloatType;
    kernel_properties_fp32["defines/cfloat"] = cfloatType;
    kernel_properties_fp32["compiler_flags"] += " -Ddfloat=" + dfloatType;
    kernel_properties_fp32["compiler_flags"] += " -Dcfloat=" + cfloatType;
    kernel_properties_fp32["compiler_flags"] += " -D__NEKRK_MAX=fmaxf";
  }
}

static occa::kernel _buildKernel(const std::string &path, const std::string &fileName, occa::properties prop)
{
  occa::kernel kernel;
  const auto okl_path = std::string(getenv("NEKCRF_PATH")) + "/okl/";

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

  if(isStandalone()){
    if (!getenv("NEKCRF_PATH")) {
      std::string path = std::string(getenv("HOME")) + "/.local/nekCRF";
      setenv("NEKCRF_PATH", path.c_str(), 0);
    }
    if (!getenv("OCCA_DIR")) {
      occa::env::OCCA_DIR = std::string(getenv("NEKCRF_PATH")) + "/";
    }
  } else {
    std::string path = std::string(getenv("NEKRS_HOME")) + "/3rd_party/nekCRF";
    setenv("NEKCRF_PATH", path.c_str(), 1);
    if(!getenv("NEKRS_MPI_UNDERLYING_COMPILER")) { // check if call is from nekRS
      occa::env::OCCA_DIR = std::string(getenv("NEKRS_HOME")) + "/";
    }
  }

  const auto installDir = std::string(getenv("NEKCRF_PATH"));

  {
    const std::string yamlName = fs::path(yamlPath).stem();
    cacheDir = getenv("NEKRS_CACHE_DIR") ? std::string(getenv("NEKRS_CACHE_DIR")) + "/nekCRF/" + yamlName
                                         : ".cache/nekCRF/" + yamlName;
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
    std::string cmdline = installDir + "/generator/generate.py" + " --header-only" + " --mechanism " +
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
    const auto oklpath = std::string(getenv("NEKCRF_PATH")) + "/okl/";
    occa::kernel nSpeciesKernel = buildKernel("mech.okl", "nSpecies", kernel_properties);
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
  {
    occa::properties props = kernel_properties;
    props["compiler_flags"] += " -include " + cacheDir + "/ref.h";
    occa::kernel kernel = buildKernel("ref.okl", "refRhoDiffCoeffs", props);
    auto tmp = (double *)calloc(n_species, sizeof(double));
    auto o_tmp = device.malloc(n_species * sizeof(double));
    kernel(o_tmp);
    o_tmp.copyTo(tmp);
    for (int k = 0; k < n_species; k++) {
      ref_rhoDiffCoeffs.push_back(tmp[k]);
    }
    free(tmp);
  }

  {
    occa::properties props = kernel_properties;
    props["compiler_flags"] += " -include " + cacheDir + "/ref.h";
    occa::kernel kernel = buildKernel("ref.okl", "refCp", props);
    auto tmp = (double *)calloc(1, sizeof(double));
    auto o_tmp = device.malloc(1 * sizeof(double));
    kernel(o_tmp);
    o_tmp.copyTo(tmp);
    ref_cp = tmp[0];
    free(tmp);
  }

  {
    occa::properties props = kernel_properties;
    props["compiler_flags"] += " -include " + cacheDir + "/ref.h";
    occa::kernel kernel = buildKernel("ref.okl", "refCv", props);
    auto tmp = (double *)calloc(1, sizeof(double));
    auto o_tmp = device.malloc(1 * sizeof(double));
    kernel(o_tmp);
    o_tmp.copyTo(tmp);
    ref_cv = tmp[0];
    free(tmp);
  }

  {
    occa::properties props = kernel_properties;
    props["compiler_flags"] += " -include " + cacheDir + "/ref.h";
    occa::kernel kernel = buildKernel("ref.okl", "refViscosity", props);
    auto tmp = (double *)calloc(1, sizeof(double));
    auto o_tmp = device.malloc(1 * sizeof(double));
    kernel(o_tmp);
    o_tmp.copyTo(tmp);
    ref_viscosity = tmp[0];
    free(tmp);
  }

  {
    occa::properties props = kernel_properties;
    props["compiler_flags"] += " -include " + cacheDir + "/ref.h";
    occa::kernel kernel = buildKernel("ref.okl", "refConductivity", props);
    auto tmp = (double *)calloc(1, sizeof(double));
    auto o_tmp = device.malloc(1 * sizeof(double));
    kernel(o_tmp);
    o_tmp.copyTo(tmp);
    ref_conductivity = tmp[0];
    free(tmp);
  }

  {
    occa::properties includeProp;
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fheat_capacity_R.inc";

    auto prop = kernel_properties_mixed;
    if (useFP64Transport) prop = kernel_properties;

    thermoCoeffs_fpmix_kernel =
        buildKernel("thermoCoeffs.okl", "thermoCoeffs", addOccaCompilerFlags(includeProp, prop));

    prop = kernel_properties_fp32;
    thermoCoeffs_fp32_kernel =
        buildKernel("thermoCoeffs.okl", "thermoCoeffs", addOccaCompilerFlags(includeProp, prop));
  }

  {
    occa::properties includeProp;
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fheat_capacity_R.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fenthalpy_RT.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/rates.inc";

    production_rates_kernel =
        buildKernel("productionRates.okl", "productionRates", addOccaCompilerFlags(includeProp, kernel_properties));
  }

  {
    occa::properties includeProp;
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fheat_capacity_R.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fenthalpy_RT.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/frates.inc";

    production_rates_fpmix_kernel = buildKernel("productionRates.okl",
                                                 "productionRates",
                                                 addOccaCompilerFlags(includeProp, kernel_properties_mixed));

    production_rates_fp32_kernel = buildKernel("productionRates.okl",
                                                "productionRates",
                                                addOccaCompilerFlags(includeProp, kernel_properties_fp32));
  }

  if (transport) {
    const auto invRe = 1 / _nekCRF::Re();
    const auto invRePr = 1 / (_nekCRF::Re() * _nekCRF::Pr());

    occa::properties includeProp;
    includeProp["compiler_flags"] = " -include " + cacheDir + "/fconductivity.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fviscosity.inc";
    includeProp["compiler_flags"] += " -include " + cacheDir + "/fdiffusivity.inc";

    auto prop = kernel_properties_mixed;
    if (useFP64Transport) prop = kernel_properties;
    prop["defines/p_invRe"] += invRe;
    prop["defines/p_invRePr"] += invRePr;

    transport_fpmix_kernel =
        buildKernel("transport.okl", "transport", addOccaCompilerFlags(includeProp, prop));


    prop = kernel_properties_fp32;
    prop["defines/p_invRe"] += invRe;
    prop["defines/p_invRePr"] += invRePr;

    transport_fp32_kernel =
        buildKernel("transport.okl", "transport", addOccaCompilerFlags(includeProp, prop));

  }
}

///////////////////////////////////////////////////////////////////////////////
//                                  API
///////////////////////////////////////////////////////////////////////////////

void _nekCRF::init(const std::string &model_path,
                   occa::device _device,
                   occa::properties _props,
                   int _group_size,
                   bool _unroll_loops,
                   bool _unroll_loops_transport,
                   bool _pragma_unroll_loops,
		   bool _loop_gibbsexp,
 		   bool _nonsymDij,
 		   bool _fit_rcpDiffCoeffs,
                   int _align_width,
                   const std::string &_target,
                   bool _useFP64Transport,
                   MPI_Comm _comm,
                   bool _verbose)
{
  _nekCRF::nekCRFBuildKernel_t build = [&](const std::string &path, const std::string &fileName, occa::properties prop)
  {
    return _buildKernel(path, fileName, prop);
  };
  _nekCRF::init(model_path,
                _device,
                _props,
                _group_size,
                _unroll_loops,
                _unroll_loops_transport,
                _pragma_unroll_loops,
	       	_loop_gibbsexp,
	       	_nonsymDij,
	       	_fit_rcpDiffCoeffs,
                _align_width,
                _target,
                _useFP64Transport,
                _comm,
                build,
                _verbose);
}

void _nekCRF::init(const std::string &model_path,
                   occa::device _device,
                   occa::properties _props,
                   int _group_size,
                   bool _unroll_loops,
                   bool _unroll_loops_transport,
                   bool _pragma_unroll_loops,
                   bool _loop_gibbsexp,
                   bool _nonsymDij,
                   bool _fit_rcpDiffCoeffs,
                   int _align_width,
                   const std::string &_target,
                   bool _useFP64Transport,
                   MPI_Comm _comm,
                   nekCRFBuildKernel_t build,
                   bool _verbose)
{
  buildKernel = build;

  verbose = _verbose;
  comm = _comm;
  device = _device;

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
      kernel_properties["compiler_flags"] += " -include cmath";
      kernel_properties["compiler_flags"] += " -include cstdio";
    }
  }
  else {
    kernel_properties = _props;
  }

  yamlPath = fs::path(model_path);
  group_size = std::max(_group_size, 32);
  unroll_loops = _unroll_loops;
  unroll_loops_transport = _unroll_loops_transport;
  pragma_unroll_loops = _pragma_unroll_loops;
  loop_gibbsexp = _loop_gibbsexp;
  nonsymDij = _nonsymDij;
  fit_rcpDiffCoeffs = _fit_rcpDiffCoeffs;
  align_width = _align_width;
  target = _target;
  useFP64Transport = _useFP64Transport;

  setup();
  MPI_Barrier(comm);
}

double _nekCRF::Re()
{
  const auto nu = ref_viscosity / ref_density;
  return (ref_velocity * ref_length) / nu;
}

double _nekCRF::Pr()
{
  const auto alpha = ref_conductivity / (ref_density * ref_cp);
  const auto nu = ref_viscosity / ref_density;
  return nu / alpha;
}

const std::vector<double> _nekCRF::Le()
{
  std::vector<double> tmp;
  for (int k = 0; k < n_species; k++) {
    tmp.push_back(ref_conductivity / (ref_cp * ref_rhoDiffCoeffs[k]));
  }
  return tmp;
}

double _nekCRF::refTime() { return ref_time; }

double _nekCRF::refLength() { return ref_length; }

double _nekCRF::refVelocity() { return ref_velocity; }

double _nekCRF::refPressure() { return ref_pressure; }

double _nekCRF::refTemperature() { return ref_temperature; }

const std::vector<double> _nekCRF::refMassFractions() { return ref_mass_fractions; }

double _nekCRF::refCp() { return ref_cp; }

double _nekCRF::refCv() { return ref_cv; }

double _nekCRF::refMeanMolecularWeight() { return ref_meanMolarMass; }

double _nekCRF::refDensity() { return ref_density; }

double _nekCRF::refViscosity() { return ref_viscosity; }

double _nekCRF::refThermalConductivity() { return ref_conductivity; }

const std::vector<double> _nekCRF::refRhoDiffCoeffs() { return ref_rhoDiffCoeffs; }

const std::vector<double> _nekCRF::refDiffCoeffs()
{
  std::vector<double> tmp;
  for (int k = 0; k < n_species; k++) {
    tmp.push_back(ref_rhoDiffCoeffs[k] / ref_density);
  }
  return tmp;
}

void _nekCRF::build(double _ref_pressure,
                    double _ref_temperature,
                    double _ref_length,
                    double _ref_velocity,
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
  ref_length = _ref_length;
  ref_velocity = _ref_velocity;
  ref_time = ref_length / ref_velocity;

  double sum = 0.;
  for (int k = 0; k < nSpecies(); k++) {
    sum += _ref_mass_fractions[k] / m_molar[k];
    ref_mass_fractions.push_back(_ref_mass_fractions[k]);
  }
  ref_meanMolarMass = 1. / sum;

  ref_density = (ref_pressure / R / ref_temperature) * ref_meanMolarMass;

  auto ref_mole_fractions = new double[nSpecies()];
  std::string ref_mole_fractions_string;
  for (int k = 0; k < nSpecies(); k++) {
    ref_mole_fractions[k] = 1. / m_molar[k] * ref_meanMolarMass * _ref_mass_fractions[k];
    ref_mole_fractions_string += _nekCRF::to_string_f(ref_mole_fractions[k]);
    if (k < nSpecies() - 1)
      ref_mole_fractions_string += ',';
  }

  const auto installDir = std::string(getenv("NEKCRF_PATH") ?: ".");
  if (rank == 0) {
    std::string cmdline(installDir +
                        "/generator/generate.py" +
                        " --mechanism " +
                        yamlPath + " --output " + cacheDir + " --velocityRef " + _nekCRF::to_string_f(ref_velocity) +
                        " --lengthRef " + _nekCRF::to_string_f(ref_length) + " --pressureRef " +
                        _nekCRF::to_string_f(ref_pressure) + " --temperatureRef " + _nekCRF::to_string_f(ref_temperature) +
                        " --moleFractionsRef " + ref_mole_fractions_string.c_str() + " --align-width " +
                        std::to_string(align_width) + " --target " + target);
    if (unroll_loops)
      cmdline.append(" --unroll-loops");
    if (unroll_loops_transport)
      cmdline.append(" --unroll-loops-transport");
    if (pragma_unroll_loops)
      cmdline.append(" --pragma-unroll-loops");
    if (loop_gibbsexp)
      cmdline.append(" --loop-gibbsexp");
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
 
      fileSync(std::string(cacheDir + "/fconductivity.inc").c_str());
      fileSync(std::string(cacheDir + "/fdiffusivity.inc").c_str());
      fileSync(std::string(cacheDir + "/fenthalpy_RT.inc").c_str());
      fileSync(std::string(cacheDir + "/fheat_capacity_R.inc").c_str());
      fileSync(std::string(cacheDir + "/fviscosity.inc").c_str());
      fileSync(std::string(cacheDir + "/frates.inc").c_str());
      fileSync(std::string(cacheDir + "/rates.inc").c_str());
      fileSync(std::string(cacheDir + "/ref.h").c_str());

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

  ref_mass_rate = ref_density / ref_time;
  ref_volumetric_energy_rate = -ref_cp * ref_temperature * ref_mass_rate;

  delete[] ref_mole_fractions;

  if (rank == 0) {
    std::cout   << "\n================= nekCRF =================\n";
    if (isStandalone())
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
    for (auto &&species : _nekCRF::speciesNames()) {
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
      for (auto &&species : _nekCRF::speciesNames()) {
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

    if(!isStandalone()) {
      std::cout  << "tRef: " << ref_time << " s\n" 
                 << "rhoRef: " << ref_density << " kg/m^3\n"
                 << "mueRef: " << ref_viscosity << " kg/(ms)\n"
                 << "lambdaRef: " << ref_conductivity << " W/(mK)\n"
                 << "cpRef: " << ref_cp << " J/K\n"
                 << "gamma0Ref: " << ref_cp/ref_cv << "\n"
                 << "WbarRef: " << ref_meanMolarMass << " kg/mol\n";
      printVec("DiRef: ", refDiffCoeffs(), " m^2/s");
      std::cout  << "Re: " << Re() << "\n"
                 << "Pr: " << Pr() << "\n";
      printVec("Le: ", Le()); 
      std::cout << "\n";
    }
    fflush(stdout);
  }

  occa::env::OCCA_CACHE_DIR = occaCacheDir0;
  setenv("OCCA_CACHE_DIR", occaCacheDir0.c_str(), 1);

  MPI_Barrier(comm);
}

void _nekCRF::productionRates(int n_states,
                              int offsetT,
                              int offset,
                              bool normalize, /* by rhoCp and rho */
                              bool lumpInert,
                              double pressure,
                              const occa::memory &o_state,
                              occa::memory &o_rates,
                              bool fp32)
{
  assert(initialized);

  if (o_state.dtype() == occa::dtype::float_)
    fp32 = true;
  const bool fpmix = (o_state.dtype() != occa::dtype::float_ && fp32) ? true : false;

  occa::kernel kernel = production_rates_kernel;
  if (fpmix)
    kernel = production_rates_fpmix_kernel;
  else if (fp32)
    kernel = production_rates_fp32_kernel;

  const double pressure_R = pressure * ref_pressure / R;

  kernel(n_states,
         offsetT,
         offset,
         (int)normalize,
         (int)lumpInert,
         pressure_R,
         o_state,
         o_rates,
         ref_temperature,
         1. / ref_mass_rate,
         1. / ref_volumetric_energy_rate,
         1. / ref_density,
         1. / ref_cp);
}

void _nekCRF::mixtureAvgTransportProps(int n_states,
                                       int offsetT,
                                       int offset,
                                       double pressure,
                                       const occa::memory &o_state,
                                       occa::memory &o_viscosity,
                                       occa::memory &o_conductivity,
                                       occa::memory &o_density_diffusivity)
{
  assert(initialized);

  bool fp32 = false;
  if (o_state.dtype() == occa::dtype::float_)
    fp32 = true;

  occa::kernel kernel = transport_fpmix_kernel;
  if (fp32)
    kernel = transport_fp32_kernel;

  kernel(n_states, 
         offsetT, 
         offset, 
         pressure, 
         o_state, 
         o_conductivity, 
         o_viscosity,
         o_density_diffusivity);
}

void _nekCRF::thermodynamicProps(int n_states,
                                 int offsetT,
                                 int offset,
                                 double pressure,
                                 const occa::memory &o_state,
                                 occa::memory &o_rho,
                                 occa::memory &o_cpi,
                                 occa::memory &o_rhocp,
                                 occa::memory &o_mmw)
{
  assert(initialized);

  bool fp32 = false;
  if (o_state.dtype() == occa::dtype::float_)
    fp32 = true;

  occa::kernel kernel = thermoCoeffs_fpmix_kernel;
  if (fp32)
    kernel = thermoCoeffs_fp32_kernel;

  const double pressure_R = pressure * ref_pressure / R;

  kernel(n_states,
         offsetT,
         offset,
         pressure_R,
         o_state,
         o_rho,
         o_cpi,
         o_rhocp,
         o_mmw,
         ref_temperature,
         1. / ref_density,
         1. / ref_cp,
         1. / ref_meanMolarMass);
}

int _nekCRF::nSpecies() 
{ 
  return n_species; 
}

int _nekCRF::nActiveSpecies() 
{ 
  return n_active_species; 
}

const std::vector<double> _nekCRF::molecularWeights()
{
  std::vector<double> tmp;
  for (int k = 0; k < n_species; k++) {
    tmp.push_back(m_molar[k] / ref_meanMolarMass);
  }
  return tmp;
}

const std::vector<std::string> _nekCRF::speciesNames() 
{ 
  return ::species_names; 
}

int _nekCRF::speciesIndex(const std::string &name)
{
  auto it = find(::species_names.begin(), ::species_names.end(), name);
  if (it != ::species_names.end())
    return it - ::species_names.begin();
  return -1;
}
