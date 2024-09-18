#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  H
1  O
2  N
*/

// Species
#define H2_ID 0
#define O2_ID 1
#define H2O_ID 2
#define H_ID 3
#define O_ID 4
#define OH_ID 5
#define HO2_ID 6
#define H2O2_ID 7
#define N2_ID 8

#define NUM_ELEMENTS 3
#define NUM_SPECIES 9
#define NUM_IONS 0
#define NUM_REACTIONS 21

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 3;
  kk = 9;
  ii = 21;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[9] = {
  0.4960317460317460, // H2
  0.0312519532470779, // O2
  0.0555092978073827, // H2O
  0.9920634920634921, // H
  0.0625039064941559, // O
  0.0587993179279120, // OH
  0.0302975216627280, // HO2
  0.0293996589639560, // H2O2
  0.0356964374955379, // N2
};
#endif
const amrex::Real h_global_imw[9] = {
  0.4960317460317460, // H2
  0.0312519532470779, // O2
  0.0555092978073827, // H2O
  0.9920634920634921, // H
  0.0625039064941559, // O
  0.0587993179279120, // OH
  0.0302975216627280, // HO2
  0.0293996589639560, // H2O2
  0.0356964374955379, // N2
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[9] = {
  2.016000,  // H2
  31.998000, // O2
  18.015000, // H2O
  1.008000,  // H
  15.999000, // O
  17.007000, // OH
  33.006000, // HO2
  34.014000, // H2O2
  28.014000, // N2
};
#endif
const amrex::Real h_global_mw[9] = {
  2.016000,  // H2
  31.998000, // O2
  18.015000, // H2O
  1.008000,  // H
  15.999000, // O
  17.007000, // OH
  33.006000, // HO2
  34.014000, // H2O2
  28.014000, // N2
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.4960317460317460; // H2
  imw_new[1] = 0.0312519532470779; // O2
  imw_new[2] = 0.0555092978073827; // H2O
  imw_new[3] = 0.9920634920634921; // H
  imw_new[4] = 0.0625039064941559; // O
  imw_new[5] = 0.0587993179279120; // OH
  imw_new[6] = 0.0302975216627280; // HO2
  imw_new[7] = 0.0293996589639560; // H2O2
  imw_new[8] = 0.0356964374955379; // N2
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 2.016000;  // H2
  mw_new[1] = 31.998000; // O2
  mw_new[2] = 18.015000; // H2O
  mw_new[3] = 1.008000;  // H
  mw_new[4] = 15.999000; // O
  mw_new[5] = 17.007000; // OH
  mw_new[6] = 33.006000; // HO2
  mw_new[7] = 34.014000; // H2O2
  mw_new[8] = 28.014000; // N2
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 3: H
  species[3] = +1.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +2.29812431e+00 + 8.24944174e-04 * T - 8.14301529e-07 * T2 -
                 9.47543433e-11 * T3 + 4.13487224e-13 * T4;
    // species 1: O2
    species[1] = +2.21293640e+00 + 1.12748635e-03 * T - 5.75615047e-07 * T2 +
                 1.31387723e-09 * T3 - 8.76855392e-13 * T4;
    // species 2: H2O
    species[2] = +2.38684249e+00 + 3.47498246e-03 * T - 6.35469633e-06 * T2 +
                 6.96858127e-09 * T3 - 2.50658847e-12 * T4;
    // species 4: O
    species[4] = +1.94642878e+00 - 1.63816649e-03 * T + 2.42103170e-06 * T2 -
                 1.60284319e-09 * T3 + 3.89069636e-13 * T4;
    // species 5: OH
    species[5] = +3.12530561e+00 - 3.22544939e-03 * T + 6.52764691e-06 * T2 -
                 5.79853643e-09 * T3 + 2.06237379e-12 * T4;
    // species 6: HO2
    species[6] = +3.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
                 2.42763894e-08 * T3 + 9.29225124e-12 * T4;
    // species 7: H2O2
    species[7] = +2.38875365e+00 + 6.56922581e-03 * T - 1.48501258e-07 * T2 -
                 4.62580552e-09 * T3 + 2.47151475e-12 * T4;
    // species 8: N2
    species[8] = +2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
  } else {
    // species 0: H2
    species[0] = +1.99142337e+00 + 7.00064411e-04 * T - 5.63382869e-08 * T2 -
                 9.23157818e-12 * T3 + 1.58275179e-15 * T4;
    // species 1: O2
    species[1] = +2.69757819e+00 + 6.13519689e-04 * T - 1.25884199e-07 * T2 +
                 1.77528148e-11 * T3 - 1.13643531e-15 * T4;
    // species 2: H2O
    species[2] = +1.67214561e+00 + 3.05629289e-03 * T - 8.73026011e-07 * T2 +
                 1.20099639e-10 * T3 - 6.39161787e-15 * T4;
    // species 4: O
    species[4] = +1.54205966e+00 - 2.75506191e-05 * T - 3.10280335e-09 * T2 +
                 4.55106742e-12 * T3 - 4.36805150e-16 * T4;
    // species 5: OH
    species[5] = +1.86472886e+00 + 1.05650448e-03 * T - 2.59082758e-07 * T2 +
                 3.05218674e-11 * T3 - 1.33195876e-15 * T4;
    // species 6: HO2
    species[6] = +3.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
                 1.14246370e-10 * T3 - 1.07908535e-14 * T4;
    // species 7: H2O2
    species[7] = +3.57316685e+00 + 4.33613639e-03 * T - 1.47468882e-06 * T2 +
                 2.34890357e-10 * T3 - 1.43165356e-14 * T4;
    // species 8: N2
    species[8] = +1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 3: H
  species[3] = +2.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +3.29812431e+00 + 8.24944174e-04 * T - 8.14301529e-07 * T2 -
                 9.47543433e-11 * T3 + 4.13487224e-13 * T4;
    // species 1: O2
    species[1] = +3.21293640e+00 + 1.12748635e-03 * T - 5.75615047e-07 * T2 +
                 1.31387723e-09 * T3 - 8.76855392e-13 * T4;
    // species 2: H2O
    species[2] = +3.38684249e+00 + 3.47498246e-03 * T - 6.35469633e-06 * T2 +
                 6.96858127e-09 * T3 - 2.50658847e-12 * T4;
    // species 4: O
    species[4] = +2.94642878e+00 - 1.63816649e-03 * T + 2.42103170e-06 * T2 -
                 1.60284319e-09 * T3 + 3.89069636e-13 * T4;
    // species 5: OH
    species[5] = +4.12530561e+00 - 3.22544939e-03 * T + 6.52764691e-06 * T2 -
                 5.79853643e-09 * T3 + 2.06237379e-12 * T4;
    // species 6: HO2
    species[6] = +4.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
                 2.42763894e-08 * T3 + 9.29225124e-12 * T4;
    // species 7: H2O2
    species[7] = +3.38875365e+00 + 6.56922581e-03 * T - 1.48501258e-07 * T2 -
                 4.62580552e-09 * T3 + 2.47151475e-12 * T4;
    // species 8: N2
    species[8] = +3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
  } else {
    // species 0: H2
    species[0] = +2.99142337e+00 + 7.00064411e-04 * T - 5.63382869e-08 * T2 -
                 9.23157818e-12 * T3 + 1.58275179e-15 * T4;
    // species 1: O2
    species[1] = +3.69757819e+00 + 6.13519689e-04 * T - 1.25884199e-07 * T2 +
                 1.77528148e-11 * T3 - 1.13643531e-15 * T4;
    // species 2: H2O
    species[2] = +2.67214561e+00 + 3.05629289e-03 * T - 8.73026011e-07 * T2 +
                 1.20099639e-10 * T3 - 6.39161787e-15 * T4;
    // species 4: O
    species[4] = +2.54205966e+00 - 2.75506191e-05 * T - 3.10280335e-09 * T2 +
                 4.55106742e-12 * T3 - 4.36805150e-16 * T4;
    // species 5: OH
    species[5] = +2.86472886e+00 + 1.05650448e-03 * T - 2.59082758e-07 * T2 +
                 3.05218674e-11 * T3 - 1.33195876e-15 * T4;
    // species 6: HO2
    species[6] = +4.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
                 1.14246370e-10 * T3 - 1.07908535e-14 * T4;
    // species 7: H2O2
    species[7] = +4.57316685e+00 + 4.33613639e-03 * T - 1.47468882e-06 * T2 +
                 2.34890357e-10 * T3 - 1.43165356e-14 * T4;
    // species 8: N2
    species[8] = +2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 3: H
  species[3] = +2.547162700000000e+04 * invT + 2.960117638000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = -1.012520870000000e+03 * invT + 6.592218400000000e+00 -
                 3.298124310000000e+00 * logT - 4.124720870000000e-04 * T +
                 1.357169215000000e-07 * T2 + 7.896195275000000e-12 * T3 -
                 2.067436120000000e-14 * T4;
    // species 1: O2
    species[1] = -1.005249020000000e+03 * invT - 2.821801190000000e+00 -
                 3.212936400000000e+00 * logT - 5.637431750000000e-04 * T +
                 9.593584116666666e-08 * T2 - 1.094897691666667e-10 * T3 +
                 4.384276960000000e-14 * T4;
    // species 2: H2O
    species[2] = -3.020811330000000e+04 * invT + 7.966096399999998e-01 -
                 3.386842490000000e+00 * logT - 1.737491230000000e-03 * T +
                 1.059116055000000e-06 * T2 - 5.807151058333333e-10 * T3 +
                 1.253294235000000e-13 * T4;
    // species 4: O
    species[4] = +2.914764450000000e+04 * invT - 1.756619999999964e-02 -
                 2.946428780000000e+00 * logT + 8.190832450000000e-04 * T -
                 4.035052833333333e-07 * T2 + 1.335702658333333e-10 * T3 -
                 1.945348180000000e-14 * T4;
    // species 5: OH
    species[5] = +3.346309130000000e+03 * invT + 4.815738570000000e+00 -
                 4.125305610000000e+00 * logT + 1.612724695000000e-03 * T -
                 1.087941151666667e-06 * T2 + 4.832113691666666e-10 * T3 -
                 1.031186895000000e-13 * T4;
    // species 6: HO2
    species[6] = +2.948080400000000e+02 * invT + 5.851355599999999e-01 -
                 4.301798010000000e+00 * logT + 2.374560255000000e-03 * T -
                 3.526381516666666e-06 * T2 + 2.023032450000000e-09 * T3 -
                 4.646125620000001e-13 * T4;
    // species 7: H2O2
    species[7] = -1.766314650000000e+04 * invT - 3.396609550000000e+00 -
                 3.388753650000000e+00 * logT - 3.284612905000000e-03 * T +
                 2.475020966666667e-08 * T2 + 3.854837933333333e-10 * T3 -
                 1.235757375000000e-13 * T4;
    // species 8: N2
    species[8] = -1.020900000000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041200000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427500000000e-13 * T4;
  } else {
    // species 0: H2
    species[0] = -8.350339970000000e+02 * invT + 4.346533540000000e+00 -
                 2.991423370000000e+00 * logT - 3.500322055000000e-04 * T +
                 9.389714483333333e-09 * T2 + 7.692981816666667e-13 * T3 -
                 7.913758950000000e-17 * T4;
    // species 1: O2
    species[1] = -1.233930180000000e+03 * invT + 5.084126000000002e-01 -
                 3.697578190000000e+00 * logT - 3.067598445000000e-04 * T +
                 2.098069983333333e-08 * T2 - 1.479401233333333e-12 * T3 +
                 5.682176550000000e-17 * T4;
    // species 2: H2O
    species[2] = -2.989920900000000e+04 * invT - 4.190671200000001e+00 -
                 2.672145610000000e+00 * logT - 1.528146445000000e-03 * T +
                 1.455043351666667e-07 * T2 - 1.000830325000000e-11 * T3 +
                 3.195808935000000e-16 * T4;
    // species 4: O
    species[4] = +2.923080270000000e+04 * invT - 2.378248450000000e+00 -
                 2.542059660000000e+00 * logT + 1.377530955000000e-05 * T +
                 5.171338916666667e-10 * T2 - 3.792556183333333e-13 * T3 +
                 2.184025750000000e-17 * T4;
    // species 5: OH
    species[5] = +3.683628750000000e+03 * invT - 2.836911870000000e+00 -
                 2.864728860000000e+00 * logT - 5.282522400000000e-04 * T +
                 4.318045966666667e-08 * T2 - 2.543488950000000e-12 * T3 +
                 6.659793800000000e-17 * T4;
    // species 6: HO2
    species[6] = +1.118567130000000e+02 * invT + 2.321087500000001e-01 -
                 4.017210900000000e+00 * logT - 1.119910065000000e-03 * T +
                 1.056096916666667e-07 * T2 - 9.520530833333334e-12 * T3 +
                 5.395426750000000e-16 * T4;
    // species 7: H2O2
    species[7] = -1.800696090000000e+04 * invT + 4.072029891000000e+00 -
                 4.573166850000000e+00 * logT - 2.168068195000000e-03 * T +
                 2.457814700000000e-07 * T2 - 1.957419641666667e-11 * T3 +
                 7.158267800000000e-16 * T4;
    // species 8: N2
    species[8] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439885000000000e-04 * T +
                 9.474601666666666e-08 * T2 - 8.414199999999999e-12 * T3 +
                 3.376675500000000e-16 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 3: H
  species[3] = +2.54716270e+04 * invT + 1.96011764e+00 - 2.50000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = -1.01252087e+03 * invT + 5.59221840e+00 -
                 3.29812431e+00 * logT - 4.12472087e-04 * T +
                 1.35716922e-07 * T2 + 7.89619527e-12 * T3 -
                 2.06743612e-14 * T4;
    // species 1: O2
    species[1] = -1.00524902e+03 * invT - 3.82180119e+00 -
                 3.21293640e+00 * logT - 5.63743175e-04 * T +
                 9.59358412e-08 * T2 - 1.09489769e-10 * T3 +
                 4.38427696e-14 * T4;
    // species 2: H2O
    species[2] = -3.02081133e+04 * invT - 2.03390360e-01 -
                 3.38684249e+00 * logT - 1.73749123e-03 * T +
                 1.05911606e-06 * T2 - 5.80715106e-10 * T3 +
                 1.25329424e-13 * T4;
    // species 4: O
    species[4] = +2.91476445e+04 * invT - 1.01756620e+00 -
                 2.94642878e+00 * logT + 8.19083245e-04 * T -
                 4.03505283e-07 * T2 + 1.33570266e-10 * T3 -
                 1.94534818e-14 * T4;
    // species 5: OH
    species[5] = +3.34630913e+03 * invT + 3.81573857e+00 -
                 4.12530561e+00 * logT + 1.61272470e-03 * T -
                 1.08794115e-06 * T2 + 4.83211369e-10 * T3 -
                 1.03118689e-13 * T4;
    // species 6: HO2
    species[6] = +2.94808040e+02 * invT - 4.14864440e-01 -
                 4.30179801e+00 * logT + 2.37456025e-03 * T -
                 3.52638152e-06 * T2 + 2.02303245e-09 * T3 -
                 4.64612562e-13 * T4;
    // species 7: H2O2
    species[7] = -1.76631465e+04 * invT - 4.39660955e+00 -
                 3.38875365e+00 * logT - 3.28461290e-03 * T +
                 2.47502097e-08 * T2 + 3.85483793e-10 * T3 -
                 1.23575738e-13 * T4;
    // species 8: N2
    species[8] = -1.02090000e+03 * invT - 1.65169500e+00 -
                 3.29867700e+00 * logT - 7.04120000e-04 * T +
                 6.60537000e-07 * T2 - 4.70126250e-10 * T3 +
                 1.22242750e-13 * T4;
  } else {
    // species 0: H2
    species[0] = -8.35033997e+02 * invT + 3.34653354e+00 -
                 2.99142337e+00 * logT - 3.50032206e-04 * T +
                 9.38971448e-09 * T2 + 7.69298182e-13 * T3 -
                 7.91375895e-17 * T4;
    // species 1: O2
    species[1] = -1.23393018e+03 * invT - 4.91587400e-01 -
                 3.69757819e+00 * logT - 3.06759845e-04 * T +
                 2.09806998e-08 * T2 - 1.47940123e-12 * T3 +
                 5.68217655e-17 * T4;
    // species 2: H2O
    species[2] = -2.98992090e+04 * invT - 5.19067120e+00 -
                 2.67214561e+00 * logT - 1.52814644e-03 * T +
                 1.45504335e-07 * T2 - 1.00083033e-11 * T3 +
                 3.19580894e-16 * T4;
    // species 4: O
    species[4] = +2.92308027e+04 * invT - 3.37824845e+00 -
                 2.54205966e+00 * logT + 1.37753096e-05 * T +
                 5.17133892e-10 * T2 - 3.79255618e-13 * T3 +
                 2.18402575e-17 * T4;
    // species 5: OH
    species[5] = +3.68362875e+03 * invT - 3.83691187e+00 -
                 2.86472886e+00 * logT - 5.28252240e-04 * T +
                 4.31804597e-08 * T2 - 2.54348895e-12 * T3 +
                 6.65979380e-17 * T4;
    // species 6: HO2
    species[6] = +1.11856713e+02 * invT - 7.67891250e-01 -
                 4.01721090e+00 * logT - 1.11991006e-03 * T +
                 1.05609692e-07 * T2 - 9.52053083e-12 * T3 +
                 5.39542675e-16 * T4;
    // species 7: H2O2
    species[7] = -1.80069609e+04 * invT + 3.07202989e+00 -
                 4.57316685e+00 * logT - 2.16806820e-03 * T +
                 2.45781470e-07 * T2 - 1.95741964e-11 * T3 +
                 7.15826780e-16 * T4;
    // species 8: N2
    species[8] = -9.22797700e+02 * invT - 4.05388800e+00 -
                 2.92664000e+00 * logT - 7.43988500e-04 * T +
                 9.47460167e-08 * T2 - 8.41420000e-12 * T3 +
                 3.37667550e-16 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 3: H
  species[3] = +1.50000000e+00 + 2.54716270e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +2.29812431e+00 + 4.12472087e-04 * T - 2.71433843e-07 * T2 -
                 2.36885858e-11 * T3 + 8.26974448e-14 * T4 -
                 1.01252087e+03 * invT;
    // species 1: O2
    species[1] = +2.21293640e+00 + 5.63743175e-04 * T - 1.91871682e-07 * T2 +
                 3.28469308e-10 * T3 - 1.75371078e-13 * T4 -
                 1.00524902e+03 * invT;
    // species 2: H2O
    species[2] = +2.38684249e+00 + 1.73749123e-03 * T - 2.11823211e-06 * T2 +
                 1.74214532e-09 * T3 - 5.01317694e-13 * T4 -
                 3.02081133e+04 * invT;
    // species 4: O
    species[4] = +1.94642878e+00 - 8.19083245e-04 * T + 8.07010567e-07 * T2 -
                 4.00710797e-10 * T3 + 7.78139272e-14 * T4 +
                 2.91476445e+04 * invT;
    // species 5: OH
    species[5] = +3.12530561e+00 - 1.61272470e-03 * T + 2.17588230e-06 * T2 -
                 1.44963411e-09 * T3 + 4.12474758e-13 * T4 +
                 3.34630913e+03 * invT;
    // species 6: HO2
    species[6] = +3.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
                 6.06909735e-09 * T3 + 1.85845025e-12 * T4 +
                 2.94808040e+02 * invT;
    // species 7: H2O2
    species[7] = +2.38875365e+00 + 3.28461290e-03 * T - 4.95004193e-08 * T2 -
                 1.15645138e-09 * T3 + 4.94302950e-13 * T4 -
                 1.76631465e+04 * invT;
    // species 8: N2
    species[8] = +2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
  } else {
    // species 0: H2
    species[0] = +1.99142337e+00 + 3.50032206e-04 * T - 1.87794290e-08 * T2 -
                 2.30789455e-12 * T3 + 3.16550358e-16 * T4 -
                 8.35033997e+02 * invT;
    // species 1: O2
    species[1] = +2.69757819e+00 + 3.06759845e-04 * T - 4.19613997e-08 * T2 +
                 4.43820370e-12 * T3 - 2.27287062e-16 * T4 -
                 1.23393018e+03 * invT;
    // species 2: H2O
    species[2] = +1.67214561e+00 + 1.52814644e-03 * T - 2.91008670e-07 * T2 +
                 3.00249098e-11 * T3 - 1.27832357e-15 * T4 -
                 2.98992090e+04 * invT;
    // species 4: O
    species[4] = +1.54205966e+00 - 1.37753096e-05 * T - 1.03426778e-09 * T2 +
                 1.13776685e-12 * T3 - 8.73610300e-17 * T4 +
                 2.92308027e+04 * invT;
    // species 5: OH
    species[5] = +1.86472886e+00 + 5.28252240e-04 * T - 8.63609193e-08 * T2 +
                 7.63046685e-12 * T3 - 2.66391752e-16 * T4 +
                 3.68362875e+03 * invT;
    // species 6: HO2
    species[6] = +3.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
                 2.85615925e-11 * T3 - 2.15817070e-15 * T4 +
                 1.11856713e+02 * invT;
    // species 7: H2O2
    species[7] = +3.57316685e+00 + 2.16806820e-03 * T - 4.91562940e-07 * T2 +
                 5.87225893e-11 * T3 - 2.86330712e-15 * T4 -
                 1.80069609e+04 * invT;
    // species 8: N2
    species[8] = +1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 3: H
  species[3] = +2.50000000e+00 + 2.54716270e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +3.29812431e+00 + 4.12472087e-04 * T - 2.71433843e-07 * T2 -
                 2.36885858e-11 * T3 + 8.26974448e-14 * T4 -
                 1.01252087e+03 * invT;
    // species 1: O2
    species[1] = +3.21293640e+00 + 5.63743175e-04 * T - 1.91871682e-07 * T2 +
                 3.28469308e-10 * T3 - 1.75371078e-13 * T4 -
                 1.00524902e+03 * invT;
    // species 2: H2O
    species[2] = +3.38684249e+00 + 1.73749123e-03 * T - 2.11823211e-06 * T2 +
                 1.74214532e-09 * T3 - 5.01317694e-13 * T4 -
                 3.02081133e+04 * invT;
    // species 4: O
    species[4] = +2.94642878e+00 - 8.19083245e-04 * T + 8.07010567e-07 * T2 -
                 4.00710797e-10 * T3 + 7.78139272e-14 * T4 +
                 2.91476445e+04 * invT;
    // species 5: OH
    species[5] = +4.12530561e+00 - 1.61272470e-03 * T + 2.17588230e-06 * T2 -
                 1.44963411e-09 * T3 + 4.12474758e-13 * T4 +
                 3.34630913e+03 * invT;
    // species 6: HO2
    species[6] = +4.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
                 6.06909735e-09 * T3 + 1.85845025e-12 * T4 +
                 2.94808040e+02 * invT;
    // species 7: H2O2
    species[7] = +3.38875365e+00 + 3.28461290e-03 * T - 4.95004193e-08 * T2 -
                 1.15645138e-09 * T3 + 4.94302950e-13 * T4 -
                 1.76631465e+04 * invT;
    // species 8: N2
    species[8] = +3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
  } else {
    // species 0: H2
    species[0] = +2.99142337e+00 + 3.50032206e-04 * T - 1.87794290e-08 * T2 -
                 2.30789455e-12 * T3 + 3.16550358e-16 * T4 -
                 8.35033997e+02 * invT;
    // species 1: O2
    species[1] = +3.69757819e+00 + 3.06759845e-04 * T - 4.19613997e-08 * T2 +
                 4.43820370e-12 * T3 - 2.27287062e-16 * T4 -
                 1.23393018e+03 * invT;
    // species 2: H2O
    species[2] = +2.67214561e+00 + 1.52814644e-03 * T - 2.91008670e-07 * T2 +
                 3.00249098e-11 * T3 - 1.27832357e-15 * T4 -
                 2.98992090e+04 * invT;
    // species 4: O
    species[4] = +2.54205966e+00 - 1.37753096e-05 * T - 1.03426778e-09 * T2 +
                 1.13776685e-12 * T3 - 8.73610300e-17 * T4 +
                 2.92308027e+04 * invT;
    // species 5: OH
    species[5] = +2.86472886e+00 + 5.28252240e-04 * T - 8.63609193e-08 * T2 +
                 7.63046685e-12 * T3 - 2.66391752e-16 * T4 +
                 3.68362875e+03 * invT;
    // species 6: HO2
    species[6] = +4.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
                 2.85615925e-11 * T3 - 2.15817070e-15 * T4 +
                 1.11856713e+02 * invT;
    // species 7: H2O2
    species[7] = +4.57316685e+00 + 2.16806820e-03 * T - 4.91562940e-07 * T2 +
                 5.87225893e-11 * T3 - 2.86330712e-15 * T4 -
                 1.80069609e+04 * invT;
    // species 8: N2
    species[8] = +2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 3: H
  species[3] = +2.50000000e+00 * logT - 4.60117638e-01;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +3.29812431e+00 * logT + 8.24944174e-04 * T -
                 4.07150765e-07 * T2 - 3.15847811e-11 * T3 +
                 1.03371806e-13 * T4 - 3.29409409e+00;
    // species 1: O2
    species[1] = +3.21293640e+00 * logT + 1.12748635e-03 * T -
                 2.87807523e-07 * T2 + 4.37959077e-10 * T3 -
                 2.19213848e-13 * T4 + 6.03473759e+00;
    // species 2: H2O
    species[2] = +3.38684249e+00 * logT + 3.47498246e-03 * T -
                 3.17734817e-06 * T2 + 2.32286042e-09 * T3 -
                 6.26647117e-13 * T4 + 2.59023285e+00;
    // species 4: O
    species[4] = +2.94642878e+00 * logT - 1.63816649e-03 * T +
                 1.21051585e-06 * T2 - 5.34281063e-10 * T3 +
                 9.72674090e-14 * T4 + 2.96399498e+00;
    // species 5: OH
    species[5] = +4.12530561e+00 * logT - 3.22544939e-03 * T +
                 3.26382346e-06 * T2 - 1.93284548e-09 * T3 +
                 5.15593447e-13 * T4 - 6.90432960e-01;
    // species 6: HO2
    species[6] = +4.30179801e+00 * logT - 4.74912051e-03 * T +
                 1.05791445e-05 * T2 - 8.09212980e-09 * T3 +
                 2.32306281e-12 * T4 + 3.71666245e+00;
    // species 7: H2O2
    species[7] = +3.38875365e+00 * logT + 6.56922581e-03 * T -
                 7.42506290e-08 * T2 - 1.54193517e-09 * T3 +
                 6.17878688e-13 * T4 + 6.78536320e+00;
    // species 8: N2
    species[8] = +3.29867700e+00 * logT + 1.40824000e-03 * T -
                 1.98161100e-06 * T2 + 1.88050500e-09 * T3 -
                 6.11213750e-13 * T4 + 3.95037200e+00;
  } else {
    // species 0: H2
    species[0] = +2.99142337e+00 * logT + 7.00064411e-04 * T -
                 2.81691434e-08 * T2 - 3.07719273e-12 * T3 +
                 3.95687948e-16 * T4 - 1.35511017e+00;
    // species 1: O2
    species[1] = +3.69757819e+00 * logT + 6.13519689e-04 * T -
                 6.29420995e-08 * T2 + 5.91760493e-12 * T3 -
                 2.84108828e-16 * T4 + 3.18916559e+00;
    // species 2: H2O
    species[2] = +2.67214561e+00 * logT + 3.05629289e-03 * T -
                 4.36513005e-07 * T2 + 4.00332130e-11 * T3 -
                 1.59790447e-15 * T4 + 6.86281681e+00;
    // species 4: O
    species[4] = +2.54205966e+00 * logT - 2.75506191e-05 * T -
                 1.55140167e-09 * T2 + 1.51702247e-12 * T3 -
                 1.09201287e-16 * T4 + 4.92030811e+00;
    // species 5: OH
    species[5] = +2.86472886e+00 * logT + 1.05650448e-03 * T -
                 1.29541379e-07 * T2 + 1.01739558e-11 * T3 -
                 3.32989690e-16 * T4 + 5.70164073e+00;
    // species 6: HO2
    species[6] = +4.01721090e+00 * logT + 2.23982013e-03 * T -
                 3.16829075e-07 * T2 + 3.80821233e-11 * T3 -
                 2.69771337e-15 * T4 + 3.78510215e+00;
    // species 7: H2O2
    species[7] = +4.57316685e+00 * logT + 4.33613639e-03 * T -
                 7.37344410e-07 * T2 + 7.82967857e-11 * T3 -
                 3.57913390e-15 * T4 + 5.01136959e-01;
    // species 8: N2
    species[8] = +2.92664000e+00 * logT + 1.48797700e-03 * T -
                 2.84238050e-07 * T2 + 3.36568000e-11 * T3 -
                 1.68833775e-15 * T4 + 5.98052800e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;

  // species with no change across T
  // species 3: H
  species[3] = 0.0;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    species[0] = +8.24944174e-04 - 1.62860306e-06 * T - 2.84263030e-10 * T2 +
                 1.65394890e-12 * T3;
    // species 1: O2
    species[1] = +1.12748635e-03 - 1.15123009e-06 * T + 3.94163169e-09 * T2 -
                 3.50742157e-12 * T3;
    // species 2: H2O
    species[2] = +3.47498246e-03 - 1.27093927e-05 * T + 2.09057438e-08 * T2 -
                 1.00263539e-11 * T3;
    // species 4: O
    species[4] = -1.63816649e-03 + 4.84206340e-06 * T - 4.80852957e-09 * T2 +
                 1.55627854e-12 * T3;
    // species 5: OH
    species[5] = -3.22544939e-03 + 1.30552938e-05 * T - 1.73956093e-08 * T2 +
                 8.24949516e-12 * T3;
    // species 6: HO2
    species[6] = -4.74912051e-03 + 4.23165782e-05 * T - 7.28291682e-08 * T2 +
                 3.71690050e-11 * T3;
    // species 7: H2O2
    species[7] = +6.56922581e-03 - 2.97002516e-07 * T - 1.38774166e-08 * T2 +
                 9.88605900e-12 * T3;
    // species 8: N2
    species[8] = +1.40824000e-03 - 7.92644400e-06 * T + 1.69245450e-08 * T2 -
                 9.77942000e-12 * T3;
  } else {
    // species 0: H2
    species[0] = +7.00064411e-04 - 1.12676574e-07 * T - 2.76947345e-11 * T2 +
                 6.33100716e-15 * T3;
    // species 1: O2
    species[1] = +6.13519689e-04 - 2.51768398e-07 * T + 5.32584444e-11 * T2 -
                 4.54574124e-15 * T3;
    // species 2: H2O
    species[2] = +3.05629289e-03 - 1.74605202e-06 * T + 3.60298917e-10 * T2 -
                 2.55664715e-14 * T3;
    // species 4: O
    species[4] = -2.75506191e-05 - 6.20560670e-09 * T + 1.36532023e-11 * T2 -
                 1.74722060e-15 * T3;
    // species 5: OH
    species[5] = +1.05650448e-03 - 5.18165516e-07 * T + 9.15656022e-11 * T2 -
                 5.32783504e-15 * T3;
    // species 6: HO2
    species[6] = +2.23982013e-03 - 1.26731630e-06 * T + 3.42739110e-10 * T2 -
                 4.31634140e-14 * T3;
    // species 7: H2O2
    species[7] = +4.33613639e-03 - 2.94937764e-06 * T + 7.04671071e-10 * T2 -
                 5.72661424e-14 * T3;
    // species 8: N2
    species[8] = +1.48797700e-03 - 1.13695220e-06 * T + 3.02911200e-10 * T2 -
                 2.70134040e-14 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[9]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 3: H
  result += y[3] * (+2.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result += y[0] *
              (+3.29812431e+00 + 8.24944174e-04 * T - 8.14301529e-07 * T2 -
               9.47543433e-11 * T3 + 4.13487224e-13 * T4) *
              0.4960317460317460;
    // species 1: O2
    result += y[1] *
              (+3.21293640e+00 + 1.12748635e-03 * T - 5.75615047e-07 * T2 +
               1.31387723e-09 * T3 - 8.76855392e-13 * T4) *
              0.0312519532470779;
    // species 2: H2O
    result += y[2] *
              (+3.38684249e+00 + 3.47498246e-03 * T - 6.35469633e-06 * T2 +
               6.96858127e-09 * T3 - 2.50658847e-12 * T4) *
              0.0555092978073827;
    // species 4: O
    result += y[4] *
              (+2.94642878e+00 - 1.63816649e-03 * T + 2.42103170e-06 * T2 -
               1.60284319e-09 * T3 + 3.89069636e-13 * T4) *
              0.0625039064941559;
    // species 5: OH
    result += y[5] *
              (+4.12530561e+00 - 3.22544939e-03 * T + 6.52764691e-06 * T2 -
               5.79853643e-09 * T3 + 2.06237379e-12 * T4) *
              0.0587993179279120;
    // species 6: HO2
    result += y[6] *
              (+4.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
               2.42763894e-08 * T3 + 9.29225124e-12 * T4) *
              0.0302975216627280;
    // species 7: H2O2
    result += y[7] *
              (+3.38875365e+00 + 6.56922581e-03 * T - 1.48501258e-07 * T2 -
               4.62580552e-09 * T3 + 2.47151475e-12 * T4) *
              0.0293996589639560;
    // species 8: N2
    result += y[8] *
              (+3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: H2
    result += y[0] *
              (+2.99142337e+00 + 7.00064411e-04 * T - 5.63382869e-08 * T2 -
               9.23157818e-12 * T3 + 1.58275179e-15 * T4) *
              0.4960317460317460;
    // species 1: O2
    result += y[1] *
              (+3.69757819e+00 + 6.13519689e-04 * T - 1.25884199e-07 * T2 +
               1.77528148e-11 * T3 - 1.13643531e-15 * T4) *
              0.0312519532470779;
    // species 2: H2O
    result += y[2] *
              (+2.67214561e+00 + 3.05629289e-03 * T - 8.73026011e-07 * T2 +
               1.20099639e-10 * T3 - 6.39161787e-15 * T4) *
              0.0555092978073827;
    // species 4: O
    result += y[4] *
              (+2.54205966e+00 - 2.75506191e-05 * T - 3.10280335e-09 * T2 +
               4.55106742e-12 * T3 - 4.36805150e-16 * T4) *
              0.0625039064941559;
    // species 5: OH
    result += y[5] *
              (+2.86472886e+00 + 1.05650448e-03 * T - 2.59082758e-07 * T2 +
               3.05218674e-11 * T3 - 1.33195876e-15 * T4) *
              0.0587993179279120;
    // species 6: HO2
    result += y[6] *
              (+4.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
               1.14246370e-10 * T3 - 1.07908535e-14 * T4) *
              0.0302975216627280;
    // species 7: H2O2
    result += y[7] *
              (+4.57316685e+00 + 4.33613639e-03 * T - 1.47468882e-06 * T2 +
               2.34890357e-10 * T3 - 1.43165356e-14 * T4) *
              0.0293996589639560;
    // species 8: N2
    result += y[8] *
              (+2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[9]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 3: H
  result += y[3] * (+1.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result += y[0] *
              (+2.29812431e+00 + 8.24944174e-04 * T - 8.14301529e-07 * T2 -
               9.47543433e-11 * T3 + 4.13487224e-13 * T4) *
              0.4960317460317460;
    // species 1: O2
    result += y[1] *
              (+2.21293640e+00 + 1.12748635e-03 * T - 5.75615047e-07 * T2 +
               1.31387723e-09 * T3 - 8.76855392e-13 * T4) *
              0.0312519532470779;
    // species 2: H2O
    result += y[2] *
              (+2.38684249e+00 + 3.47498246e-03 * T - 6.35469633e-06 * T2 +
               6.96858127e-09 * T3 - 2.50658847e-12 * T4) *
              0.0555092978073827;
    // species 4: O
    result += y[4] *
              (+1.94642878e+00 - 1.63816649e-03 * T + 2.42103170e-06 * T2 -
               1.60284319e-09 * T3 + 3.89069636e-13 * T4) *
              0.0625039064941559;
    // species 5: OH
    result += y[5] *
              (+3.12530561e+00 - 3.22544939e-03 * T + 6.52764691e-06 * T2 -
               5.79853643e-09 * T3 + 2.06237379e-12 * T4) *
              0.0587993179279120;
    // species 6: HO2
    result += y[6] *
              (+3.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
               2.42763894e-08 * T3 + 9.29225124e-12 * T4) *
              0.0302975216627280;
    // species 7: H2O2
    result += y[7] *
              (+2.38875365e+00 + 6.56922581e-03 * T - 1.48501258e-07 * T2 -
               4.62580552e-09 * T3 + 2.47151475e-12 * T4) *
              0.0293996589639560;
    // species 8: N2
    result += y[8] *
              (+2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: H2
    result += y[0] *
              (+1.99142337e+00 + 7.00064411e-04 * T - 5.63382869e-08 * T2 -
               9.23157818e-12 * T3 + 1.58275179e-15 * T4) *
              0.4960317460317460;
    // species 1: O2
    result += y[1] *
              (+2.69757819e+00 + 6.13519689e-04 * T - 1.25884199e-07 * T2 +
               1.77528148e-11 * T3 - 1.13643531e-15 * T4) *
              0.0312519532470779;
    // species 2: H2O
    result += y[2] *
              (+1.67214561e+00 + 3.05629289e-03 * T - 8.73026011e-07 * T2 +
               1.20099639e-10 * T3 - 6.39161787e-15 * T4) *
              0.0555092978073827;
    // species 4: O
    result += y[4] *
              (+1.54205966e+00 - 2.75506191e-05 * T - 3.10280335e-09 * T2 +
               4.55106742e-12 * T3 - 4.36805150e-16 * T4) *
              0.0625039064941559;
    // species 5: OH
    result += y[5] *
              (+1.86472886e+00 + 1.05650448e-03 * T - 2.59082758e-07 * T2 +
               3.05218674e-11 * T3 - 1.33195876e-15 * T4) *
              0.0587993179279120;
    // species 6: HO2
    result += y[6] *
              (+3.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
               1.14246370e-10 * T3 - 1.07908535e-14 * T4) *
              0.0302975216627280;
    // species 7: H2O2
    result += y[7] *
              (+3.57316685e+00 + 4.33613639e-03 * T - 1.47468882e-06 * T2 +
               2.34890357e-10 * T3 - 1.43165356e-14 * T4) *
              0.0293996589639560;
    // species 8: N2
    result += y[8] *
              (+1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[9];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 3: H
  result +=
    y[3] * (+2.50000000e+00 + 2.54716270e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result +=
      y[0] *
      (+3.29812431e+00 + 4.12472087e-04 * T - 2.71433843e-07 * T2 -
       2.36885858e-11 * T3 + 8.26974448e-14 * T4 - 1.01252087e+03 * invT) *
      0.4960317460317460;
    // species 1: O2
    result +=
      y[1] *
      (+3.21293640e+00 + 5.63743175e-04 * T - 1.91871682e-07 * T2 +
       3.28469308e-10 * T3 - 1.75371078e-13 * T4 - 1.00524902e+03 * invT) *
      0.0312519532470779;
    // species 2: H2O
    result +=
      y[2] *
      (+3.38684249e+00 + 1.73749123e-03 * T - 2.11823211e-06 * T2 +
       1.74214532e-09 * T3 - 5.01317694e-13 * T4 - 3.02081133e+04 * invT) *
      0.0555092978073827;
    // species 4: O
    result +=
      y[4] *
      (+2.94642878e+00 - 8.19083245e-04 * T + 8.07010567e-07 * T2 -
       4.00710797e-10 * T3 + 7.78139272e-14 * T4 + 2.91476445e+04 * invT) *
      0.0625039064941559;
    // species 5: OH
    result +=
      y[5] *
      (+4.12530561e+00 - 1.61272470e-03 * T + 2.17588230e-06 * T2 -
       1.44963411e-09 * T3 + 4.12474758e-13 * T4 + 3.34630913e+03 * invT) *
      0.0587993179279120;
    // species 6: HO2
    result +=
      y[6] *
      (+4.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
       6.06909735e-09 * T3 + 1.85845025e-12 * T4 + 2.94808040e+02 * invT) *
      0.0302975216627280;
    // species 7: H2O2
    result +=
      y[7] *
      (+3.38875365e+00 + 3.28461290e-03 * T - 4.95004193e-08 * T2 -
       1.15645138e-09 * T3 + 4.94302950e-13 * T4 - 1.76631465e+04 * invT) *
      0.0293996589639560;
    // species 8: N2
    result +=
      y[8] *
      (+3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 0: H2
    result +=
      y[0] *
      (+2.99142337e+00 + 3.50032206e-04 * T - 1.87794290e-08 * T2 -
       2.30789455e-12 * T3 + 3.16550358e-16 * T4 - 8.35033997e+02 * invT) *
      0.4960317460317460;
    // species 1: O2
    result +=
      y[1] *
      (+3.69757819e+00 + 3.06759845e-04 * T - 4.19613997e-08 * T2 +
       4.43820370e-12 * T3 - 2.27287062e-16 * T4 - 1.23393018e+03 * invT) *
      0.0312519532470779;
    // species 2: H2O
    result +=
      y[2] *
      (+2.67214561e+00 + 1.52814644e-03 * T - 2.91008670e-07 * T2 +
       3.00249098e-11 * T3 - 1.27832357e-15 * T4 - 2.98992090e+04 * invT) *
      0.0555092978073827;
    // species 4: O
    result +=
      y[4] *
      (+2.54205966e+00 - 1.37753096e-05 * T - 1.03426778e-09 * T2 +
       1.13776685e-12 * T3 - 8.73610300e-17 * T4 + 2.92308027e+04 * invT) *
      0.0625039064941559;
    // species 5: OH
    result +=
      y[5] *
      (+2.86472886e+00 + 5.28252240e-04 * T - 8.63609193e-08 * T2 +
       7.63046685e-12 * T3 - 2.66391752e-16 * T4 + 3.68362875e+03 * invT) *
      0.0587993179279120;
    // species 6: HO2
    result +=
      y[6] *
      (+4.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
       2.85615925e-11 * T3 - 2.15817070e-15 * T4 + 1.11856713e+02 * invT) *
      0.0302975216627280;
    // species 7: H2O2
    result +=
      y[7] *
      (+4.57316685e+00 + 2.16806820e-03 * T - 4.91562940e-07 * T2 +
       5.87225893e-11 * T3 - 2.86330712e-15 * T4 - 1.80069609e+04 * invT) *
      0.0293996589639560;
    // species 8: N2
    result +=
      y[8] *
      (+2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[9];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 3: H
  result +=
    y[3] * (+1.50000000e+00 + 2.54716270e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: H2
    result +=
      y[0] *
      (+2.29812431e+00 + 4.12472087e-04 * T - 2.71433843e-07 * T2 -
       2.36885858e-11 * T3 + 8.26974448e-14 * T4 - 1.01252087e+03 * invT) *
      0.4960317460317460;
    // species 1: O2
    result +=
      y[1] *
      (+2.21293640e+00 + 5.63743175e-04 * T - 1.91871682e-07 * T2 +
       3.28469308e-10 * T3 - 1.75371078e-13 * T4 - 1.00524902e+03 * invT) *
      0.0312519532470779;
    // species 2: H2O
    result +=
      y[2] *
      (+2.38684249e+00 + 1.73749123e-03 * T - 2.11823211e-06 * T2 +
       1.74214532e-09 * T3 - 5.01317694e-13 * T4 - 3.02081133e+04 * invT) *
      0.0555092978073827;
    // species 4: O
    result +=
      y[4] *
      (+1.94642878e+00 - 8.19083245e-04 * T + 8.07010567e-07 * T2 -
       4.00710797e-10 * T3 + 7.78139272e-14 * T4 + 2.91476445e+04 * invT) *
      0.0625039064941559;
    // species 5: OH
    result +=
      y[5] *
      (+3.12530561e+00 - 1.61272470e-03 * T + 2.17588230e-06 * T2 -
       1.44963411e-09 * T3 + 4.12474758e-13 * T4 + 3.34630913e+03 * invT) *
      0.0587993179279120;
    // species 6: HO2
    result +=
      y[6] *
      (+3.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
       6.06909735e-09 * T3 + 1.85845025e-12 * T4 + 2.94808040e+02 * invT) *
      0.0302975216627280;
    // species 7: H2O2
    result +=
      y[7] *
      (+2.38875365e+00 + 3.28461290e-03 * T - 4.95004193e-08 * T2 -
       1.15645138e-09 * T3 + 4.94302950e-13 * T4 - 1.76631465e+04 * invT) *
      0.0293996589639560;
    // species 8: N2
    result +=
      y[8] *
      (+2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 0: H2
    result +=
      y[0] *
      (+1.99142337e+00 + 3.50032206e-04 * T - 1.87794290e-08 * T2 -
       2.30789455e-12 * T3 + 3.16550358e-16 * T4 - 8.35033997e+02 * invT) *
      0.4960317460317460;
    // species 1: O2
    result +=
      y[1] *
      (+2.69757819e+00 + 3.06759845e-04 * T - 4.19613997e-08 * T2 +
       4.43820370e-12 * T3 - 2.27287062e-16 * T4 - 1.23393018e+03 * invT) *
      0.0312519532470779;
    // species 2: H2O
    result +=
      y[2] *
      (+1.67214561e+00 + 1.52814644e-03 * T - 2.91008670e-07 * T2 +
       3.00249098e-11 * T3 - 1.27832357e-15 * T4 - 2.98992090e+04 * invT) *
      0.0555092978073827;
    // species 4: O
    result +=
      y[4] *
      (+1.54205966e+00 - 1.37753096e-05 * T - 1.03426778e-09 * T2 +
       1.13776685e-12 * T3 - 8.73610300e-17 * T4 + 2.92308027e+04 * invT) *
      0.0625039064941559;
    // species 5: OH
    result +=
      y[5] *
      (+1.86472886e+00 + 5.28252240e-04 * T - 8.63609193e-08 * T2 +
       7.63046685e-12 * T3 - 2.66391752e-16 * T4 + 3.68362875e+03 * invT) *
      0.0587993179279120;
    // species 6: HO2
    result +=
      y[6] *
      (+3.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
       2.85615925e-11 * T3 - 2.15817070e-15 * T4 + 1.11856713e+02 * invT) *
      0.0302975216627280;
    // species 7: H2O2
    result +=
      y[7] *
      (+3.57316685e+00 + 2.16806820e-03 * T - 4.91562940e-07 * T2 +
       5.87225893e-11 * T3 - 2.86330712e-15 * T4 - 1.80069609e+04 * invT) *
      0.0293996589639560;
    // species 8: N2
    result +=
      y[8] *
      (+1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[9]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 9; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[9];  //  temporary storage
  amrex::Real x[9];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (2.016000 * YOW);
  x[1] = y[1] / (31.998000 * YOW);
  x[2] = y[2] / (18.015000 * YOW);
  x[3] = y[3] / (1.008000 * YOW);
  x[4] = y[4] / (15.999000 * YOW);
  x[5] = y[5] / (17.007000 * YOW);
  x[6] = y[6] / (33.006000 * YOW);
  x[7] = y[7] / (34.014000 * YOW);
  x[8] = y[8] / (28.014000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 9; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 2.016000;                   // H2
  XW += x[1] * 31.998000;                  // O2
  XW += x[2] * 18.015000;                  // H2O
  XW += x[3] * 1.008000;                   // H
  XW += x[4] * 15.999000;                  // O
  XW += x[5] * 17.007000;                  // OH
  XW += x[6] * 33.006000;                  // HO2
  XW += x[7] * 34.014000;                  // H2O2
  XW += x[8] * 28.014000;                  // N2
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 2.016000;  // H2
  W += c[1] * 31.998000; // O2
  W += c[2] * 18.015000; // H2O
  W += c[3] * 1.008000;  // H
  W += c[4] * 15.999000; // O
  W += c[5] * 17.007000; // OH
  W += c[6] * 33.006000; // HO2
  W += c[7] * 34.014000; // H2O2
  W += c[8] * 28.014000; // N2

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 2.016000;                     // H2
  XW += x[1] * 31.998000;                    // O2
  XW += x[2] * 18.015000;                    // H2O
  XW += x[3] * 1.008000;                     // H
  XW += x[4] * 15.999000;                    // O
  XW += x[5] * 17.007000;                    // OH
  XW += x[6] * 33.006000;                    // HO2
  XW += x[7] * 34.014000;                    // H2O2
  XW += x[8] * 28.014000;                    // N2
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 2.016000;  // H2
  W += c[1] * 31.998000; // O2
  W += c[2] * 18.015000; // H2O
  W += c[3] * 1.008000;  // H
  W += c[4] * 15.999000; // O
  W += c[5] * 17.007000; // OH
  W += c[6] * 33.006000; // HO2
  W += c[7] * 34.014000; // H2O2
  W += c[8] * 28.014000; // N2

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;     //  see Eq 4 in CK Manual
  XW += x[0] * 2.016000;  // H2
  XW += x[1] * 31.998000; // O2
  XW += x[2] * 18.015000; // H2O
  XW += x[3] * 1.008000;  // H
  XW += x[4] * 15.999000; // O
  XW += x[5] * 17.007000; // OH
  XW += x[6] * 33.006000; // HO2
  XW += x[7] * 34.014000; // H2O2
  XW += x[8] * 28.014000; // N2
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 2.016000;  // H2
  W += c[1] * 31.998000; // O2
  W += c[2] * 18.015000; // H2O
  W += c[3] * 1.008000;  // H
  W += c[4] * 15.999000; // O
  W += c[5] * 17.007000; // OH
  W += c[6] * 33.006000; // HO2
  W += c[7] * 34.014000; // H2O2
  W += c[8] * 28.014000; // N2

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 9; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 9; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 9; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 9; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 2.016000;  // H2
  XW += x[1] * 31.998000; // O2
  XW += x[2] * 18.015000; // H2O
  XW += x[3] * 1.008000;  // H
  XW += x[4] * 15.999000; // O
  XW += x[5] * 17.007000; // OH
  XW += x[6] * 33.006000; // HO2
  XW += x[7] * 34.014000; // H2O2
  XW += x[8] * 28.014000; // N2
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 2.016000 * XWinv;
  y[1] = x[1] * 31.998000 * XWinv;
  y[2] = x[2] * 18.015000 * XWinv;
  y[3] = x[3] * 1.008000 * XWinv;
  y[4] = x[4] * 15.999000 * XWinv;
  y[5] = x[5] * 17.007000 * XWinv;
  y[6] = x[6] * 33.006000 * XWinv;
  y[7] = x[7] * 34.014000 * XWinv;
  y[8] = x[8] * 28.014000 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 2.016000;  // H2
  XW += x[1] * 31.998000; // O2
  XW += x[2] * 18.015000; // H2O
  XW += x[3] * 1.008000;  // H
  XW += x[4] * 15.999000; // O
  XW += x[5] * 17.007000; // OH
  XW += x[6] * 33.006000; // HO2
  XW += x[7] * 34.014000; // H2O2
  XW += x[8] * 28.014000; // N2
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 9; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 2.016000;  // H2
  CW += c[1] * 31.998000; // O2
  CW += c[2] * 18.015000; // H2O
  CW += c[3] * 1.008000;  // H
  CW += c[4] * 15.999000; // O
  CW += c[5] * 17.007000; // OH
  CW += c[6] * 33.006000; // HO2
  CW += c[7] * 34.014000; // H2O2
  CW += c[8] * 28.014000; // N2
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 2.016000 * CWinv;
  y[1] = c[1] * 31.998000 * CWinv;
  y[2] = c[2] * 18.015000 * CWinv;
  y[3] = c[3] * 1.008000 * CWinv;
  y[4] = c[4] * 15.999000 * CWinv;
  y[5] = c[5] * 17.007000 * CWinv;
  y[6] = c[6] * 33.006000 * CWinv;
  y[7] = c[7] * 34.014000 * CWinv;
  y[8] = c[8] * 28.014000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 4.124237409798234e+07; // H2
  cvms[1] *= 2.598431970171023e+06; // O2
  cvms[2] *= 4.615299815794193e+06; // H2O
  cvms[3] *= 8.248474819596468e+07; // H
  cvms[4] *= 5.196863940342046e+06; // O
  cvms[5] *= 4.888847308845322e+06; // OH
  cvms[6] *= 2.519076112874398e+06; // HO2
  cvms[7] *= 2.444423654422661e+06; // H2O2
  cvms[8] *= 2.967966951578939e+06; // N2
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 4.124237409798234e+07; // H2
  cpms[1] *= 2.598431970171023e+06; // O2
  cpms[2] *= 4.615299815794193e+06; // H2O
  cpms[3] *= 8.248474819596468e+07; // H
  cpms[4] *= 5.196863940342046e+06; // O
  cpms[5] *= 4.888847308845322e+06; // OH
  cpms[6] *= 2.519076112874398e+06; // HO2
  cpms[7] *= 2.444423654422661e+06; // H2O2
  cpms[8] *= 2.967966951578939e+06; // N2
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 9; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 9; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 4.124237409798234e+07; // H2
  sms[1] *= 2.598431970171023e+06; // O2
  sms[2] *= 4.615299815794193e+06; // H2O
  sms[3] *= 8.248474819596468e+07; // H
  sms[4] *= 5.196863940342046e+06; // O
  sms[5] *= 4.888847308845322e+06; // OH
  sms[6] *= 2.519076112874398e+06; // HO2
  sms[7] *= 2.444423654422661e+06; // H2O2
  sms[8] *= 2.967966951578939e+06; // N2
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 8: H + O2 (+M) <=> HO2 (+M)
  qf[0] = sc[1] * sc[3];
  qr[0] = sc[6];

  // reaction 15: H2O2 (+M) <=> 2 OH (+M)
  qf[1] = sc[7];
  qr[1] = (sc[5] * sc[5]);

  // reaction 4: H2 + M <=> 2 H + M
  qf[2] = sc[0];
  qr[2] = (sc[3] * sc[3]);

  // reaction 5: 2 O + M <=> O2 + M
  qf[3] = (sc[4] * sc[4]);
  qr[3] = sc[1];

  // reaction 6: H + O + M <=> OH + M
  qf[4] = sc[3] * sc[4];
  qr[4] = sc[5];

  // reaction 7: H + OH + M <=> H2O + M
  qf[5] = sc[3] * sc[5];
  qr[5] = sc[2];

  // reaction 0: H + O2 <=> O + OH
  qf[6] = sc[1] * sc[3];
  qr[6] = sc[4] * sc[5];

  // reaction 1: H2 + O <=> H + OH
  qf[7] = sc[0] * sc[4];
  qr[7] = sc[3] * sc[5];

  // reaction 2: H2 + OH <=> H + H2O
  qf[8] = sc[0] * sc[5];
  qr[8] = sc[2] * sc[3];

  // reaction 3: H2O + O <=> 2 OH
  qf[9] = sc[2] * sc[4];
  qr[9] = (sc[5] * sc[5]);

  // reaction 9: H + HO2 <=> H2 + O2
  qf[10] = sc[3] * sc[6];
  qr[10] = sc[0] * sc[1];

  // reaction 10: H + HO2 <=> 2 OH
  qf[11] = sc[3] * sc[6];
  qr[11] = (sc[5] * sc[5]);

  // reaction 11: HO2 + O <=> O2 + OH
  qf[12] = sc[4] * sc[6];
  qr[12] = sc[1] * sc[5];

  // reaction 12: HO2 + OH <=> H2O + O2
  qf[13] = sc[5] * sc[6];
  qr[13] = sc[1] * sc[2];

  // reaction 13: 2 HO2 <=> H2O2 + O2
  qf[14] = (sc[6] * sc[6]);
  qr[14] = sc[1] * sc[7];

  // reaction 14: 2 HO2 <=> H2O2 + O2
  qf[15] = (sc[6] * sc[6]);
  qr[15] = sc[1] * sc[7];

  // reaction 16: H + H2O2 <=> H2O + OH
  qf[16] = sc[3] * sc[7];
  qr[16] = sc[2] * sc[5];

  // reaction 17: H + H2O2 <=> H2 + HO2
  qf[17] = sc[3] * sc[7];
  qr[17] = sc[0] * sc[6];

  // reaction 18: H2O2 + O <=> HO2 + OH
  qf[18] = sc[4] * sc[7];
  qr[18] = sc[5] * sc[6];

  // reaction 19: H2O2 + OH <=> H2O + HO2
  qf[19] = sc[5] * sc[7];
  qr[19] = sc[2] * sc[6];

  // reaction 20: H2O2 + OH <=> H2O + HO2
  qf[20] = sc[5] * sc[7];
  qr[20] = sc[2] * sc[6];

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;
  amrex::Real redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;

  // reaction 0:  H + O2 <=> O + OH
  k_f = 3547000000 * exp((-0.406) * logT - (8352.89343569254) * invT);
  qf[6] *= k_f;
  qr[6] *= k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[4] - g_RT[5]));
  // reaction 1:  H2 + O <=> H + OH
  k_f = 0.0508 * exp((2.67) * logT - (3165.23282791169) * invT);
  qf[7] *= k_f;
  qr[7] *= k_f * exp(-(g_RT[0] - g_RT[3] + g_RT[4] - g_RT[5]));
  // reaction 2:  H2 + OH <=> H + H2O
  k_f = 216 * exp((1.51) * logT - (1726.03316371019) * invT);
  qf[8] *= k_f;
  qr[8] *= k_f * exp(-(g_RT[0] - g_RT[2] - g_RT[3] + g_RT[5]));
  // reaction 3:  H2O + O <=> 2 OH
  k_f = 2.97 * exp((2.02) * logT - (6743.10332178324) * invT);
  qf[9] *= k_f;
  qr[9] *= k_f * exp(-(g_RT[2] + g_RT[4] - 2.000000 * g_RT[5]));
  // reaction 4:  H2 + M <=> 2 H + M
  k_f = 45770000000000 * exp((-1.4) * logT - (52525.7555766966) * invT);
  Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[3])) * (refCinv);
  // reaction 5:  2 O + M <=> O2 + M
  k_f = 6165 * exp((-0.5) * logT);
  Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
  qf[3] *= Corr * k_f;
  qr[3] *= Corr * k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[4])) * (refC);
  // reaction 6:  H + O + M <=> OH + M
  k_f = 4714000 * exp((-1) * logT);
  Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
  qf[4] *= Corr * k_f;
  qr[4] *= Corr * k_f * exp(-(g_RT[3] + g_RT[4] - g_RT[5])) * (refC);
  // reaction 7:  H + OH + M <=> H2O + M
  k_f = 38000000000 * exp((-2) * logT);
  Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
  qf[5] *= Corr * k_f;
  qr[5] *= Corr * k_f * exp(-(-g_RT[2] + g_RT[3] + g_RT[5])) * (refC);
  // reaction 8:  H + O2 (+M) <=> HO2 (+M)
  k_f = 1475000 * exp((0.6) * logT);
  Corr = mixture + sc[0] + (-0.22) * sc[1] + (10) * sc[2];
  redP = Corr / k_f * 636600000 * exp(-1.72 * logT - (264.088106214317) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10((0.2) * exp(-T * 1e+30) + 0.8 * exp(-T * 1e-30) + 0.0);
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[6])) * (refC);
  // reaction 9:  H + HO2 <=> H2 + O2
  k_f = 16600000 * exp(-(414.147315957284) * invT);
  qf[10] *= k_f;
  qr[10] *= k_f * exp(-(-g_RT[0] - g_RT[1] + g_RT[3] + g_RT[6]));
  // reaction 10:  H + HO2 <=> 2 OH
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  qf[11] *= k_f;
  qr[11] *= k_f * exp(-(g_RT[3] - 2.000000 * g_RT[5] + g_RT[6]));
  // reaction 11:  HO2 + O <=> O2 + OH
  k_f = 32500000;
  qf[12] *= k_f;
  qr[12] *= k_f * exp(-(-g_RT[1] + g_RT[4] - g_RT[5] + g_RT[6]));
  // reaction 12:  HO2 + OH <=> H2O + O2
  k_f = 28900000 * exp(-(-250.098682904946) * invT);
  qf[13] *= k_f;
  qr[13] *= k_f * exp(-(-g_RT[1] - g_RT[2] + g_RT[5] + g_RT[6]));
  // reaction 13:  2 HO2 <=> H2O2 + O2
  k_f = 420000000 * exp(-(6029.54208967215) * invT);
  qf[14] *= k_f;
  qr[14] *= k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[6] - g_RT[7]));
  // reaction 14:  2 HO2 <=> H2O2 + O2
  k_f = 130000 * exp(-(-819.89091359563) * invT);
  qf[15] *= k_f;
  qr[15] *= k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[6] - g_RT[7]));
  // reaction 15:  H2O2 (+M) <=> 2 OH (+M)
  k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
  Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
  redP = Corr / k_f * 120200000000 * exp(-(22896.3582941147) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10((0.5) * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + 0.0);
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f * exp(-(-2.000000 * g_RT[5] + g_RT[7])) * (refCinv);
  // reaction 16:  H + H2O2 <=> H2O + OH
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  qf[16] *= k_f;
  qr[16] *= k_f * exp(-(-g_RT[2] + g_RT[3] - g_RT[5] + g_RT[7]));
  // reaction 17:  H + H2O2 <=> H2 + HO2
  k_f = 48200000 * exp(-(4000.57249314752) * invT);
  qf[17] *= k_f;
  qr[17] *= k_f * exp(-(-g_RT[0] + g_RT[3] - g_RT[6] + g_RT[7]));
  // reaction 18:  H2O2 + O <=> HO2 + OH
  k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
  qf[18] *= k_f;
  qr[18] *= k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[6] + g_RT[7]));
  // reaction 19:  H2O2 + OH <=> H2O + HO2
  k_f = 1000000;
  qf[19] *= k_f;
  qr[19] *= k_f * exp(-(-g_RT[2] + g_RT[5] - g_RT[6] + g_RT[7]));
  // reaction 20:  H2O2 + OH <=> H2O + HO2
  k_f = 580000000 * exp(-(4809.24167509571) * invT);
  qf[20] *= k_f;
  qr[20] *= k_f * exp(-(-g_RT[2] + g_RT[5] - g_RT[6] + g_RT[7]));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 9; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  {
    // reaction 8:  H + O2 (+M) <=> HO2 (+M)
    const amrex::Real k_f = 1475000 * exp((0.6) * logT);
    amrex::Real Corr = mixture + sc[0] + (-0.22) * sc[1] + (10) * sc[2];
    const amrex::Real redP =
      Corr / k_f * 636600000 * exp(-1.72 * logT - 264.088106214317 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(0.2 * exp(-T * 1e+30) + 0.8 * exp(-T * 1e-30) + 0.0);
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[1] * sc[3]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[6])) * (refC) * (sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 15:  H2O2 (+M) <=> 2 OH (+M)
    const amrex::Real k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
    amrex::Real Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
    const amrex::Real redP =
      Corr / k_f * 120200000000 * exp(-22896.3582941147 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent =
      log10(0.5 * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + 0.0);
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[7]);
    const amrex::Real qr = Corr * k_f * exp(-(-2.000000 * g_RT[5] + g_RT[7])) *
                           (refCinv) * ((sc[5] * sc[5]));
    const amrex::Real qdot = qf - qr;
    wdot[5] += 2.000000 * qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 4:  H2 + M <=> 2 H + M
    const amrex::Real k_f =
      45770000000000 * exp((-1.4) * logT - (52525.7555766966) * invT);
    const amrex::Real Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
    const amrex::Real qf = Corr * k_f * (sc[0]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[3])) *
                           (refCinv) * ((sc[3] * sc[3]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] += 2.000000 * qdot;
  }

  {
    // reaction 5:  2 O + M <=> O2 + M
    const amrex::Real k_f = 6165 * exp((-0.5) * logT);
    const amrex::Real Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
    const amrex::Real qf = Corr * k_f * ((sc[4] * sc[4]));
    const amrex::Real qr =
      Corr * k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[4])) * (refC) * (sc[1]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[4] -= 2.000000 * qdot;
  }

  {
    // reaction 6:  H + O + M <=> OH + M
    const amrex::Real k_f = 4714000 * exp((-1) * logT);
    const amrex::Real Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
    const amrex::Real qf = Corr * k_f * (sc[3] * sc[4]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[3] + g_RT[4] - g_RT[5])) * (refC) * (sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 7:  H + OH + M <=> H2O + M
    const amrex::Real k_f = 38000000000 * exp((-2) * logT);
    const amrex::Real Corr = mixture + (1.5) * sc[0] + (11) * sc[2];
    const amrex::Real qf = Corr * k_f * (sc[3] * sc[5]);
    const amrex::Real qr =
      Corr * k_f * exp(-(-g_RT[2] + g_RT[3] + g_RT[5])) * (refC) * (sc[2]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 0:  H + O2 <=> O + OH
    const amrex::Real k_f =
      3547000000 * exp((-0.406) * logT - (8352.89343569254) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] + g_RT[3] - g_RT[4] - g_RT[5])) * (sc[4] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 1:  H2 + O <=> H + OH
    const amrex::Real k_f =
      0.0508 * exp((2.67) * logT - (3165.23282791169) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[3] + g_RT[4] - g_RT[5])) * (sc[3] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 2:  H2 + OH <=> H + H2O
    const amrex::Real k_f =
      216 * exp((1.51) * logT - (1726.03316371019) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[2] - g_RT[3] + g_RT[5])) * (sc[2] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] += qdot;
    wdot[3] += qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 3:  H2O + O <=> 2 OH
    const amrex::Real k_f =
      2.97 * exp((2.02) * logT - (6743.10332178324) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] + g_RT[4] - 2.000000 * g_RT[5])) * ((sc[5] * sc[5]));
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += 2.000000 * qdot;
  }

  {
    // reaction 9:  H + HO2 <=> H2 + O2
    const amrex::Real k_f = 16600000 * exp(-(414.147315957284) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] - g_RT[1] + g_RT[3] + g_RT[6])) * (sc[0] * sc[1]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 10:  H + HO2 <=> 2 OH
    const amrex::Real k_f = 70790000 * exp(-(148.448916412392) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[3] - 2.000000 * g_RT[5] + g_RT[6])) * ((sc[5] * sc[5]));
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[5] += 2.000000 * qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 11:  HO2 + O <=> O2 + OH
    const amrex::Real k_f = 32500000;
    const amrex::Real qf = k_f * (sc[4] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + g_RT[4] - g_RT[5] + g_RT[6])) * (sc[1] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 12:  HO2 + OH <=> H2O + O2
    const amrex::Real k_f = 28900000 * exp(-(-250.098682904946) * invT);
    const amrex::Real qf = k_f * (sc[5] * sc[6]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] - g_RT[2] + g_RT[5] + g_RT[6])) * (sc[1] * sc[2]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 13:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 420000000 * exp(-(6029.54208967215) * invT);
    const amrex::Real qf = k_f * ((sc[6] * sc[6]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[6] - g_RT[7])) * (sc[1] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[6] -= 2.000000 * qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 14:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 130000 * exp(-(-819.89091359563) * invT);
    const amrex::Real qf = k_f * ((sc[6] * sc[6]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[1] + 2.000000 * g_RT[6] - g_RT[7])) * (sc[1] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[6] -= 2.000000 * qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 16:  H + H2O2 <=> H2O + OH
    const amrex::Real k_f = 24100000 * exp(-(1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[3] - g_RT[5] + g_RT[7])) * (sc[2] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[5] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 17:  H + H2O2 <=> H2 + HO2
    const amrex::Real k_f = 48200000 * exp(-(4000.57249314752) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[3] - g_RT[6] + g_RT[7])) * (sc[0] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 18:  H2O2 + O <=> HO2 + OH
    const amrex::Real k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[6] + g_RT[7])) * (sc[5] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 19:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f = 1000000;
    const amrex::Real qf = k_f * (sc[5] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[5] - g_RT[6] + g_RT[7])) * (sc[2] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 20:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f = 580000000 * exp(-(4809.24167509571) * invT);
    const amrex::Real qf = k_f * (sc[5] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[2] + g_RT[5] - g_RT[6] + g_RT[7])) * (sc[2] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }
}

#endif
