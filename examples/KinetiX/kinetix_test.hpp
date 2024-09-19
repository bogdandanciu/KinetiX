// Chemistry
__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_species_rates(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, const cfloat rcpT, const cfloat* Ci, cfloat* wdot);
__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_enthalpy_RT(const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, const cfloat rcpT, cfloat* h_RT);
__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_molar_heat_capacity_R(const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, cfloat* cp_R);
// Transport
__KINETIX_DEVICE__ __KINETIX_INLINE__ cfloat kinetix_conductivity(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]);
__KINETIX_DEVICE__ __KINETIX_INLINE__ cfloat kinetix_viscosity(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]);
__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_diffusivity(cfloat Mbar, cfloat p, cfloat TsqrT, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[], cfloat* Dkm);

