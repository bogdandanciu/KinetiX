// Chemistry
__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_species_rates(const cfloat Ci[], cfloat rates[]);
__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_enthalpy_RT(cfloat h_RT[]);
__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_molar_heat_capacity_R(cfloat cp_R[]);
// Transport
__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivityNIVT(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]);
__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosityIVT(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]);
__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity(unsigned int id, cfloat mean_molar_mass_VTN, cfloat VT, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[], dfloat* out, unsigned int stride);

