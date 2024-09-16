"""
Module containing routines for computing mixture-averaged transport properties.
"""

# Third-party library imports
from numpy import (
    dot,
    linspace,
    log as ln,
    pi,
    sqrt,
    square as sq
)

# Local imports
import constants as const
from utils.general_utils import (
    cube,
    f,
    polynomial_regression
)
from utils.write_utils import (
    CodeGenerator,
    write_const_expression
)


class NASA7:
    """
    NASA7 class.

    Represents a NASA7 polynomial model for calculating thermodynamic
    properties of a chemical species.
    """

    def __init__(self, pieces, temp_split):
        self.pieces = pieces
        self.temp_split = temp_split

    def piece(self, T):
        return self.pieces[0 if T < self.temp_split else 1]

    def molar_heat_capacity_R(self, T):
        a = self.piece(T)
        return a[0] + a[1] * T + a[2] * T * T + a[3] * T * T * T + a[4] * T * T * T * T


class Species:
    """
    Species class.

    Represents chemical species with associated thermochemical and
    transport properties.
    """

    def __init__(self, rcp_diffcoeffs, sp_names, molar_masses, thermodynamics,
                 temperature_ranges, well_depth, dipole_moment, diameter,
                 rotational_relaxation, degrees_of_freedom, polarizability, sp_len):
        self.rcp_diffcoeffs = rcp_diffcoeffs
        self.sp_names = sp_names
        self.Mi = molar_masses
        self.thermo = thermodynamics
        self._temperature_ranges = temperature_ranges
        self._eps = well_depth
        self._mu = dipole_moment
        self._sigma = diameter
        self._rot = rotational_relaxation
        self._dof = degrees_of_freedom
        self._alpha = polarizability
        self._sp_len = sp_len
        self._header_lnT_star = ln(const.header_T_star)

    def _T_star(self, j, k, T):
        return T * const.kB / self._epsilon(j, k)

    def _epsilon(self, j, k):
        eps = self._eps
        return sqrt(eps[j] * eps[k]) * sq(self._xi(j, k))

    def _xi(self, j, k):
        (mu, alpha, sigma, eps) = (self._mu, self._alpha, self._sigma, self._eps)
        if (mu[j] > 0.) == (mu[k] > 0.):
            return 1.
        (p, np) = (j, k) if mu[j] != 0. else (k, j)
        return (1. + 1. / 4. * alpha[np] / cube(sigma[np]) *
                sq(mu[p] / sqrt(4. * pi * const.epsilon0 * eps[p] * cube(sigma[p]))) * sqrt(eps[p] / eps[np]))

    def _rM(self, j, k):
        Mi = self.Mi
        return Mi[j] / const.NA * Mi[k] / const.NA / (Mi[j] / const.NA + Mi[k] / const.NA)

    def _sigma_star(self, j, k):
        sigma = self._sigma
        return (sigma[j] + sigma[k]) / 2. * pow(self._xi(j, k), -1. / 6.)

    def _delta_star(self, j, k):
        (eps, mu, sigma) = (self._eps, self._mu, self._sigma)
        return (0.5 * mu[j] * mu[k] / (4. * pi * const.epsilon0 * sqrt(eps[j] * eps[k]) *
                                       cube((sigma[j] + sigma[k]) / 2.)))

    def _collision_integral(self, I0, table, fit, j, k, T):
        lnT_star = ln(self._T_star(j, k, T))
        header_lnT_star = self._header_lnT_star[1:-1]
        # Find the first index where lnT_star is less than the corresponding value in header_lnT_star
        start = 0
        for i, val in enumerate(header_lnT_star):
            if lnT_star < val:
                start = i
                break
        interp_start_index = max(start-1, 0)
        if interp_start_index + 3 > len(header_lnT_star)-1:
            interp_start_index = len(header_lnT_star)-4
        header_lnT_star_slice = header_lnT_star[interp_start_index:][:3]
        assert (len(header_lnT_star_slice) == 3)
        polynomials = fit[interp_start_index + I0:][:3]
        assert (len(polynomials) == 3)

        def _evaluate_polynomial(P, x):
            return dot(P, [pow(x, k) for k in range(len(P))])

        def _quadratic_interpolation(x, y, x0):
            L0 = ((x0 - x[1]) * (x0 - x[2])) / ((x[0] - x[1]) * (x[0] - x[2]))
            L1 = ((x0 - x[0]) * (x0 - x[2])) / ((x[1] - x[0]) * (x[1] - x[2]))
            L2 = ((x0 - x[0]) * (x0 - x[1])) / ((x[2] - x[0]) * (x[2] - x[1]))

            return L0 * y[0] + L1 * y[1] + L2 * y[2]

        delta_star = self._delta_star(j, k)
        for P in polynomials:
            assert len(P) == 7, len(P)
        table = table[interp_start_index + I0:][:3]
        assert (len(table) == 3)
        # P[:6]: Reproduces Cantera truncated polynomial mistake
        if delta_star == 0.0:
            y = [row[0] for row in table]
        else:
            y = [_evaluate_polynomial(P[:6], delta_star) for P in polynomials]
        return _quadratic_interpolation(header_lnT_star_slice, y, lnT_star)

    def _omega_star_22(self, j, k, T):
        return self._collision_integral(0, const.collision_integrals_Omega_star_22, const.Omega_star_22, j, k, T)

    def _a_star(self, j, k, T):
        return self._collision_integral(1, const.collision_integrals_A_star, const.A_star, j, k, T)

    def _b_star(self, j, k, T):
        return self._collision_integral(1, const.collision_integrals_B_star, const.B_star, j, k, T)

    def _c_star(self, j, k, T):
        return self._collision_integral(1, const.collision_integrals_C_star, const.C_star, j, k, T)

    def _omega_star_11(self, j, k, T):
        return self._omega_star_22(j, k, T) / self._a_star(j, k, T)

    def _viscosity(self, k, T):
        (Mi, sigma) = (self.Mi, self._sigma)
        return (5. / 16. * sqrt(pi * Mi[k] / const.NA * const.kB * T) /
                (pi * sq(sigma[k]) * self._omega_star_22(k, k, T)))

    # p*Dkj
    def _diffusivity(self, j, k, T):
        return (3. / 16. * sqrt(2. * pi / self._rM(j, k)) * pow(const.kB * T, 3. / 2.) /
                (pi * sq(self._sigma_star(j, k)) * self._omega_star_11(j, k, T)))

    def _conductivity(self, k, T):
        (Mi, thermo, eps, rot, dof) = (self.Mi, self.thermo, self._eps, self._rot, self._dof)
        f_vib = (Mi[k] / const.NA / (const.kB * T) * self._diffusivity(k, k, T) / self._viscosity(k, T))
        T_star = self._T_star(k, k, T)

        def _F(T_star):
            return (1. + pow(pi, 3. / 2.) / sqrt(T_star) *
                    (1. / 2. + 1. / T_star) + (1. / 4. * sq(pi) + 2.) / T_star)

        A = 5. / 2. - f_vib
        B = (rot[k] * _F(298. * const.kB / eps[k]) / _F(T_star) + 2. / pi * (5. / 3. * dof[k] + f_vib))
        f_rot = f_vib * (1. + 2. / pi * A / B)
        f_trans = 5. / 2. * (1. - 2. / pi * A / B * dof[k] / (3. / 2.))
        Cv = (thermo[k].molar_heat_capacity_R(T) - 5. / 2. - dof[k])
        return ((self._viscosity(k, T) / (Mi[k] / const.NA)) * const.kB *
                (f_trans * 3. / 2. + f_rot * dof[k] + f_vib * Cv))

    def transport_polynomials(self):
        T_min = max(sp_temp_rng[0] for sp_temp_rng in self._temperature_ranges)
        T_max = min(sp_temp_rng[-1] for sp_temp_rng in self._temperature_ranges)
        nump = 50
        T_rng = linspace(T_min, T_max, nump)

        class TransportPolynomials:
            pass

        transport_polynomials = TransportPolynomials()
        transport_polynomials.conductivity = [
            polynomial_regression(ln(T_rng), [self._conductivity(k, T) / sqrt(T) for T in T_rng])
            for k in range(self._sp_len)]
        transport_polynomials.viscosity = [
            polynomial_regression(ln(T_rng), [sqrt(self._viscosity(k, T) / sqrt(T)) for T in T_rng])
            for k in range(self._sp_len)]
        if self.rcp_diffcoeffs:
            # Evaluate the reciprocal polynomial to avoid expensive divisions during runtime evaluation
            transport_polynomials.diffusivity = [
                [polynomial_regression(ln(T_rng), [((T * sqrt(T)) / self._diffusivity(j, k, T)) for T in T_rng])
                 for k in range(self._sp_len)] for j in range(self._sp_len)]
        else:
            transport_polynomials.diffusivity = [
                [polynomial_regression(ln(T_rng), [(self._diffusivity(j, k, T)) / (T * sqrt(T)) for T in T_rng])
                 for k in range(self._sp_len)] for j in range(self._sp_len)]
        return transport_polynomials
        
    
def get_species_from_model(species, rcp_diffcoeffs):
    """
    Extract species information and attributes from the mechanism model.
    """

    p = lambda f: list(map(f, species))
    sp_names = p(lambda s: s['name'])
    sp_len = len(sp_names)
    molar_masses = p(lambda s: sum(
        [element[1] * const.standard_atomic_weights[element[0]] / 1e3
         for element in s['composition'].items()]))

    def from_model(s):
        temperature_split = s['temperature-ranges'][1]
        pieces = s['data']
        if len(pieces) < 2:
            # Add a dummy set of thermo coefficients and temperature split of 1000K
            # in case the specie has only one set
            temperature_split = 1000
            pieces.append(pieces[0].copy())
        nasa7 = NASA7(pieces, temperature_split)
        return nasa7

    thermodynamics = p(lambda s: from_model(s['thermo']))
    # Check if species have more than two sets of thermodynamic coefficients
    for idx, specie in enumerate(thermodynamics):
        if len(specie.pieces) > 2:
            raise SystemExit(
                f'Specie {sp_names[idx]} has more than 2 sets of thermodynamic coefficients. '
                f'This is not currently supported!')
    temperature_ranges = p(lambda s: s['thermo']['temperature-ranges'])
    degrees_of_freedom = p(lambda s: {'atom': 0, 'linear': 1, 'nonlinear': 3 / 2}[s['transport']['geometry']])
    well_depth = p(lambda s: s['transport']['well-depth'] * const.kB)
    diameter = p(lambda s: s['transport']['diameter'] * 1e-10)  # Å
    dipole_moment = p(lambda s: s['transport'].get('dipole', 0) * const.Cm_per_Debye)
    polarizability = p(lambda s: s['transport'].get('polarizability', 0) * 1e-30)  # Å³
    rotational_relaxation = p(lambda s: float(s['transport'].get('rotational-relaxation', 0)))

    species = Species(rcp_diffcoeffs, sp_names, molar_masses, thermodynamics,
                      temperature_ranges, well_depth, dipole_moment, diameter,
                      rotational_relaxation, degrees_of_freedom, polarizability, sp_len)
    return species


def evaluate_polynomial(P):
    """
    Create a string representation of the polynomial evaluation.
    """
    return f'{f(P[0])}+{f(P[1])}*lnT+{f(P[2])}*lnT2+{f(P[3])}*lnT3+{f(P[4])}*lnT4'


def write_file_conductivity_roll(file_name, output_dir, align_width, target, transport_polynomials, sp_len):
    """
    Write the 'conductivity.inc' file with rolled loop specification.
    """

    b0, b1, b2, b3, b4 = [], [], [], [], []
    for cond_coeff in transport_polynomials.conductivity:
        b0.append(cond_coeff[0])
        b1.append(cond_coeff[1])
        b2.append(cond_coeff[2])
        b3.append(cond_coeff[3])
        b4.append(cond_coeff[4])
    var_str = ['b0', 'b1', 'b2', 'b3', 'b4']
    var = [b0, b1, b2, b3, b4]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivity"
                f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[])")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"cfloat lambda_k, sum1=0., sum2=0.;", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"lambda_k = b0[k] + b1[k]*lnT + b2[k]*lnT2 + b3[k]*lnT3 + b4[k]*lnT4;", 2)
    cg.add_line(f"sum1 += Xi[k]*lambda_k;", 2)
    cg.add_line(f"sum2 += Xi[k]/lambda_k;", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"return {f(0.5)}*(sum1 + {f(1.)}/sum2);", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_viscosity_roll(file_name, output_dir, align_width, target, transport_polynomials, sp_len, Mi):
    """
    Write the 'viscosity.inc' file with rolled loop specification.
    """
    a0, a1, a2, a3, a4 = [], [], [], [], []
    for vis_coeff in transport_polynomials.viscosity:
        a0.append(vis_coeff[0])
        a1.append(vis_coeff[1])
        a2.append(vis_coeff[2])
        a3.append(vis_coeff[3])
        a4.append(vis_coeff[4])
    var1_str = ['a0', 'a1', 'a2', 'a3', 'a4']
    var1 = [a0, a1, a2, a3, a4]

    C1, C2 = [], []
    for k in range(sp_len):
        for j in range(sp_len):
            c1 = 1 / sqrt(sqrt(8 * (1. + Mi[k] / Mi[j])))
            c2 = c1 * sqrt(sqrt((Mi[j] / Mi[k])))
            C1.append(c1)
            C2.append(c2)
    var2_str = ['C1', 'C2']
    var2 = [C1, C2]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosity"
                f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]) ")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var1_str, var1)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} eta[{sp_len}];", 1)
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} rcp_eta[{sp_len}];", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"eta[k] = a0[k] + a1[k]*lnT + a2[k]*lnT2 + a3[k]*lnT3 + a4[k]*lnT4;", 2)
    cg.add_line(f"rcp_eta[k] = 1./eta[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{write_const_expression(align_width, target, True, var2_str, var2)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"sums[{sp_len}]={{0.}};", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"for(unsigned int j=0; j<{sp_len}; j++)", 2)
    cg.add_line(f"{{", 2)
    cg.add_line(f"unsigned int idx = {sp_len}*k+j;", 3)
    cg.add_line(f"cfloat sqrt_Phi_kj = C1[idx] + C2[idx]*eta[k]*rcp_eta[j];", 3)
    cg.add_line(f"sums[k] += Xi[j]*sqrt_Phi_kj*sqrt_Phi_kj;", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"cfloat vis = 0.;", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"vis += Xi[k]*eta[k]*eta[k]/sums[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"return vis;", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_nonsym_roll(file_name, output_dir, align_width, target, rcp_diffcoeffs,
                                       transport_polynomials, sp_len, Mi):
    """
    Write the 'diffusivity.inc ' file with rolled loop specification
    and  computation of the full Dij matrix (non-symmetrical matrix assumption).
    """

    d0, d1, d2, d3, d4 = [], [], [], [], []
    for k in range(len(transport_polynomials.diffusivity)):
        for j in range(len(transport_polynomials.diffusivity)):
            d0.append(transport_polynomials.diffusivity[k][j][0])
            d1.append(transport_polynomials.diffusivity[k][j][1])
            d2.append(transport_polynomials.diffusivity[k][j][2])
            d3.append(transport_polynomials.diffusivity[k][j][3])
            d4.append(transport_polynomials.diffusivity[k][j][4])
    var_str = ['d0', 'd1', 'd2', 'd3', 'd4']
    var = [d0, d1, d2, d3, d4]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_diffusivity"
                 f"(cfloat Mbar, cfloat p, cfloat TsqrT, cfloat lnT, cfloat lnT2, "
                 f"cfloat lnT3, cfloat lnT4, cfloat Xi[], cfloat* Dkm) ")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                f"sums[{sp_len}]={{0.}};", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"for(unsigned int j=0; j<{sp_len}; j++)", 2)
    cg.add_line(f"{{", 2)
    cg.add_line(f"if (k != j) {{", 3)
    cg.add_line(f"unsigned int idx = k*{sp_len}+j;", 4)
    if rcp_diffcoeffs:
        cg.add_line(f"cfloat rcp_Dkj = d0[idx] + d1[idx]*lnT + d2[idx]*lnT2 + d3[idx]*lnT3 + d4[idx]*lnT4;", 4)
    else:
        cg.add_line(f"cfloat rcp_Dkj = 1/(d0[idx] + d1[idx]*lnT + d2[idx]*lnT2 + d3[idx]*lnT3 + d4[idx]*lnT4);", 4)
    cg.add_line(f"sums[k] += Xi[j]*rcp_Dkj;", 4)
    cg.add_line(f"}}", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{write_const_expression(align_width, target, True, 'Mi', Mi)}")
    cg.add_line("")
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"Dkm[k] = TsqrT * (Mbar - Mi[k]*Xi[k])/(p*Mbar*sums[k]);", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_roll(file_name, output_dir, align_width, target, rcp_diffcoeffs,
                                transport_polynomials, sp_len, Mi):
    """
    Write the 'diffusivity.inc' file with rolled loop specification.
    """

    d0, d1, d2, d3, d4 = [], [], [], [], []
    for k in range(1, len(transport_polynomials.diffusivity)):
        for j in range(k):
            d0.append(transport_polynomials.diffusivity[k][j][0])
            d1.append(transport_polynomials.diffusivity[k][j][1])
            d2.append(transport_polynomials.diffusivity[k][j][2])
            d3.append(transport_polynomials.diffusivity[k][j][3])
            d4.append(transport_polynomials.diffusivity[k][j][4])
    var_str = ['d0', 'd1', 'd2', 'd3', 'd4']
    var = [d0, d1, d2, d3, d4]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_diffusivity"
                f"(cfloat Mbar, cfloat p, cfloat TsqrT, cfloat lnT, cfloat lnT2, "
                f"cfloat lnT3, cfloat lnT4, cfloat Xi[], cfloat* Dkm) ")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"sums1[{sp_len}]={{0.}};", 1)
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"sums2[{sp_len}]={{0.}};", 1)
    cg.add_line(f"for(unsigned int k=1; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"for(unsigned int j=0; j<k; j++)", 2)
    cg.add_line(f"{{", 2)
    cg.add_line(f"unsigned int idx = k*(k-1)/2+j;", 3)
    if rcp_diffcoeffs:
        cg.add_line(f"cfloat rcp_Dkj = d0[idx] + d1[idx]*lnT + d2[idx]*lnT2 + d3[idx]*lnT3 + d4[idx]*lnT4;", 3)
    else:
        cg.add_line(f"cfloat rcp_Dkj = 1/(d0[idx] + d1[idx]*lnT + d2[idx]*lnT2 + d3[idx]*lnT3 + d4[idx]*lnT4);", 3)
    cg.add_line(f"sums1[k] += Xi[j]*rcp_Dkj;", 3)
    cg.add_line(f"sums2[j] += Xi[k]*rcp_Dkj;", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} sums[{sp_len}];", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"sums[k] = sums1[k] + sums2[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{write_const_expression(align_width, target, True, 'Mi', Mi)}")
    cg.add_line(f"")
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"Dkm[k] = TsqrT * (Mbar - Mi[k]*Xi[k])/(p*Mbar*sums[k]);", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_conductivity_unroll(file_name, output_dir, transport_polynomials, sp_names):
    """
    Write the 'conductivity.inc' file with unrolled loop specification.
    """

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivity"
                f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[])")
    cg.add_line(f"{{")
    cg.add_line(f"cfloat lambda_k, sum1 = 0., sum2 = 0.;", 1)
    cg.add_line(f"")
    for k, P in enumerate(transport_polynomials.conductivity):
        cg.add_line(f"//{sp_names[k]}", 1)
        cg.add_line(f"lambda_k = {evaluate_polynomial(P)};", 1)
        cg.add_line(f"sum1 += Xi[{k}]*lambda_k;", 1)
        cg.add_line(f"sum2 += Xi[{k}]/lambda_k;", 1)
        cg.add_line(f"")
    cg.add_line(f"return {f(0.5)} * (sum1 + {f(1.)}/sum2);", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_viscosity_unroll(file_name, output_dir, group_vis, transport_polynomials, sp_names, sp_len, Mi):
    """
    Write the 'viscosity.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat sq(cfloat x) {{ return x*x; }}")
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosity"
                f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat Xi[]) ")
    if group_vis:
        cg.add_line(f"{{")
        for k, P in enumerate(transport_polynomials.viscosity):
            cg.add_line(f"cfloat r{k} = {f(1.)}/({evaluate_polynomial(P)});", 1)
        cg.add_line(f"cfloat v, vis = 0.;", 1)
        for k, P in enumerate(transport_polynomials.viscosity):
            v_expr = f"{evaluate_polynomial(P)}"
            denominator_parts = []
            for j in range(sp_len):
                Va = sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j]))
                part = f"{cg.new_line}{cg.di}Xi[{j}]*sq({f(Va)}+{f(Va * sqrt(sqrt(Mi[j] / Mi[k])))}*r{j}*v)"
                denominator_parts.append(part)
            denominator = " + ".join(denominator_parts)
            cg.add_line("")
            cg.add_line(f"//{sp_names[k]}", 1)
            cg.add_line(f"v = {v_expr};", 1)
            cg.add_line(f"vis += Xi[{k}]*sq(v)/({denominator}{cg.new_line}{cg.si});", 1)
        cg.add_line(f"return vis;", 1)
        cg.add_line(f"}}")
    else:
        cg.add_line(f"{{")
        for k, P in enumerate(transport_polynomials.viscosity):
            cg.add_line(f"cfloat v{k} = {evaluate_polynomial(P)};", 1)
        for k in range(sp_len):
            cg.add_line(f"cfloat sum_{k} = 0.;", 1)
        cg.add_line(f"// same name is used to refer to the reciprocal evaluation "
                    f"explicitly interleaved into the last iteration", 1)

        def sq_v(Va):
            return f'sq({f(Va)}+{f(Va * sqrt(sqrt(Mi[j] / Mi[k])))}*v{k}*r{j})'

        for j in range(sp_len - 1):
            cg.add_line(f"{{", 1)
            cg.add_line(f"cfloat r{j} = {f(1.)}/v{j};", 2)
            for k in range(sp_len):
                cg.add_line(f"sum_{k} += Xi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))};", 2)
            cg.add_line(f"}}", 1)
        for j in [sp_len - 1]:
            cg.add_line(f"{{", 1)
            cg.add_line(f"cfloat r{j} = {f(1.)}/v{j};", 2)
            for k in range(sp_len):
                cg.add_line(f"sum_{k} += Xi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))}; "
                            f"/*rcp_*/sum_{k} = {f(1.)}/sum_{k};", 2)
            cg.add_line(f"}}", 1)
        cg.add_line("")
        cg.add_line(f"""return {('+' + cg.new_line).join(f"{cg.ti if k > 0 else ' '}Xi[{k}]*sq(v{k}) * /*rcp_*/sum_{k}"
                                                           for k in range(sp_len))};""", 1)
        cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_nonsym_unroll(file_name, output_dir, rcp_diffcoeffs,
                                         transport_polynomials, sp_names, sp_len, Mi):
    """
    Write the 'diffusivity.inc' file with unrolled loop specification
    and  computation of the full Dij matrix (non-symmetrical matrix assumption).
    """

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_diffusivity"
                f"(cfloat Mbar, cfloat p, cfloat TsqrT, cfloat lnT, cfloat lnT2, "
                f"cfloat lnT3, cfloat lnT4, cfloat Xi[], cfloat* Dkm) ")
    cg.add_line(f"{{")
    for k in range(sp_len):
        cg.add_line(f"//{sp_names[k]}", 1)
        cg.add_line(f"Dkm[{k}] = TsqrT * (Mbar - nekrk_molar_mass[{k}]*Xi[{k}]) / (p*Mbar*(", 1)
        if rcp_diffcoeffs:
            cg.add_line(
                f"""{('+' + cg.new_line).join(
                    f"{cg.di}Xi[{j}] * "
                    f"({evaluate_polynomial(transport_polynomials.diffusivity[k if k > j else j][j if k > j else k])})"
                    for j in list(range(k)) + list(range(k + 1, sp_len)))}));""")
        else:
            cg.add_line(
                f"""{('+' + cg.new_line).join(
                    f"{cg.di}Xi[{j}] * "
                    f"(1/"
                    f"({evaluate_polynomial(transport_polynomials.diffusivity[k if k > j else j][j if k > j else k])}))"
                    for j in list(range(k)) + list(range(k + 1, sp_len)))}));""")
        cg.add_line(f"")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_unroll(file_name, output_dir, rcp_diffcoeffs, transport_polynomials, sp_len, Mi):
    """
    Write the 'diffusivity.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()

    S = [''] * sp_len

    def mut(y, i, v):
        S[i] = v
        return y

    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_diffusivity"
                f"(cfloat Mbar, cfloat p, cfloat TsqrT, cfloat lnT, cfloat lnT2, "
                f"cfloat lnT3, cfloat lnT4, cfloat Xi[], cfloat* Dkm) ")
    cg.add_line(f"{{")
    for k in range(sp_len):
        for j in range(k):
            if rcp_diffcoeffs:
                cg.add_line(
                    f"cfloat D{k}_{j} = {evaluate_polynomial(transport_polynomials.diffusivity[k][j])};", 1)
            else:
                cg.add_line(
                    f"cfloat D{k}_{j} = 1/({evaluate_polynomial(transport_polynomials.diffusivity[k][j])});", 1)
            cg.add_line(f"cfloat S{k}_{j} = {mut(f'{S[k]}+' if S[k] else '', k, f'S{k}_{j}')}Xi[{j}]*D{k}_{j};", 1)
            cg.add_line(f"cfloat S{j}_{k} = {mut(f'{S[j]}+' if S[j] else '', j, f'S{j}_{k}')}Xi[{k}]*D{k}_{j};", 1)
    for k in range(sp_len):
        cg.add_line(f"Dkm[{k}] = TsqrT * (Mbar - {f(Mi[k])}*Xi[{k}])/(p*Mbar*{S[k]});", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0
