#!/usr/bin/env python3

"""
Generate source code files for computing chemical production rates,
thermodynamic properties, and transport coefficients.
"""

# Standard libraries
import re
import math
import argparse
from numpy import dot
from numpy import pi
from numpy import sqrt
from numpy import linspace
from numpy import log as ln
from numpy import square as sq
from functools import reduce
from ruamel.yaml import YAML
from sys import version_info

# Local imports
from utils import cube
from utils import polynomial_regression
from utils import set_precision, f, f_sci_not
from utils import sum_of_list, partition, imul
from utils import FLOAT_MIN, FLOAT_MAX
from utils import CodeGenerator
import constants as const


def get_parser():
    """
    Create a command line argument parser for running the code generator.
    """

    parser = argparse.ArgumentParser(description=
                                     'Generates production rates, thermodynamic '
                                     'and transport properties evaluation code')
    parser.add_argument('--mechanism',
                        required=True,
                        default="mechanisms/gri30.yaml",
                        help='Path to yaml mechanism file.')
    parser.add_argument('--output',
                        required=True,
                        default="share/mechanisms",
                        help='Output directory.')
    parser.add_argument('--single-precision',
                        required=False,
                        action='store_true',
                        help='Generate single precision source code.')
    parser.add_argument('--header-only',
                        required=False,
                        action='store_true',
                        help='Only create the header file, mech.h.')
    parser.add_argument('--unroll-loops',
                        required=False,
                        action='store_true',
                        help='Unroll loops in the generated subroutines.')
    parser.add_argument('--align-width',
                        required=False,
                        default=64,
                        help='Alignment width of arrays')
    parser.add_argument('--target',
                        required=False,
                        default="CUDA",
                        help='Target platform and c++ version')
    parser.add_argument('--loop-gibbsexp',
                        required=False,
                        action='store_true',
                        help='Loop calculation of gibbs exponential in case '
                             'the code is unrolled.')
    parser.add_argument('--group-rxnunroll',
                        required=False,
                        action='store_true',
                        help='Group reactions in the case of unrolled code '
                             'based on repeated beta and E_R.')
    parser.add_argument('--transport',
                        required=False,
                        default=True,
                        help='Write transport properties')
    parser.add_argument('--group-vis',
                        required=False,
                        action='store_true',
                        help='Group species viscosity')
    parser.add_argument('--nonsymDij',
                        required=False,
                        action='store_true',
                        help='Compute both the upper and lower part of the'
                             'Dij matrix, although it is symmetric. It avoids'
                             'expensive memory allocation in some cases.')
    parser.add_argument('--fit-rcpdiffcoeffs',
                        required=False,
                        action='store_true',
                        help='Compute the reciprocal of the diffusion '
                             'coefficients to avoid expensive divisions '
                             'in the diffusivity kernel.')
    args = parser.parse_args()
    return args


def read_mechanism_yaml(mechanism_file):
    """
    Read a reaction mechanism from a YAML file and return the corresponding model.
    """

    model = YAML().load(open(mechanism_file))
    # Check units for consistency
    yaml_units = model['units']
    assert (yaml_units.get("length", "m") == "cm" and
            yaml_units.get("time", "s") == "s" and
            yaml_units.get("quantity", "mol") == "mol")
    return model


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
                 well_depth, dipole_moment, diameter, rotational_relaxation,
                 degrees_of_freedom, polarizability, sp_len):
        self.rcp_diffcoeffs = rcp_diffcoeffs
        self.sp_names = sp_names
        self.Mi = molar_masses
        self.thermo = thermodynamics
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
        header_lnT_star = self._header_lnT_star
        # Find the first index where lnT_star is less than the corresponding value in header_lnT_star
        for i, val in enumerate(header_lnT_star[1:], start=1):
            if lnT_star < val:
                interp_start_index = i - 1
                break
        else:
            interp_start_index = len(header_lnT_star) - 2
        interp_start_index = min(interp_start_index, I0 + len(table) - 3)
        header_lnT_star_slice = header_lnT_star[interp_start_index:][:3]
        assert (len(header_lnT_star_slice) == 3)
        polynomials = fit[interp_start_index - I0:][:3]
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
        table = table[interp_start_index - I0:][:3]
        assert (len(table) == 3)
        # P[:6]: Reproduces Cantera truncated polynomial mistake
        if delta_star == 0.0:
            y = [row[0] for row in table]
        else:
            y = [_evaluate_polynomial(P[:6], delta_star) for P in polynomials]
        return _quadratic_interpolation(header_lnT_star_slice, y, lnT_star)

    def _omega_star_22(self, j, k, T):
        return self._collision_integral(1, const.collision_integrals_Omega_star_22, const.Omega_star_22, j, k, T)

    def _omega_star_11(self, j, k, T):
        return (self._omega_star_22(j, k, T) /
                self._collision_integral(0, const.collision_integrals_A_star, const.A_star, j, k, T))

    def _viscosity(self, k, T):
        (Mi, sigma) = (self.Mi, self._sigma)
        return (5. / 16. * sqrt(pi * Mi[k] / const.NA * const.kB * T) /
                (pi * sq(sigma[k]) * self._omega_star_22(k, k, T)))

    # p*Djk
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
        T_rng = linspace(300., 3000., 50)

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
    degrees_of_freedom = p(lambda s: {'atom': 0, 'linear': 1, 'nonlinear': 3 / 2}[s['transport']['geometry']])
    well_depth = p(lambda s: s['transport']['well-depth'] * const.kB)
    diameter = p(lambda s: s['transport']['diameter'] * 1e-10)  # Å
    dipole_moment = p(lambda s: s['transport'].get('dipole', 0) * const.Cm_per_Debye)
    polarizability = p(lambda s: s['transport'].get('polarizability', 0) * 1e-30)  # Å³
    rotational_relaxation = p(lambda s: float(s['transport'].get('rotational-relaxation', 0)))

    species = Species(rcp_diffcoeffs, sp_names, molar_masses, thermodynamics,
                      well_depth, dipole_moment, diameter, rotational_relaxation,
                      degrees_of_freedom, polarizability, sp_len)
    return species


def get_reaction_from_model(sp_names, units, r):
    """
    Extract reaction information and attributes from the mechanism model.
    """

    class Reaction:
        pass

    reaction = Reaction()
    reaction.description = r['equation']
    # Regular expression to match third body and fall-off reactions
    third_body_pattern = re.compile(r'(\s*\(\+[^\)]+\))|(\s*\+\s*M\s*)')
    if version_info.minor >= 9:
        # print(r['equation'])
        [reaction.reactants, reaction.products] = [
            [sum([c for (s, c) in side if s == specie]) for specie in sp_names] for side in
            [[(s.split(' ')[1], int(s.split(' ')[0])) if ' ' in s else (s, 1) for s in
              [re.sub(third_body_pattern, '', s.strip()) for s in side.split(' + ')]]
             for side in [s.strip() for s in re.split('<?=>', r['equation'])]
             ]
        ]
    else:
        # Regular expression to match third body and fall-off reactions
        third_body_pattern = re.compile(r'(\s*\(\+[^)]+\)$)|(\s*\+\s*M\s*$)')
        [reaction.reactants, reaction.products] = [
            [sum([c for (s, c) in side if s == specie]) for specie in sp_names] for side in
            [[(s.split(' ')[1], int(s.split(' ')[0])) if ' ' in s else (s, 1) for s in
              [re.sub(third_body_pattern, '', s.strip()) for s in side.split(' + ')]]
             for side in [s.strip() for s in re.split('<?=>', r['equation'])]
             ]
        ]
    reaction.net = [-reactant + product for reactant, product in zip(reaction.reactants, reaction.products)]
    reaction.sum_net = sum(reaction.net)

    match = third_body_pattern.search(r['equation'])
    if match:
        third_body = match.group(1) or match.group(2)
        third_body = third_body.replace('+', '').replace('(', '').replace(')', '').replace(' ', '')
        if third_body != 'M' and third_body in sp_names:
            reaction.third_body_index =  sp_names.index(third_body)
        else:
            reaction.third_body_index = -1
    else:
        reaction.third_body_index = -1

    def rate_constant(self, concentration_cm3_unit_conversion_factor_exponent):

        class RateConstant:
            pass

        rate_constant = RateConstant()
        rate_constant.preexponential_factor = self['A'] * pow(1e-6, concentration_cm3_unit_conversion_factor_exponent)
        rate_constant.temperature_exponent = self['b']
        Ea = self['Ea']
        if units['activation-energy'] == 'K':
            rate_constant.activation_temperature = Ea
        elif units['activation-energy'] == 'cal/mol':
            rate_constant.activation_temperature = Ea * const.J_per_cal / const.R
        else:
            exit('activation-energy')
        return rate_constant

    reactants = sum(reaction.reactants)
    if r.get('type') == "three-body":
        reactants += 1

    if r.get('rate-constant') or r.get('high-P-rate-constant'):
        reaction.rate_constant = rate_constant(r.get('rate-constant', r.get('high-P-rate-constant')), reactants - 1)
    else:
        reaction.rate_constant = rate_constant(r['rate-constants'][0], reactants - 1)
        if r['type'] != 'pressure-dependent-Arrhenius':
            print(f"expected P-log: {r}")
        if len(r['rate-constants']) != 1:
            exit(f"unimplemented P-log: always using first rate constant {r['rate-constants'][0]} for {r} instead")

    if r.get('type') is None or r.get('type') == 'elementary':
        if re.search('[^<]=>', r['equation']):
            reaction.type = 'irreversible'
            reaction.direction = 'irreversible'
        elif '<=>' in r['equation'] or '= ' in r['equation']:
            # Keep the space after = ('= ') here. or let '[^<]=>' match first
            assert (r.get('reversible', True))
            reaction.type = 'elementary'
            reaction.direction = 'reversible'
        else:
            exit(r)
    elif r.get('type') == 'three-body':
        reaction.type = 'three-body'
        if re.search('[^<]=>', r['equation']):
            reaction.direction = 'irreversible'
        else:
            reaction.direction = 'reversible'
    elif r.get('type') == 'falloff' and not r.get('Troe') and not r.get('SRI'):
        reaction.type = 'pressure-modification'
        if re.search('[^<]=>', r['equation']):
            reaction.direction = 'irreversible'
        else:
            reaction.direction = 'reversible'
        reaction.k0 = rate_constant(r['low-P-rate-constant'], reactants)
    elif r.get('type') == 'falloff' and r.get('Troe'):
        reaction.type = 'Troe'
        if re.search('[^<]=>', r['equation']):
            reaction.direction = 'irreversible'
        else:
            reaction.direction = 'reversible'
        reaction.k0 = rate_constant(r['low-P-rate-constant'], reactants)

        class Troe:
            pass

        reaction.troe = Troe()
        reaction.troe.A = r['Troe']['A']
        reaction.troe.T3 = r['Troe']['T3']
        reaction.troe.T1 = r['Troe']['T1']
        reaction.troe.T2 = r['Troe'].get('T2', float('inf'))
    elif r.get('type') == 'falloff' and r.get('SRI'):
        reaction.type = 'SRI'
        if re.search('[^<]=>', r['equation']):
            reaction.direction = 'irreversible'
        else:
            reaction.direction = 'reversible'
        reaction.k0 = rate_constant(r['low-P-rate-constant'], reactants)

        class SRI:
            pass

        reaction.sri = SRI()
        reaction.sri.A = r['SRI']['A']
        reaction.sri.B = r['SRI']['B']
        reaction.sri.C = r['SRI']['C']
        reaction.sri.D = r['SRI'].get('D', 1)
        reaction.sri.E = r['SRI'].get('E', 0)
    elif r.get('type') == 'pressure-dependent-Arrhenius':
        if '<=>' in r['equation'] or '=' in r['equation']:
            assert (r.get('reversible', True))
            reaction.type = 'elementary'
        elif re.search('[^<]=>', r['equation']):
            reaction.type = 'irreversible'
        else:
            exit(r)
    else:
        exit(r)

    if r.get('efficiencies'):
        reaction.efficiencies = [
            r['efficiencies'].get(specie, r.get('default-efficiency', 1)) for specie in sp_names]
    return reaction


def write_thermo_piece(out, sp_indices, sp_thermo, expression, p):
    """
    Write a thermodynamic piece expression.
    """
    cg = CodeGenerator()
    for specie in sp_indices:
        if '[' in out:
            line = f"{out}{specie}] = {expression(sp_thermo[specie].pieces[p])};"
        else:
            line = f"cfloat {out}{specie} = {expression(sp_thermo[specie].pieces[p])};"
        cg.add_line(line)

    return cg.get_code()


def write_energy(out, length, expression, sp_thermo):
    """
    Write the evaluation of the energy log polynomial for a given species.
    """
    cg = CodeGenerator()
    temperature_splits = {}
    for index, specie in enumerate(sp_thermo[:length]):
        temperature_splits.setdefault(specie.temp_split, []).append(index)

    for temperature_split, species in temperature_splits.items():
        cg.add_line(f"if (T <= {temperature_split}) {{", 1)
        inner_code = write_thermo_piece(out, species, sp_thermo, expression, 0)
        for line in inner_code.split('\n'):
            cg.add_line(line, 2)
        cg.add_line("} else {", 1)
        inner_code = write_thermo_piece(out, species, sp_thermo, expression, 1)
        for line in inner_code.split('\n'):
            cg.add_line(line, 2)
        cg.add_line("}", 1)

    return cg.get_code()


def compute_k_rev_unroll(r):
    """
    Calculate the reverse rate constant for a given reaction for the unrolled code.
    """

    pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
    gibbs_terms = []
    gibbs_terms_div = []
    rcp_gibbs_terms = []
    for j, net in enumerate(r.net):
        if net > 0:
            for o in range(net):
                gibbs_terms.append(f"gibbs0_RT[{j}]")
    rcp_gibbs_terms.append(f"1./(")
    for k, net in enumerate(r.net):
        if net < 0:
            for n in range(abs(net)):
                gibbs_terms_div.append(f"gibbs0_RT[{k}]")
    rcp_gibbs_terms.append(f"{'*'.join(gibbs_terms_div)}")
    rcp_gibbs_terms.append(f")")
    gibbs_terms.append(''.join(rcp_gibbs_terms))
    k_rev = f"{'*'.join(gibbs_terms)}{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};"
    return k_rev


def write_reaction(idx, r, loop_gibbsexp):
    """
    Write reaction for the unrolled code.
    """
    cg = CodeGenerator()
    cg.add_line(f"//{idx + 1}: {r.description}", 1)

    if hasattr(r, 'efficiencies'):
        efficiency = [f'{f(efficiency - 1)}*Ci[{specie}]' if efficiency != 2 else f'Ci[{specie}]'
                      for specie, efficiency in enumerate(r.efficiencies) if efficiency != 1]
        cg.add_line(f'''eff = Cm{f"+{'+'.join(efficiency)}" if efficiency else ''};''', 1)

    def arrhenius(rc):
        A, beta, E = (
            rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
        if beta == 0 and E != 0:
            expression = f'exp({f(ln(A))} + {f(-E)}*rcpT)'
        elif beta == 0 and E == 0:
            expression = f'{f(A)}'
        elif beta != 0 and E == 0:
            if beta == -2 and E == 0:
                expression = f'{f(A)}*rcpT*rcpT'
            elif beta == -1 and E == 0:
                expression = f'{f(A)}*rcpT'
            elif beta == 1 and E == 0:
                expression = f'{f(A)}*T'
            elif beta == 2 and E == 0:
                expression = f'{f(A)}*T*T'
            else:
                expression = f'exp({f(ln(A))} + {f(beta)}*lnT)'
        else:
            expression = f'exp({f(ln(A))} + {f(beta)}*lnT + {f(-E)}*rcpT)'
        return expression

    def arrhenius_diff(rc):
        A_inf, beta_inf, E_inf = (
            rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
        A0, beta0, E0 = r.k0.preexponential_factor, r.k0.temperature_exponent, r.k0.activation_temperature
        expression = (
            f"exp("
            f"{f'{f(-E0 + E_inf)}*rcpT+' if (E0 - E_inf) != 0 else ''}"
            f"{f'{f(beta0 - beta_inf)}*lnT+' if (beta0 - beta_inf) != 0 else ''}{f(ln(A0) - ln(A_inf))})"
            if (A0 - A_inf) != 0 and ((beta0 - beta_inf) != 0 or (E0 - E_inf) != 0) else f'{f(A0 / A_inf)}'
        )
        return expression

    if r.type == 'elementary' or r.type == 'irreversible':
        cg.add_line(f'k = {arrhenius(r.rate_constant)};', 1)
    elif r.type == 'three-body':
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"k = {arrhenius(r.rate_constant)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"k = {arrhenius(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"k = {arrhenius(r.rate_constant)} * Cm;", 1)
    elif r.type == 'pressure-modification':
        cg.add_line(f"k_inf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
        cg.add_line(f"k = k_inf * Pr/(1 + Pr);", 1)
    elif r.type == 'Troe':
        cg.add_line(f"k_inf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
        cg.add_line(f"logPr = log10(Pr + CFLOAT_MIN);", 1)
        # Add checks for troe coefficients
        if r.troe.A == 0:
            cg.add_line(f"logFcent = log10(exp({-1. / (r.troe.T3 + FLOAT_MIN)}*T) + "
                        f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
        elif r.troe.A == 1:
            cg.add_line(f"logFcent = log10(exp({-1. / (r.troe.T1 + FLOAT_MIN)}*T)"
                         f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
        else:
            cg.add_line(f"logFcent = log10({1 - r.troe.A}*exp({-1. / (r.troe.T3 + FLOAT_MIN)}*T) + "
                        f"{r.troe.A}*exp({-1. / (r.troe.T1 + FLOAT_MIN)}*T)"
                        f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
        cg.add_line(f"troe_c = -.4 - .67 * logFcent;", 1)
        cg.add_line(f"troe_n = .75 - 1.27 * logFcent;", 1)
        cg.add_line(f"troe = (troe_c + logPr)/(troe_n - .14*(troe_c + logPr));", 1)
        cg.add_line(f"F = pow(10, logFcent/(1.0 + troe*troe));", 1)
        cg.add_line(f"k = k_inf * Pr/(1 + Pr) * F;", 1)
    elif r.type == 'SRI':
        cg.add_line(f"k_inf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
        cg.add_line(f"logPr = log10(Pr);", 1)
        cg.add_line(f"F = {r.sri.D}*pow({r.sri.A}*exp({-r.sri.B}*rcpT)+"
                    f"exp({-1. / (r.sri.C + FLOAT_MIN)}*T), 1./(1.+logPr*logPr))*pow(T, {r.sri.E});", 1)
        cg.add_line(f"k = k_inf * Pr/(1 + Pr) * F;", 1)
    else:
        exit(r.type)

    phase_space = lambda reagents: '*'.join(
        '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
        coefficient != 0.)
    Rf = phase_space(r.reactants)
    cg.add_line(f"Rf= {Rf};", 1)
    if r.type == 'irreversible' or r.direction == 'irreversible':
        cg.add_line(f"cR = k * Rf;", 1)
    else:
        pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
        if loop_gibbsexp:
            cg.add_line(f"k_rev = {compute_k_rev_unroll(r)}", 1)
        else:
            cg.add_line(f"k_rev = __NEKRK_EXP_OVERFLOW__("
                        f"{'+'.join(imul(net, f'gibbs0_RT[{k}]') for k, net in enumerate(r.net) if net != 0)})"
                        f"{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};", 1)
        cg.add_line(f"Rr = k_rev * {phase_space(r.products)};", 1)
        cg.add_line(f"cR = k * (Rf - Rr);", 1)
    cg.add_line(f"#ifdef DEBUG")
    cg.add_line(f'printf("{idx + 1}: %+.15e\\n", cR);', 1)
    cg.add_line(f"#endif")
    for specie, net in enumerate(r.net):
        if net != 0:
            cg.add_line(f"rates[{specie}] += {imul(net, 'cR')};", 1)
    cg.add_line(f"")

    return cg.get_code()


def write_reaction_grouped(grouped_rxn, first_idx, loop_gibbsexp):
    """
    Write grouped reaction for the unrolled code.
    """
    cg = CodeGenerator()
    for count, (idx, r) in enumerate(grouped_rxn, start=0):
        if count >= 1:
            previous_r = grouped_rxn[count-1][1]

        cg.add_line(f"//{idx + 1}: {r.description}", 1)

        if hasattr(r, 'efficiencies'):
            efficiency = [f'{f(efficiency - 1)}*Ci[{specie}]' if efficiency != 2 else f'Ci[{specie}]'
                          for specie, efficiency in enumerate(r.efficiencies) if efficiency != 1]
            cg.add_line(f'''eff = Cm{f"+{'+'.join(efficiency)}" if efficiency else ''};''', 1)

        def arrhenius(rc):
            A, beta, E = (
                rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
            if beta == 0 and E != 0:
                expression = f'exp({f(ln(A))} + {f(-E)}*rcpT)'
            elif beta == 0 and E == 0:
                expression = f'{f(A)}'
            elif beta != 0 and E == 0:
                if beta == -2 and E == 0:
                    expression = f'{f(A)}*rcpT*rcpT'
                elif beta == -1 and E == 0:
                    expression = f'{f(A)}*rcpT'
                elif beta == 1 and E == 0:
                    expression = f'{f(A)}*T'
                elif beta == 2 and E == 0:
                    expression = f'{f(A)}*T*T'
                else:
                    expression = f'exp({f(ln(A))} + {f(beta)}*lnT)'
            else:
                expression = f'exp({f(ln(A))} + {f(beta)}*lnT + {f(-E)}*rcpT)'
            return expression

        def arrhenius_diff(rc):
            A_inf, beta_inf, E_inf = (
                rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
            A0, beta0, E0 = r.k0.preexponential_factor, r.k0.temperature_exponent, r.k0.activation_temperature
            expression = (
                f"exp("
                f"{f'{f(-E0 + E_inf)}*rcpT+' if (E0 - E_inf) != 0 else ''}"
                f"{f'{f(beta0 - beta_inf)}*lnT+' if (beta0 - beta_inf) != 0 else ''}{f(ln(A0) - ln(A_inf))})"
                if (A0 - A_inf) != 0 and ((beta0 - beta_inf) != 0 or (E0 - E_inf) != 0) else f'{f(A0 / A_inf)}'
            )
            return expression

        if r.type == 'elementary' or r.type == 'irreversible':
            if idx == first_idx:
                cg.add_line(f'k = {arrhenius(r.rate_constant)};', 1)
            else:
                cg.add_line(f'k *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
        elif r.type == 'three-body':
            if idx == first_idx:
                cg.add_line(f"k = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f"k *= "
                            f"{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};",
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"k_corr = k * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"k_corr = k * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"k_corr = k * Cm;", 1)
        elif r.type == 'pressure-modification':
            if idx == first_idx:
                cg.add_line(f"k = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'k *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
            cg.add_line(f"k_corr = k * Pr/(1 + Pr);", 1)
        elif r.type == 'Troe':
            if idx == first_idx:
                cg.add_line(f"k = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'k *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
            cg.add_line(f"logPr = log10(Pr + CFLOAT_MIN);", 1)
            # Add checks for troe coefficients
            if r.troe.A == 0:
                cg.add_line(f"logFcent = log10(exp({-1. / (r.troe.T3 + FLOAT_MIN)}*T) + "
                             f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
            elif r.troe.A == 1:
                cg.add_line(f"logFcent = log10(exp({-1. / (r.troe.T1 + FLOAT_MIN)}*T)"
                            f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
            else:
                cg.add_line(f"logFcent = log10({1 - r.troe.A}*exp({-1. / (r.troe.T3 + FLOAT_MIN)}*T) + "
                            f"{r.troe.A}*exp({-1. / (r.troe.T1 + FLOAT_MIN)}*T)"
                            f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});", 1)
            cg.add_line(f"troe_c = -.4 - .67 * logFcent;", 1)
            cg.add_line(f"troe_n = .75 - 1.27 * logFcent;", 1)
            cg.add_line(f"troe = (troe_c + logPr)/(troe_n - .14*(troe_c + logPr));", 1)
            cg.add_line(f"F = pow(10, logFcent/(1.0 + troe*troe));", 1)
            cg.add_line(f"k_corr = k * Pr/(1 + Pr) * F;", 1)
        elif r.type == 'SRI':
            if idx == first_idx:
                cg.add_line(f"k = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'k *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r.rate_constant)} * Cm;", 1)
            cg.add_line(f"logPr = log10(Pr);", 1)
            cg.add_line(f"F = {r.sri.D}*pow({r.sri.A}*exp({-r.sri.B}*rcpT)+"
                        f"exp({-1. / (r.sri.C + FLOAT_MIN)}*T), 1./(1.+logPr*logPr))*pow(T, {r.sri.E});", 1)
            cg.add_line(f"k_corr = k * Pr/(1 + Pr) * F;", 1)
        else:
            exit(r.type)

        phase_space = lambda reagents: '*'.join(
            '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
            coefficient != 0.)
        Rf = phase_space(r.reactants)
        cg.add_line(f"Rf= {Rf};", 1)
        if r.type == 'irreversible' or r.direction == 'irreversible':
            cg.add_line(f"cR = k * Rf;", 1)
        else:
            pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
            if loop_gibbsexp:
                cg.add_line(f"k_rev = {compute_k_rev_unroll(r)}", 1)
            else:
                cg.add_line(f"k_rev = __NEKRK_EXP_OVERFLOW__("
                            f"{'+'.join(imul(net, f'gibbs0_RT[{k}]') for k, net in enumerate(r.net) if net != 0)})"
                            f"{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};", 1)
            cg.add_line(f"Rr = k_rev * {phase_space(r.products)};", 1)
            if r.type == 'elementary' or r.type == 'irreversible':
                cg.add_line(f"cR = k * (Rf - Rr);", 1)
            else:
                cg.add_line(f"cR = k_corr * (Rf - Rr);", 1)
        cg.add_line(f"#ifdef DEBUG")
        cg.add_line(f'printf("{idx + 1}: %+.15e\\n", cR);', 1)
        cg.add_line(f"#endif")
        for specie, net in enumerate(r.net):
            if net != 0:
                cg.add_line(f"rates[{specie}] += {imul(net, 'cR')};", 1)
        cg.add_line(f"")

    return cg.get_code()


def evaluate_polynomial(P):
    """
    Create a string representation of the polynomial evaluation.
    """
    return f'{f(P[0])}+{f(P[1])}*lnT+{f(P[2])}*lnT2+{f(P[3])}*lnT3+{f(P[4])}*lnT4'


def write_const_expression(align_width, target, static, var_str, var):
    """
    Create a constant expression using a scientific notation for variables or list
    of variables.
    """
    cg = CodeGenerator()

    if static:
        if target == 'c++17':
            vtype = f'alignas({align_width}) static constexpr'
        else:
            vtype = 'const'
    else:
        vtype = ""

    if isinstance(var_str, list):
        assert len(var_str) == len(var)
        for i in range(len(var_str)):
            cg.add_line(f"{vtype} cfloat {var_str[i]}[{len(var[i])}] = {{{f_sci_not(var[i])}}};", 1)
            cg.add_line("")
    else:
        cg.add_line(f"{vtype} cfloat {var_str}[{len(var)}] = {{{f_sci_not(var)}}};", 1)

    return cg.get_code()


def get_thermo_coeffs(thermo_prop, sp_thermo, sp_len):
    """
    Extract thermodynamic coefficients, temperature thresholds and reordering indices
    from species thermodynamic data.
    """
    # Energy coefficients
    a0, a1, a2, a3, a4, a5, a6 = [], [], [], [], [], [], []
    temp_splits = []
    len_unique_temp_splits = []
    ids_thermo_old = [i for i in range(sp_len)]
    ids_thermo_new = []
    for index, specie in enumerate(sp_thermo):
        temp_splits.append(specie.temp_split)

    unique_temp_split = sorted(list(set(temp_splits)))
    for i in range(len(unique_temp_split)):
        n = 0
        for idx_j, specie_j in enumerate(sp_thermo):
            if specie_j.temp_split == unique_temp_split[i]:
                n += 1
                a0.append(specie_j.pieces[0][0])
                a1.append(specie_j.pieces[0][1])
                a2.append(specie_j.pieces[0][2])
                a3.append(specie_j.pieces[0][3])
                a4.append(specie_j.pieces[0][4])
                a5.append(specie_j.pieces[0][5])
                a6.append(specie_j.pieces[0][6])
                ids_thermo_new.append(idx_j)
        len_unique_temp_splits.append(n)
        for idx_k, specie_k in enumerate(sp_thermo):
            if specie_k.temp_split == unique_temp_split[i]:
                a0.append(specie_k.pieces[1][0])
                a1.append(specie_k.pieces[1][1])
                a2.append(specie_k.pieces[1][2])
                a3.append(specie_k.pieces[1][3])
                a4.append(specie_k.pieces[1][4])
                a5.append(specie_k.pieces[1][5])
                a6.append(specie_k.pieces[1][6])

    if thermo_prop == 'cp_R':
        pass
    elif thermo_prop == 'h_RT':
        a1 = [i/2 for i in a1]
        a2 = [i/3 for i in a2]
        a3 = [i/4 for i in a3]
        a4 = [i/5 for i in a4]
    elif thermo_prop == 'g_RT':
        a1 = [i*(1/2-1) for i in a1]
        a2 = [i*(1/3-1/2) for i in a2]
        a3 = [i*(1/4-1/3) for i in a3]
        a4 = [i*(1/5-1/4) for i in a4]
    else:
        exit("Undefined thermodynamic property")

    return a0, a1, a2, a3, a4, a5, a6, ids_thermo_new, len_unique_temp_splits, unique_temp_split


def get_thermo_prop(thermo_prop, unique_temp_split, len_unique_temp_splits):
    """
    Compute the polynomial evaluation of a specific thermodynamic property.
    """
    cg = CodeGenerator()

    cg.add_line("unsigned int offset;", 1)
    cg.add_line("unsigned int i_off;", 1)
    for i in range(len(unique_temp_split)):
        if len_unique_temp_splits[i] > 1:
            cg.add_line(
                f"offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + "
                f"(T>{unique_temp_split[i]})*{len_unique_temp_splits[i]};", 1)
        else:
            cg.add_line(
                f"offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + (T>{unique_temp_split[i]});", 1)
        cg.add_line(f"for(unsigned int i=0; i<{len_unique_temp_splits[i]}; ++i)", 1)
        cg.add_line("{", 1)
        cg.add_line("i_off = i + offset;", 2)
        if thermo_prop == 'cp_R':
            cg.add_line(
                f"cp_R[i+{sum_of_list(len_unique_temp_splits[:i])}] = a0[i_off] + a1[i_off]*T + a2[i_off]*T2 + "
                f"a3[i_off]*T3 + a4[i_off]*T4;", 2)
        elif thermo_prop == 'h_RT':
            cg.add_line(
                f"h_RT[i+{sum_of_list(len_unique_temp_splits[:i])}] = a0[i_off] + a1[i_off]*T + "
                f"a2[i_off]*T2 + a3[i_off]*T3 + a4[i_off]*T4 + a5[i_off]*rcpT;", 2)
        elif thermo_prop == 'g_RT':
            cg.add_line(
                f"gibbs0_RT[i+{sum_of_list(len_unique_temp_splits[:i])}] = "
                f"a0[i_off]*(1-lnT) + a1[i_off]*T + a2[i_off]*T2 +"
                f"a3[i_off]*T3 + a4[i_off]*T4 + a5[i_off]*rcpT - a6[i_off];", 2)
        else:
            exit('Undefined thermodynamic property')
        cg.add_line("}", 1)
        cg.add_line("")

    return cg.get_code()


def reorder_thermo_prop(thermo_prop, unique_temp_split, ids_thermo_prop_new, sp_len):
    """
    Reorder thermodynamic property data according to new indices.
    """
    cg = CodeGenerator()

    if len(unique_temp_split) > 1:
        cg.add_line("//Reorder thermodynamic properties", 1)
        cg.add_line(f"cfloat tmp[{sp_len}];", 1)
        cg.add_line(f"for(unsigned i=0; i<{sp_len}; ++i)", 1)
        if thermo_prop == 'cp_R':
            cg.add_line("tmp[i] = cp_R[i];", 2)
        elif thermo_prop == 'h_RT':
            cg.add_line("tmp[i] = h_RT[i];", 2)
        for i in range(sp_len):
            if thermo_prop == 'cp_R':
                cg.add_line(f"cp_R[{i}] = tmp[{ids_thermo_prop_new.index(i)}];", 1)
            elif thermo_prop == 'h_RT':
                cg.add_line(f"h_RT[{i}] = tmp[{ids_thermo_prop_new.index(i)}];", 1)
    else:
        cg.add_line("")

    return cg.get_code()


def write_file_mech(file_name, output_dir, sp_names, sp_len, active_sp_len, rxn_len, Mi):
    """
    Write the 'mech.h' file for the reaction mechanism data.
    """
    cg = CodeGenerator()

    cg.add_line(f'#define n_species {sp_len}')
    cg.add_line(f'#define n_active_species {active_sp_len}')
    cg.add_line(f'#define species_names_length {reduce(lambda x, y: x + y, [len(i) for i in sp_names]) + len(sp_names)}')
    cg.add_line(f'#define n_reactions {rxn_len}')
    cg.add_line(f"""__NEKRK_CONST__ char species_names[species_names_length] = "{' '.join(sp_names)}";""")
    cg.add_line(f"__NEKRK_CONST__ cfloat nekrk_molar_mass[n_species] = {{{', '.join([repr(w) for w in Mi])}}};")
    cg.add_line(f"__NEKRK_CONST__ cfloat nekrk_rcp_molar_mass[n_species] = {{{', '.join([repr(1. / w) for w in Mi])}}};")
    cg.add_line(f'#define __NEKRK_NSPECIES__ n_species', 0)
    cg.add_line(f'#define __NEKRK_NACTIVESPECIES__ n_active_species', 0)

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_rates_unroll(file_name, output_dir, loop_gibbsexp, group_rxnunroll,
                            reactions, active_len, sp_len, sp_thermo):
    """
    Write the 'rates.inc'('frates.inc') file with unrolled loop specification.
    Loops are expanded by replicating their body multiple times, reducing the
    overhead of loop control which can lead to more efficient code execution,
    particularly beneficial for GPUs.
    """

    cg = CodeGenerator()
    cg.add_line(f"#include <math.h>")
    cg.add_line(f"#define __NEKRK_EXP_OVERFLOW__(x) __NEKRK_MIN_CFLOAT(CFLOAT_MAX, exp(x))")
    cg.add_line(f"__NEKRK_DEVICE__ __NEKRK_INLINE__ void nekrk_species_rates"
                 f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                 f"const cfloat rcpT, const cfloat Ci[], cfloat* rates) ")
    cg.add_line(f"{{")
    cg.add_line(f"cfloat gibbs0_RT[{active_len}];", 1)
    expression = lambda a: (f"{f(a[5])} * rcpT + {f(a[0] - a[6])} + {f(-a[0])} * lnT + "
                            f"{f(-a[1] / 2)} * T + {f((1. / 3. - 1. / 2.) * a[2])} * T2 + "
                            f"{f((1. / 4. - 1. / 3.) * a[3])} * T3 + {f((1. / 5. - 1. / 4.) * a[4])} * T4")
    cg.add_line(f'{write_energy(f"gibbs0_RT[", active_len, expression, sp_thermo)}')
    if loop_gibbsexp:
        # cg.add_line(f"cfloat rcp_gibbs0_RT[{active_len}];", 1)
        cg.add_line(f"for(unsigned int i=0; i<{active_len}; ++i)", 1)
        cg.add_line(f"{{", 1)
        cg.add_line(f"gibbs0_RT[i] = exp(gibbs0_RT[i]);", 2)
        # cg.add_line(f"rcp_gibbs0_RT[i] = 1./gibbs0_RT[i];", 2)
        cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f'cfloat Cm = {"+".join([f"Ci[{specie}]" for specie in range(sp_len)])};', 1)
    cg.add_line(f"cfloat C0 = {f(const.one_atm / const.R)} * rcpT;", 1)
    cg.add_line(f"cfloat rcpC0 = {f(const.R / const.one_atm)} * T;", 1)
    if group_rxnunroll:
        cg.add_line(f"cfloat k, Rf, k_corr, Pr, logFcent, k_rev, Rr, cR;", 1)
    else:
        cg.add_line(f"cfloat k, Rf, k_inf, Pr, logFcent, k_rev, Rr, cR;", 1)
    cg.add_line(f"cfloat eff;", 1)
    cg.add_line(f"cfloat logPr, F, troe, troe_c, troe_n;", 1)
    cg.add_line("")

    if group_rxnunroll:
        rxn_grouped = {}
        for idx, r in enumerate(reactions):
            beta = r.rate_constant.temperature_exponent
            E_R = r.rate_constant.activation_temperature
            # Group reactions by their beta and E_R values
            if (beta == 0 and E_R == 0) or (E_R == 0 and beta in [-2, -1, 1, 2]):
                be_key = (beta, E_R, idx)
            else:
                be_key = (beta, E_R)
            if be_key not in rxn_grouped:
                rxn_grouped[be_key] = []
            rxn_grouped[be_key].append((idx, r))

        for be_key, grouped_reactions in rxn_grouped.items():
            first_idx = grouped_reactions[0][0]  # Index of the first reaction in the group
            cg.add_line(f"{write_reaction_grouped(grouped_reactions, first_idx, loop_gibbsexp)}")
        cg.add_line(f"}}")
    else:
        for idx, r in enumerate(reactions):
            cg.add_line(f"{write_reaction(idx, r, loop_gibbsexp)}")
        cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_enthalpy_unroll(file_name, output_dir, sp_len, sp_thermo):
    """
    Write the 'fenthalpy_RT.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_enthalpy_RT"
                f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                f"const cfloat rcpT,cfloat* h_RT)")
    cg.add_line(f"{{")
    expression = lambda a: (f'{f(a[0])} + {f(a[1] / 2)} * T + {f(a[2] / 3)} * T2 + '
                            f'{f(a[3] / 4)} * T3 + {f(a[4] / 5)} * T4 + {f(a[5])} * rcpT')
    cg.add_line(f'{write_energy(f"h_RT[", sp_len, expression, sp_thermo)}')
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_heat_capacity_unroll(file_name, output_dir, sp_len, sp_thermo):
    """
    Write the 'fheat_capacity_R.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_molar_heat_capacity_R"
                f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                f"const cfloat rcpT,cfloat* cp_R)")
    cg.add_line(f"{{")
    expression = lambda a: f'{f(a[0])} + {f(a[1])} * T + {f(a[2])} * T2 + {f(a[3])} * T3 + {f(a[4])} * T4'
    cg.add_line(f'{write_energy(f"cp_R[", sp_len, expression, sp_thermo)}')
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_conductivity_unroll(file_name, output_dir, transport_polynomials, sp_names):
    """
    Write the 'fconductivity.inc' file with unrolled loop specification.
    """

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivity"
                f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[])")
    cg.add_line(f"{{")
    cg.add_line(f"cfloat lambda_k, a = 0., b = 0.;", 1)
    cg.add_line(f"")
    for k, P in enumerate(transport_polynomials.conductivity):
        cg.add_line(f"//{sp_names[k]}", 1)
        cg.add_line(f"lambda_k = {evaluate_polynomial(P)};", 1)
        cg.add_line(f"a += nXi[{k}]*lambda_k;", 1)
        cg.add_line(f"b += nXi[{k}]/lambda_k;", 1)
        cg.add_line(f"")
    cg.add_line(f"return a/rcpMbar + rcpMbar/b;", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_viscosity_unroll(file_name, output_dir, group_vis, transport_polynomials, sp_names, sp_len, Mi):
    """
    Write the 'fviscosity.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat sq(cfloat x) {{ return x*x; }}")
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosity"
                f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[]) ")
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
                part = f"{cg.new_line}{cg.di}nXi[{j}]*sq({f(Va)}+{f(Va * sqrt(sqrt(Mi[j] / Mi[k])))}*r{j}*v)"
                denominator_parts.append(part)
            denominator = " + ".join(denominator_parts)
            cg.add_line("")
            cg.add_line(f"//{sp_names[k]}", 1)
            cg.add_line(f"v = {v_expr};", 1)
            cg.add_line(f"vis += nXi[{k}]*sq(v)/({denominator}{cg.new_line}{cg.si});", 1)
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
                cg.add_line(f"sum_{k} += nXi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))};", 2)
            cg.add_line(f"}}", 1)
        for j in [sp_len - 1]:
            cg.add_line(f"{{", 1)
            cg.add_line(f"cfloat r{j} = {f(1.)}/v{j};", 2)
            for k in range(sp_len):
                cg.add_line(f"sum_{k} += nXi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))}; "
                            f"/*rcp_*/sum_{k} = {f(1.)}/sum_{k};", 2)
            cg.add_line(f"}}", 1)
        cg.add_line("")
        cg.add_line(f"""return {('+' + cg.new_line).join(f"{cg.ti if k > 0 else ' '}nXi[{k}]*sq(v{k}) * /*rcp_*/sum_{k}"
                                                           for k in range(sp_len))};""", 1)
        cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_nonsym_unroll(file_name, output_dir, rcp_diffcoeffs,
                                         transport_polynomials, sp_names, sp_len, Mi):
    """
    Write the 'fdiffusivity.inc' file with unrolled loop specification
    and  computation of the full Dij matrix (non-symmetrical matrix assumption).
    """

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                f"cfloat nXi[], dfloat* out, unsigned int stride) ")
    cg.add_line(f"{{")
    for k in range(sp_len):
        cg.add_line(f"//{sp_names[k]}", 1)
        cg.add_line(f"out[{k}*stride+id] = scale * (1.0f - nekrk_molar_mass[{k}] * nXi[{k}]) / (", 1)
        if rcp_diffcoeffs:
            cg.add_line(
                f"""{('+' + cg.new_line).join(
                    f"{cg.di}nXi[{j}] * "
                    f"({evaluate_polynomial(transport_polynomials.diffusivity[k if k > j else j][j if k > j else k])})"
                    for j in list(range(k)) + list(range(k + 1, sp_len)))});""")
        else:
            cg.add_line(
                f"""{('+' + cg.new_line).join(
                    f"{cg.di}nXi[{j}] * "
                    f"(1/"
                    f"({evaluate_polynomial(transport_polynomials.diffusivity[k if k > j else j][j if k > j else k])}))"
                    for j in list(range(k)) + list(range(k + 1, sp_len)))});""")
        cg.add_line(f"")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_unroll(file_name, output_dir, rcp_diffcoeffs, transport_polynomials, sp_len, Mi):
    """
    Write the 'fdiffusivity.inc' file with unrolled loop specification.
    """
    cg = CodeGenerator()

    S = [''] * sp_len

    def mut(y, i, v):
        S[i] = v
        return y

    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                f"cfloat nXi[], dfloat* out, unsigned int stride) ")
    cg.add_line(f"{{")
    for k in range(sp_len):
        for j in range(k):
            if rcp_diffcoeffs:
                cg.add_line(
                    f"cfloat R{k}_{j} = {evaluate_polynomial(transport_polynomials.diffusivity[k][j])};", 1)
            else:
                cg.add_line(
                    f"cfloat R{k}_{j} = 1/({evaluate_polynomial(transport_polynomials.diffusivity[k][j])});", 1)
            cg.add_line(f"cfloat S{k}_{j} = {mut(f'{S[k]}+' if S[k] else '', k, f'S{k}_{j}')}nXi[{j}]*R{k}_{j};", 1)
            cg.add_line(f"cfloat S{j}_{k} = {mut(f'{S[j]}+' if S[j] else '', j, f'S{j}_{k}')}nXi[{k}]*R{k}_{j};", 1)
    for k in range(sp_len):
        cg.add_line(f"out[{k}*stride+id] = scale * (1.0f - {Mi[k]}f*nXi[{k}])/{S[k]};", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_rates_roll(file_name, output_dir, align_width, target, sp_thermo, sp_len, rxn, rxn_len):
    """
    Write the 'rates.inc'('frates.inc') file with rolled loop specification.
    The reaction mechanism is reordered and restructured to allow for more
    efficient memory alignment and vectorized loop calculation, which can be
    particularly beneficial for CPUs.
    """

    # Regroup rate constants to eliminate redundant operations
    A, beta, E_R = [], [], []
    ids_EB, ids_E0B0, ids_E0Bneg1, ids_E0Bneg2, ids_E0B1, ids_E0B2, ids_ErBr = [], [], [], [], [], [], []
    pos_ids_ErBr = []
    ids_old = []
    for i in range(rxn_len):
        ids_old.append(i)
        A.append(rxn[i].rate_constant.preexponential_factor)
        beta.append(rxn[i].rate_constant.temperature_exponent)
        E_R.append(rxn[i].rate_constant.activation_temperature)
        if beta[-1] == 0 and E_R[-1] == 0:
            ids_E0B0.append(i)
        elif beta[-1] == -1 and E_R[-1] == 0:
            ids_E0Bneg1.append(i)
        elif beta[-1] == -2 and E_R[-1] == 0:
            ids_E0Bneg2.append(i)
        elif beta[-1] == 1 and E_R[-1] == 0:
            ids_E0B1.append(i)
        elif beta[-1] == 2 and E_R[-1] == 0:
            ids_E0B2.append(i)
        else:
            if len(ids_EB) > 0:
                for j in range(len(ids_EB)):
                    if beta[-1] == beta[ids_old.index(ids_EB[j])] and \
                            E_R[-1] == E_R[ids_old.index(ids_EB[j])]:
                        ids_ErBr.append(i)
                        pos_ids_ErBr.append(j)
                        break
                else:
                    ids_EB.append(i)
            else:
                ids_EB.append(i)
        # Rate constants for k0
        # Indices for k0 are number of reactions + reaction index
        if hasattr(rxn[i], 'k0'):
            ids_old.append(rxn_len + i)
            A.append(rxn[i].k0.preexponential_factor)
            beta.append(rxn[i].k0.temperature_exponent)
            E_R.append(rxn[i].k0.activation_temperature)
            if beta[-1] == 0 and E_R[-1] == 0:
                ids_E0B0.append(rxn_len + i)
            elif beta[-1] == -1 and E_R[-1] == 0:
                ids_E0Bneg1.append(rxn_len + i)
            elif beta[-1] == -2 and E_R[-1] == 0:
                ids_E0Bneg2.append(rxn_len + i)
            elif beta[-1] == 1 and E_R[-1] == 0:
                ids_E0B1.append(rxn_len + i)
            elif beta[-1] == 2 and E_R[-1] == 0:
                ids_E0B2.append(rxn_len + i)
            else:
                if len(ids_EB) > 0:
                    for j in range(len(ids_EB)):
                        if beta[-1] == beta[ids_old.index(ids_EB[j])] and \
                                E_R[-1] == E_R[ids_old.index(ids_EB[j])]:
                            ids_ErBr.append(rxn_len + i)
                            pos_ids_ErBr.append(j)
                            break
                    else:
                        ids_EB.append(rxn_len + i)
                else:
                    ids_EB.append(rxn_len + i)
    # Group repeated constants for better vectorization
    unique_pos_ids_ErBr = []
    for i in range(len(pos_ids_ErBr)):
        if pos_ids_ErBr[i] not in pos_ids_ErBr[:i]:
            unique_pos_ids_ErBr.append(pos_ids_ErBr[i])
    # Move repeated constants that need to be calculated at the end
    ids_ErBr_values = [ids_EB[i] for i in pos_ids_ErBr]
    new_ids_EB_s, new_ids_EB_e = [], []
    for u in range(len(unique_pos_ids_ErBr)):
        for i in range(len(ids_EB)):
            if i == unique_pos_ids_ErBr[u]:
                new_ids_EB_e.append(ids_EB[i])
                break
    for i in range(len(ids_EB)):
        if i not in unique_pos_ids_ErBr:
            new_ids_EB_s.append(ids_EB[i])
    assert len(ids_EB) == (len(new_ids_EB_s) + len(new_ids_EB_e))
    ids_EB = new_ids_EB_s + new_ids_EB_e
    ids_rep = []
    new_pos_ids_ErBr_s, new_pos_ids_ErBr_e = [], []
    new_ids_ErBr_values_s, new_ids_ErBr_values_e = [], []
    for i in range(len(pos_ids_ErBr)):
        if pos_ids_ErBr[i] in pos_ids_ErBr[:i]:
            ids_rep.append(i)
            new_pos_ids_ErBr_e.append(pos_ids_ErBr[i])
            new_ids_ErBr_values_e.append(ids_ErBr_values[i])
        else:
            new_pos_ids_ErBr_s.append(pos_ids_ErBr[i])
            new_ids_ErBr_values_s.append(ids_ErBr_values[i])
    pos_ids_ErBr = new_pos_ids_ErBr_s + new_pos_ids_ErBr_e
    ids_ErBr_values = new_ids_ErBr_values_s + new_ids_ErBr_values_e
    new_ids_ErBr_s, new_ids_ErBr_e = [], []
    for i in range(len(ids_ErBr)):
        if i in ids_rep:
            new_ids_ErBr_e.append(ids_ErBr[i])
        else:
            new_ids_ErBr_s.append(ids_ErBr[i])
    ids_ErBr = new_ids_ErBr_s + new_ids_ErBr_e

    # Group indices
    ids_E0Bsmall = ids_E0Bneg2 + ids_E0Bneg1 + ids_E0B1 + ids_E0B2
    ids_new = ids_EB + ids_E0B0 + ids_E0Bsmall + ids_ErBr
    assert len(ids_old) == len(ids_new)
    # Rearrange constants
    A_new, beta_new, E_R_new = [], [], []
    for i in range(len(ids_new)):
        A_new.append(A[ids_old.index(ids_new[i])])
        beta_new.append(beta[ids_old.index(ids_new[i])])
        E_R_new.append(E_R[ids_old.index(ids_new[i])])
    beta_new_reduced = beta_new[:len(ids_EB)]
    E_R_new_reduced = E_R_new[:len(ids_EB)]
    # Precompute log(A) for constants that need to be computed
    for i in range(len(ids_EB)):
        A_new[i] = math.log(A_new[i])
    # Correct log(A) for repeated constants
    for i in range(len(unique_pos_ids_ErBr)):
        idx_re = len(ids_new) - len(pos_ids_ErBr) + i
        idx_a_re = len(ids_EB) - len(unique_pos_ids_ErBr) + i
        A_new[idx_re] = A_new[idx_re] / math.exp(A_new[idx_a_re])
    if (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) > 0:
        for i in range(len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)):
            idx_unique_re = len(ids_new) - (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) + i
            idx_unique_a_re = ids_EB.index(ids_ErBr_values[len(unique_pos_ids_ErBr) + i])
            A_new[idx_unique_re] = A_new[idx_unique_re] / math.exp(A_new[idx_unique_a_re])

    def set_k():
        cg = CodeGenerator()
        if len(ids_E0B0) > 0:
            cg.add_line(f'// {len(ids_E0B0)} rate constants with E_R = 0 and beta = 0 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0B0)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'k[i+{len(ids_EB)}] = A[i+{len(ids_EB)}];', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0Bneg2) > 0:
            start_ids_E0Bneg2 = len(ids_EB) + len(ids_E0B0)
            cg.add_line(f'// {len(ids_E0Bneg2)} rate constants with E_R = 0 and beta = -2 ', 1)
            cg.add_line(f'cfloat rcpT_2 = rcpT*rcpT;', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0Bneg2)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'k[i+{start_ids_E0Bneg2}] = A[i+{start_ids_E0Bneg2}]*rcpT_2;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0Bneg1) > 0:
            start_ids_E0Bneg1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2)
            cg.add_line(f'// {len(ids_E0Bneg1)} rate constants with E_R = 0 and beta = -1 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0Bneg1)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'k[i+{start_ids_E0Bneg1}] = A[i+{start_ids_E0Bneg1}]*rcpT;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0B1) > 0:
            start_ids_E0B1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1)
            cg.add_line(f'// {len(ids_E0B1)} rate constants with E_R = 0 and beta = 1 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0B1)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'k[i+{start_ids_E0B1}] = A[i+{start_ids_E0B1}]*T;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0B2) > 0:
            start_ids_E0B2 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1) + len(ids_E0B1)
            cg.add_line(f'// {len(ids_E0B2)} rate constants with E_R = 0 and beta = 2 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0B2)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'k[i+{start_ids_E0B2}] = A[i+{start_ids_E0B2}]*T2;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_ErBr) > 0:
            start_ids_ErBr = len(ids_new) - len(ids_ErBr)
            cg.add_line(
                f'// {len(ids_ErBr)} rate constants with E_R and '
                f'beta the same as for other rate constants which have already been computed ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(unique_pos_ids_ErBr)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(
                f'k[i+{start_ids_ErBr}] = A[i+{start_ids_ErBr}]*k[i+{len(ids_EB) - len(unique_pos_ids_ErBr)}];', 2)
            cg.add_line(f'}}', 1)
            if (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) > 0:
                for i in range(len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)):
                    start_ids_ErBr_rep = len(ids_new) - (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) + i
                    ids_k_rep = ids_EB.index(ids_ErBr_values[len(unique_pos_ids_ErBr) + i])
                    cg.add_line(f'k[{start_ids_ErBr_rep}]=A[{start_ids_ErBr_rep}]*k[{ids_k_rep}];', 1)
        return cg.get_code()

    ids_eff = []
    dic_unique_eff = {}

    def reodrder_eff(r):
        cg = CodeGenerator()
        dic_eff = {}
        count = 0
        for i in range(len(r)):
            if hasattr(r[i], 'efficiencies'):
                ids_eff.append(i)
                dic_eff[count] = r[i].efficiencies
                count += 1
        if count > 0:
            unique_ids_eff = []
            unique_count = 0
            for i in range(len(ids_eff)):
                test = True
                for j in range(0, i):
                    if dic_eff[j] == dic_eff[i] and i != 0:
                        dic_unique_eff[i] = dic_unique_eff[j]
                        test = False
                        break
                if test:
                    dic_unique_eff[i] = unique_count
                    unique_ids_eff.append(ids_eff[i])
                    unique_count += 1
            for i in range(unique_count):
                ci = []
                for specie, efficiency in enumerate(r[unique_ids_eff[i]].efficiencies):
                    if efficiency != 1.:
                        if efficiency == 2:
                            ci.append(f"Ci[{specie}]")
                        elif efficiency == 0:
                            ci.append(f"-Ci[{specie}]")
                        else:
                            ci.append(f"{f(efficiency - 1.)}*Ci[{specie}]")
                cg.add_line(f"cfloat eff{i} = Cm + {'+'.join(ci)};", 1)

            return cg.get_code()
        else:
            cg.add_line("")
            return cg.get_code()

    # Correct k based on reaction type
    ids_er_rxn, ids_3b_rxn, ids_pd_rxn, ids_troe_rxn, ids_sri_rxn = [], [], [], [], []
    for i in range(rxn_len):
        if rxn[i].type == 'elementary' or rxn[i].type == 'reversible':
            ids_er_rxn.append(i)
        elif rxn[i].type == 'three-body':
            ids_3b_rxn.append(i)
        elif rxn[i].type == 'pressure-modification':
            ids_pd_rxn.append(i)
        elif rxn[i].type == 'Troe':
            ids_troe_rxn.append(i)
        elif rxn[i].type == 'SRI':
            ids_sri_rxn.append(i)
        else:
            continue

    def corr_k(r):
        # Three-body reactions
        cg_tb = CodeGenerator()
        if len(ids_3b_rxn) > 0:
            cg_tb.add_line(f"//Correct k for three-body reactions", 1)
            for i in ids_3b_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_tb.add_line(
                        f"k[{ids_new.index(i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_tb.add_line(
                        f"k[{ids_new.index(i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_tb.add_line(
                        f"k[{ids_new.index(i)}] *= Cm;", 1)

        # Pressure-dependent reactions
        cg_pd = CodeGenerator()
        if len(ids_pd_rxn) > 0:
            cg_pd.add_line("")
            cg_pd.add_line(f"//Correct k for pressure-dependent reactions", 1)
            for i in ids_pd_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_pd.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_pd.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_pd.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_pd.add_line(
                    f"k[{ids_new.index(rxn_len + i)}] /= "
                    f"(1+ k[{ids_new.index(rxn_len + i)}]/(k[{ids_new.index(i)}]+ CFLOAT_MIN));", 1)

        # Troe reactions
        cg_troe = CodeGenerator()
        if len(ids_troe_rxn) > 0:
            cg_troe.add_line("")
            cg_troe.add_line(f"//Correct k for troe reactions", 1)
            for i in ids_troe_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_troe.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_troe.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_troe.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_troe.add_line(
                    f"k[{ids_new.index(rxn_len + i)}] /= (k[{ids_new.index(i)}] + CFLOAT_MIN);", 1)
            troe_A, rcp_troe_T1, troe_T2, rcp_troe_T3 = [], [], [], []
            for i in ids_troe_rxn:
                troe_A.append(r[i].troe.A)
                rcp_troe_T1.append(1 / (r[i].troe.T1+FLOAT_MIN))
                troe_T2.append(r[i].troe.T2)
                rcp_troe_T3.append(1 / (r[i].troe.T3+FLOAT_MIN))
            ids_troe = []
            for i in ids_troe_rxn:
                ids_troe.append(ids_new.index(rxn_len + i))
            cg_troe.add_line("")
            cg_troe.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat troe_A[{len(ids_troe_rxn)}] = {{{f_sci_not(troe_A)}}};", 1)
            cg_troe.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat rcp_troe_T1[{len(ids_troe_rxn)}] = {{{f_sci_not(rcp_troe_T1)}}};", 1)
            cg_troe.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat troe_T2[{len(ids_troe_rxn)}] = "
                f"{{{f_sci_not([i if i < float('inf') else -1 for i in troe_T2])}}};"
                if not (all([i == float('inf') for i in troe_T2])) else '', 1)
            cg_troe.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat rcp_troe_T3[{len(ids_troe_rxn)}] = {{{f_sci_not(rcp_troe_T3)}}};", 1)
            cg_troe.add_line("")
            cg_troe.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} int "
                f"ids_troe[{len(ids_troe_rxn)}] = {str(ids_troe).replace('[', '{').replace(']', '}')};", 1)
            cg_troe.add_line(
                f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                f"logFcent[{len(ids_troe_rxn)}];", 1)
            cg_troe.add_line(f"for(unsigned int i = 0; i<{len(ids_troe_rxn)}; ++i)", 1)
            cg_troe.add_line(f"{{", 1)
            if all([i == float('inf') for i in troe_T2]):
                cg_troe.add_line(
                    f"logFcent[i] = log10(({f(1.)} - troe_A[i])*exp(-T*rcp_troe_T3[i]) + "
                    f"troe_A[i]*exp(-T*rcp_troe_T1[i]));",
                    2)
            elif all([i < float('inf') for i in troe_T2]):
                cg_troe.add_line(
                    f"logFcent[i] = log10(({f(1.)} - troe_A[i])*exp(-T*rcp_troe_T3[i]) + "
                    f"troe_A[i]*exp(-T*rcp_troe_T1[i]) + exp(-troe_T2[i]*rcpT));",
                    2)
            else:
                cg_troe.add_line(f"cfloat term3 = (troe_T2[i] != -1) ? exp(-troe_T2[i] * rcpT) : {f(0.)};", 2)
                cg_troe.add_line(
                    f"logFcent[i] = log10(({f(1.)} - troe_A[i])*exp(-T*rcp_troe_T3[i]) + "
                    f"troe_A[i]*exp(-T*rcp_troe_T1[i]) + term3);",
                    2)
            cg_troe.add_line(f"}}", 1)
            cg_troe.add_line(f"for(unsigned int i = 0; i<{len(ids_troe_rxn)}; ++i)", 1)
            cg_troe.add_line(f"{{", 1)
            cg_troe.add_line(
                f"cfloat troe_c = {f(-0.4)} - {f(0.67)} * logFcent[i];", 2)
            cg_troe.add_line(
                f"cfloat troe_n = {f(0.75)} - {f(1.27)} * logFcent[i];", 2)
            cg_troe.add_line(
                f"cfloat logPr = log10(k[ids_troe[i]] + CFLOAT_MIN);", 2)
            cg_troe.add_line(
                f"cfloat troe = (troe_c + logPr)/(troe_n - {f(0.14)}*(troe_c + logPr)+CFLOAT_MIN);", 2)
            cg_troe.add_line(
                f"cfloat F = pow(10, logFcent[i]/({f(1.0)} + troe*troe));", 2)
            cg_troe.add_line(f"k[ids_troe[i]] /= ({f(1.)}+k[ids_troe[i]]);", 2)
            cg_troe.add_line(f"k[ids_troe[i]] *= F;", 2)
            cg_troe.add_line(f"}}", 1)
            for i in ids_troe_rxn:
                cg_troe.add_line(
                    f"k[{ids_new.index(rxn_len + i)}] *= k[{ids_new.index(i)}];", 1)

        # SRI reaction
        cg_sri = CodeGenerator()
        if len(ids_sri_rxn) > 0:
            cg_sri.add_line("")
            cg_sri.add_line(f"//Correct k for SRI reactions", 1)
            for i in ids_sri_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_sri.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_sri.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_sri.add_line(
                        f"k[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_sri.add_line(
                    f"k[{ids_new.index(rxn_len + i)}] /= (k[{ids_new.index(i)}] + CFLOAT_MIN);", 1)
            sri_A, sri_B, rcp_sri_C, sri_D, sri_E = [], [], [], [], []
            for i in ids_sri_rxn:
                sri_A.append(r[i].sri.A)
                sri_B.append(r[i].sri.B)
                rcp_sri_C.append(1/(r[i].sri.C + FLOAT_MIN))
                sri_D.append(r[i].sri.D)
                sri_E.append(r[i].sri.E)
            ids_sri = []
            for i in ids_sri_rxn:
                ids_sri.append(ids_new.index(rxn_len + i))
            cg_sri.add_line("")
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat sri_A[{len(ids_sri_rxn)}] = {{{f_sci_not(sri_A)}}};", 1)
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat sri_B[{len(ids_sri_rxn)}] = {{{f_sci_not(sri_B)}}};", 1)
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat rcp_sri_C[{len(ids_sri_rxn)}] = {{{f_sci_not(rcp_sri_C)}}};" if not (
                    any([i == float('inf') for i in rcp_sri_C])) else '', 1)
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat sri_D[{len(ids_sri_rxn)}] = {{{f_sci_not(sri_D)}}};", 1)
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                f"cfloat sri_E[{len(ids_sri_rxn)}] = {{{f_sci_not(sri_E)}}};", 1)
            cg_sri.add_line("")
            cg_sri.add_line(
                f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} int "
                f"ids_sri[{len(ids_sri_rxn)}] = {str(ids_sri).replace('[', '{').replace(']', '}')};", 1)
            cg_sri.add_line(f"for(unsigned int i = 0; i<{len(ids_sri_rxn)}; ++i)", 1)
            cg_sri.add_line(f"{{", 1)
            cg_sri.add_line(
                f"cfloat logPr = log10(k[ids_sri[i]] + CFLOAT_MIN);", 2)
            cg_sri.add_line(f"cfloat F = sri_D[i]*pow(T, sri_E[i])*"
                                  f"pow(sri_A[i]*exp(-sri_B[i]*rcpT)+exp(-rcp_sri_C[i]*T), 1./(1.+logPr*logPr));", 2)
            cg_sri.add_line(f"k[ids_sri[i]] /= (1.+k[ids_sri[i]]);", 2)
            cg_sri.add_line(f"k[ids_sri[i]] *= F;", 2)
            cg_sri.add_line(f"}}", 1)
            for i in ids_sri_rxn:
                cg_sri.add_line(
                    f"k[{ids_new.index(rxn_len + i)}]*= k[{ids_new.index(i)}];", 1)

        # Combine all reactions
        reaction_corr = cg_tb.get_code()  + cg_pd.get_code() + cg_troe.get_code() + cg_sri.get_code()
        return reaction_corr

    # Reorder reactions back to original
    def reorder_k(r):
        cg = CodeGenerator()
        for i in range(len(r)):
            if hasattr(rxn[i], 'k0'):
                cg.add_line(f"k[{i}] = tmp[{ids_new.index(rxn_len + i)}];", 1)
            else:
                cg.add_line(f"k[{i}] = tmp[{ids_new.index(i)}];", 1)
        return cg.get_code()

    # Compute the gibbs energy
    (a0, a1, a2, a3, a4, a5, a6,
     ids_gibbs_new, len_unique_temp_splits, unique_temp_split) = get_thermo_coeffs('g_RT', sp_thermo, sp_len)

    # Reciprocal gibbs energy
    repeated_rcp_gibbs = [0] * len(sp_thermo)
    for i in range(rxn_len):
        for j, net in enumerate(rxn[i].net):
            if net < 0:
                repeated_rcp_gibbs[j] += 1
    ids_rcp_gibbs = [i for i, x in enumerate(repeated_rcp_gibbs) if x > 2]
    ids_rcp_gibbs_reordered = [ids_gibbs_new.index(i) for i in ids_rcp_gibbs]

    # Compute k_rev only for reversible reactions
    ids_krev_nz = []
    for i in range(rxn_len):
        if rxn[i].direction != 'irreversible':
            ids_krev_nz.append(i)

    # Compute reverse rates
    def compute_k_rev(r):
        cg = CodeGenerator()
        for i in range(len(r)):
            pow_C0_sum_net = '*'.join(["C0" if r[i].sum_net < 0 else 'rcpC0'] * abs(-r[i].sum_net))
            gibbs_terms = []
            for j, net in enumerate(rxn[i].net):
                if net < 0:
                    for n in range(abs(net)):
                        if j in ids_rcp_gibbs:
                            gibbs_terms.append(f"rcp_gibbs0_RT[{ids_rcp_gibbs.index(j)}]")
                        else:
                            gibbs_terms.append(f"(1./gibbs0_RT[{ids_gibbs_new.index(j)}])")
                if net > 0:
                    for o in range(net):
                        gibbs_terms.append(f"gibbs0_RT[{ids_gibbs_new.index(j)}]")
            if r[i].direction != 'irreversible':
                cg.add_line(
                    f"k_rev[{ids_krev_nz.index(i)}] = "
                    f"{'*'.join(gibbs_terms)}{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};", 1)
        return cg.get_code()

    # Compute reaction rates
    def compute_rates(r):
        phaseSpace = lambda reagents: '*'.join(
            '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
            coefficient != 0.)

        cg = CodeGenerator()
        for i in range(len(r)):
            cg.add_line(f"//{i + 1}: {r[i].description}", 1)
            if r[i].type == 'irreversible' or r[i].direction == 'irreversible':
                if hasattr(r[i], 'k0'):
                    cg.add_line(f"cR = k[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)});", 1)
                else:
                    cg.add_line(f"cR = k[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)});", 1)
            else:
                if hasattr(r[i], 'k0'):
                    cg.add_line(
                        f"cR = k[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)}-"
                        f"k_rev[{ids_krev_nz.index(i)}]*{phaseSpace(r[i].products)});", 1)
                else:
                    cg.add_line(
                        f"cR = k[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)}-"
                        f"k_rev[{ids_krev_nz.index(i)}]*{phaseSpace(r[i].products)});", 1)
            cg.add_line(f"#ifdef DEBUG")
            cg.add_line(f"printf(\"{i + 1}: %+.15e\\n\", cR);", 1)
            cg.add_line(f"#endif")
            for specie, net in enumerate(r[i].net):
                if net != 0:  # Only generate code for non-zero net changes
                    cg.add_line(f"rates[{specie}] += {imul(net, 'cR')};", 1)
            cg.add_line(f"")
        return cg.get_code()

    #############
    # Write file
    #############

    cg = CodeGenerator()
    cg.add_line(f'#include <math.h>')
    cg.add_line(f"#define __NEKRK_EXP_OVERFLOW__(x) __NEKRK_MIN_CFLOAT(CFLOAT_MAX, exp(x))")
    cg.add_line(f'__NEKRK_DEVICE__ __NEKRK_INLINE__ void nekrk_species_rates'
                 f'(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4,'
                 f' const cfloat rcpT, const cfloat Ci[], cfloat* rates) ')
    cg.add_line(f'{{')
    cg.add_line(f"// Regrouping of rate constants to eliminate redundant operations", 1)
    var_str = ['A', 'beta', 'E_R']
    var = [A_new, beta_new_reduced, E_R_new_reduced]
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"k[{len(ids_new)}];", 1)
    cg.add_line(f"// Compute the {len(ids_EB)} rate constants for which an evaluation is necessary", 1)
    cg.add_line(f"for(unsigned int i=0; i<{len(ids_EB)}; ++i)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"cfloat blogT = beta[i]*lnT;", 2)
    cg.add_line(f"cfloat E_RT = E_R[i]*rcpT;", 2)
    cg.add_line(f"cfloat diff = blogT - E_RT;", 2)
    cg.add_line(f"k[i] = exp(A[i] + diff);", 2)
    cg.add_line(f"}}", 1)
    cg.add_line(f"{set_k()}")
    cg.add_line("")
    cg.add_line(f"// Correct rate constants based on reaction type", 1)
    cg.add_line(f"cfloat Cm = 0;", 1)
    cg.add_line(f"for(unsigned int i=0; i<{sp_len}; ++i)", 1)
    cg.add_line(f"Cm += Ci[i];", 2)
    cg.add_line("")
    cg.add_line(f"{reodrder_eff(rxn)}")
    cg.add_line("")
    cg.add_line(f"{corr_k(rxn)}")
    cg.add_line("")
    cg.add_line(f"// Compute the gibbs energy", 1)
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    var = [a0, a1, a2, a3, a4, a5, a6]
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"gibbs0_RT[{sp_len}];", 1)
    cg.add_line(f"{get_thermo_prop('g_RT', unique_temp_split, len_unique_temp_splits)}")
    cg.add_line(f"// Group the gibbs exponentials", 1)
    cg.add_line(f"for(unsigned int i=0; i<{sp_len}; ++i)", 1)
    cg.add_line(f"gibbs0_RT[i] = exp(gibbs0_RT[i]);", 2)
    cg.add_line("")
    cg.add_line(f"// Compute the reciprocal of the gibbs exponential", 1)
    cg.add_line(f"{f'alignas({align_width}) static constexpr' if target=='c++17' else 'const'} "
                 f"int ids_rcp_gibbs[{len(ids_rcp_gibbs_reordered)}] = "
                 f"{str(ids_rcp_gibbs_reordered).replace('[', '{').replace(']', '}')};", 1)
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                f"rcp_gibbs0_RT[{len(ids_rcp_gibbs_reordered)}];", 1)
    cg.add_line(f"for(unsigned int i=0; i<{len(ids_rcp_gibbs_reordered)}; ++i)", 1)
    cg.add_line(f"rcp_gibbs0_RT[i] = {f(1.)}/gibbs0_RT[ids_rcp_gibbs[i]];", 2)
    cg.add_line("")
    cg.add_line(f"// Compute reverse rates", 1)
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"k_rev[{len(ids_krev_nz)}]; ", 1)
    cg.add_line(f"cfloat C0 = {f(const.one_atm / const.R)} * rcpT;", 1)
    cg.add_line(f"cfloat rcpC0 = {f(const.R / const.one_atm)} * T;", 1)
    cg.add_line(f"{compute_k_rev(rxn)}")
    cg.add_line("")
    cg.add_line(f"// Compute the reaction rates", 1)
    cg.add_line(f"cfloat cR;", 1)
    cg.add_line(f"{compute_rates(rxn)}")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_enthalpy_roll(file_name, output_dir, align_width, target, sp_thermo, sp_len):
    """
    Write the 'fenthalpy_RT.inc' file with rolled loop specification.
    """

    (a0, a1, a2, a3, a4, a5, a6,
     ids_thermo_new, len_unique_temp_splits, unique_temp_split) = get_thermo_coeffs('h_RT', sp_thermo, sp_len)
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
    var = [a0, a1, a2, a3, a4, a5]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_enthalpy_RT(const cfloat lnT, const cfloat T, "
                f"const cfloat T2, const cfloat T3, const cfloat T4, const cfloat rcpT,cfloat h_RT[]) ")
    cg.add_line(f"{{")
    cg.add_line(f"//Integration coefficients", 1)
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{get_thermo_prop('h_RT', unique_temp_split, len_unique_temp_splits)}")
    cg.add_line(f"{reorder_thermo_prop('h_RT', unique_temp_split, ids_thermo_new, sp_len)}")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_heat_capacity_roll(file_name, output_dir, align_width, target, sp_thermo, sp_len):
    """
    Write the 'fheat_capacity_R.inc' file with rolled loop specification.
    """

    (a0, a1, a2, a3, a4, a5, a6,
     ids_thermo_new, len_unique_temp_splits, unique_temp_split) = get_thermo_coeffs('cp_R', sp_thermo, sp_len)
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4']
    var = [a0, a1, a2, a3, a4]

    cg = CodeGenerator()
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_molar_heat_capacity_R"
                f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                f"const cfloat rcpT,cfloat cp_R[]) ")
    cg.add_line(f"{{")
    cg.add_line(f"//Integration coefficients", 1)
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{get_thermo_prop('cp_R', unique_temp_split, len_unique_temp_splits)}")
    cg.add_line(f"{reorder_thermo_prop('cp_R', unique_temp_split, ids_thermo_new, sp_len)}")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_conductivity_roll(file_name, output_dir, align_width, target, transport_polynomials, sp_len):
    """
    Write the 'fconductivity.inc' file with rolled loop specification.
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
                f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[])")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"cfloat lambda_k, sum1=0., sum2=0.;", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"lambda_k = b0[k] + b1[k]*lnT + b2[k]*lnT2 + b3[k]*lnT3 + b4[k]*lnT4;", 2)
    cg.add_line(f"sum1 += nXi[k]*lambda_k;", 2)
    cg.add_line(f"sum2 += nXi[k]/lambda_k;", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"return sum1/rcpMbar + rcpMbar/sum2;", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_viscosity_roll(file_name, output_dir, align_width, target, transport_polynomials, sp_len, Mi):
    """
    Write the 'fviscosity.inc' file with rolled loop specification.
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
                f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[]) ")
    cg.add_line(f"{{")
    cg.add_line(f"{write_const_expression(align_width, target, True, var1_str, var1)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} mue[{sp_len}];", 1)
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} rcp_mue[{sp_len}];", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"mue[k] = a0[k] + a1[k]*lnT + a2[k]*lnT2 + a3[k]*lnT3 + a4[k]*lnT4;", 2)
    cg.add_line(f"rcp_mue[k] = 1./mue[k];", 2)
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
    cg.add_line(f"cfloat sqrt_Phi_kj = C1[idx] + C2[idx]*mue[k]*rcp_mue[j];", 3)
    cg.add_line(f"sums[k] += nXi[j]*sqrt_Phi_kj*sqrt_Phi_kj;", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"cfloat vis = 0.;", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"vis += nXi[k]*mue[k]*mue[k]/sums[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"return vis;", 1)
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_nonsym_roll(file_name, output_dir, align_width, target, rcp_diffcoeffs,
                                       transport_polynomials, sp_len, Mi):
    """
    Write the 'fdiffusivity.inc ' file with rolled loop specification
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
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                 f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                 f"cfloat nXi[], dfloat* out, unsigned int stride) ")
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
    cg.add_line(f"sums[k] += nXi[j]*rcp_Dkj;", 4)
    cg.add_line(f"}}", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{write_const_expression(align_width, target, True, 'Wi', Mi)}")
    cg.add_line("")
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"unsigned int idx = k*stride+id;", 2)
    cg.add_line(f"out[idx] = scale * (1.0f - Wi[k]*nXi[k])/sums[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def write_file_diffusivity_roll(file_name, output_dir, align_width, target, rcp_diffcoeffs,
                                transport_polynomials, sp_len, Mi):
    """
    Write the 'fdiffusivity.inc' file with rolled loop specification.
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
    cg.add_line(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                f"cfloat nXi[], dfloat* out, unsigned int stride) ")
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
    cg.add_line(f"sums1[k] += nXi[j]*rcp_Dkj;", 3)
    cg.add_line(f"sums2[j] += nXi[k]*rcp_Dkj;", 3)
    cg.add_line(f"}}", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} sums[{sp_len}];", 1)
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"sums[k] = sums1[k] + sums2[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"{write_const_expression(align_width, target, True, 'Wi', Mi)}")
    cg.add_line(f"")
    cg.add_line(f"for(unsigned int k=0; k<{sp_len}; k++)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"unsigned int idx = k*stride+id;", 2)
    cg.add_line(f"out[idx] = scale * (1.0f - Wi[k]*nXi[k])/sums[k];", 2)
    cg.add_line(f"}}", 1)
    cg.add_line("")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0


def generate_files(mech_file=None,
                   output_dir=None,
                   single_precision=None,
                   header_only=False,
                   unroll_loops=False,
                   align_width=64,
                   target=None,
                   loop_gibbsexp=False,
                   group_rxnunroll=False,
                   transport=True,
                   group_vis=False,
                   nonsymDij=False,
                   rcp_diffcoeffs=False
                   ):
    """
    Generate the production rates, thermodynamic and transport properties
    subroutine files.
    """

    # Load mechanism model
    model = read_mechanism_yaml(mech_file)

    # Initialize reactions to find inert species
    species_names_init = [i['name'] for i in model['species']]
    reactions_init = [get_reaction_from_model(species_names_init, model['units'], reaction)
                      for reaction in model['reactions']]
    (inert_sp, active_sp) = partition(
        lambda specie: any([reaction.net[species_names_init.index(specie['name'])] != 0
                            for reaction in reactions_init]), model['species'])
    # Load species and reactions with new indexing (inert species at the end)
    species = get_species_from_model(active_sp+inert_sp, rcp_diffcoeffs)
    species_names = species.sp_names
    reactions = [get_reaction_from_model(species_names, model['units'], reaction)
                 for reaction in model['reactions']]
    species_len = len(species_names)
    active_sp_len = len(active_sp)
    reactions_len = len(reactions)
    Mi = species.Mi

    # Load transport polynomials
    if transport and not header_only:
        transport_polynomials = species.transport_polynomials()
        # Includes division by 2
        transport_polynomials.conductivity = [[1/2 * p for p in P] for P in
                                                 transport_polynomials.conductivity]
        transport_polynomials.viscosity = [[p for p in P] for P in
                                                transport_polynomials.viscosity]
        if rcp_diffcoeffs:
            # The reciprocal polynomial is evaluated
            transport_polynomials.diffusivity = [
                [[const.R * p for p in P]
                 for k, P in enumerate(row)] for row in transport_polynomials.diffusivity]
        else:
            transport_polynomials.diffusivity = [
                [[1/const.R * p for p in P]
                 for k, P in enumerate(row)] for row in transport_polynomials.diffusivity]

    #########################
    # Write subroutine files
    #########################

    # File names
    mech_file = 'mech.h'
    if single_precision:
        set_precision('FP32')
        rates_file = 'frates.inc'
        enthalpy_file = 'fenthalpy_RT.inc'
        heat_capacity_file = 'fheat_capacity_R.inc'
        conductivity_file = 'fconductivity.inc'
        viscosity_file = 'fviscosity.inc'
        diffusivity_file = 'fdiffusivity.inc'
    else:
        set_precision('FP64')
        rates_file = 'rates.inc'
        enthalpy_file = 'enthalpy_RT.inc'
        heat_capacity_file = 'heat_capacity_R.inc'
        conductivity_file = 'conductivity.inc'
        viscosity_file = 'viscosity.inc'
        diffusivity_file = 'diffusivity.inc'

    if header_only:
        write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
    else:
        write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
        if unroll_loops:  # Unrolled code
            write_file_rates_unroll(rates_file, output_dir, loop_gibbsexp, group_rxnunroll,
                                    reactions, active_sp_len, species_len, species.thermo)
            write_file_enthalpy_unroll(enthalpy_file, output_dir,
                                       species_len, species.thermo)
            write_file_heat_capacity_unroll(heat_capacity_file, output_dir,
                                            species_len, species.thermo)
            if transport:
                write_file_conductivity_unroll(conductivity_file, output_dir,
                                               transport_polynomials, species_names)
                write_file_viscosity_unroll(viscosity_file, output_dir, group_vis,
                                            transport_polynomials, species_names, species_len, Mi)
                if nonsymDij:
                    write_file_diffusivity_nonsym_unroll(diffusivity_file, output_dir, rcp_diffcoeffs,
                                                         transport_polynomials, species_names, species_len, Mi)
                else:
                    write_file_diffusivity_unroll(diffusivity_file, output_dir, rcp_diffcoeffs,
                                                  transport_polynomials, species_len, Mi)
        else:  # Rolled code
            write_file_rates_roll(rates_file, output_dir, align_width, target,
                                  species.thermo, species_len, reactions, reactions_len)
            write_file_enthalpy_roll(enthalpy_file, output_dir, align_width, target,
                                     species.thermo, species_len)
            write_file_heat_capacity_roll(heat_capacity_file, output_dir, align_width, target,
                                          species.thermo, species_len)
            if transport:
                write_file_conductivity_roll(conductivity_file, output_dir, align_width, target,
                                             transport_polynomials, species_len)
                write_file_viscosity_roll(viscosity_file, output_dir, align_width, target,
                                          transport_polynomials, species_len, Mi)
                if nonsymDij:
                    write_file_diffusivity_nonsym_roll(diffusivity_file, output_dir,
                                                       align_width, target, rcp_diffcoeffs,
                                                       transport_polynomials, species_len, Mi)
                else:
                    write_file_diffusivity_roll(diffusivity_file, output_dir,
                                                align_width, target, rcp_diffcoeffs,
                                                transport_polynomials, species_len, Mi)

    return 0


if __name__ == "__main__":
    args = get_parser()

    generate_files(mech_file=args.mechanism,
                   output_dir=args.output,
                   single_precision=args.single_precision,
                   header_only=args.header_only,
                   unroll_loops=args.unroll_loops,
                   align_width=args.align_width,
                   target=args.target,
                   loop_gibbsexp=args.loop_gibbsexp,
                   group_rxnunroll=args.group_rxnunroll,
                   transport=args.transport,
                   group_vis=args.group_vis,
                   nonsymDij=args.nonsymDij,
                   rcp_diffcoeffs=args.fit_rcpdiffcoeffs
                   )
