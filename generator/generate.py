#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2023 Matthias Fauconneau, Danciu Bogdan, 
#                    Christos Frouzakis, Stefan Kerkemeier
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Generate source code files for calculating chemical production rates,
thermodynamic properties, and transport coefficients.
"""

# Standard libraries
import re
import math
import argparse
from numpy import zeros
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
from utils import write_module, code, si, di, ti, qi, new_line
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
    parser.add_argument('--pressureRef',
                        required=False,
                        default=101325.0,
                        help='Reference pressure.')
    parser.add_argument('--temperatureRef',
                        required=False,
                        default=1000.0,
                        help='Reference temperature.')
    parser.add_argument('--moleFractionsRef',
                        required=False,
                        default=1.0,
                        help='Reference mole fractions. If argument is equal to 1.0'
                             ' the mole fractions will be equal for all species.')
    parser.add_argument('--lengthRef',
                        required=False,
                        default=1.0,
                        help='Reference length')
    parser.add_argument('--velocityRef',
                        required=False,
                        default=1.0,
                        help='Reference velocity.')
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
    parser.add_argument('--transport',
                        required=False,
                        default=True,
                        help='Write transport properties')
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

    def __init__(self, sp_names, molar_masses, thermodynamics,
                 well_depth, dipole_moment, diameter, rotational_relaxation,
                 degrees_of_freedom, polarizability, sp_len):
        self.species_names = sp_names
        self.molar_masses = molar_masses
        self.thermodynamics = thermodynamics
        self._well_depth = well_depth
        self._dipole_moment = dipole_moment
        self._diameter = diameter
        self._rotational_relaxation = rotational_relaxation
        self._degrees_of_freedom = degrees_of_freedom
        self._polarizability = polarizability
        self._sp_len = sp_len

    def _interaction_well_depth(self, a, b):
        well_depth = self._well_depth
        return sqrt(well_depth[a] * well_depth[b]) * sq(self._xi(a, b))

    def _T_star(self, a, b, T):
        return T * const.kB / self._interaction_well_depth(a, b)

    def _reduced_dipole_moment(self, a, b):
        (well_depth, dipole_moment, diameter) = (
            self._well_depth, self._dipole_moment, self._diameter)
        return (dipole_moment[a] * dipole_moment[b] /
                (8. * pi * const.epsilon0 * sqrt(well_depth[a] * well_depth[b]) *
                 cube((diameter[a] + diameter[b]) / 2.)))

    def _collision_integral(self, I0, table, fit, a, b, T):
        lnT_star = ln(self._T_star(a, b, T))
        header_lnT_star = list(map(ln, const.header_T_star))
        interp_start_index = min(
            (1 + next(i for i, header_lnT_star in enumerate(header_lnT_star[1:])
                      if lnT_star < header_lnT_star)) - 1, I0 + len(table) - 3)
        header_lnT_star_slice = header_lnT_star[interp_start_index:][:3]
        assert (len(header_lnT_star_slice) == 3)
        polynomials = fit[interp_start_index - I0:][:3]
        assert (len(polynomials) == 3)

        def _evaluate_polynomial(P, x):
            return dot(P, [pow(x, k) for k in range(len(P))])

        def _quadratic_interpolation(x, y, x0):
            return (
                    ((x[1] - x[0]) * (y[2] - y[1]) - (y[1] - y[0]) * (x[2] - x[1])) /
                    ((x[1] - x[0]) * (x[2] - x[0]) * (x[2] - x[1])) *
                    (x0 - x[0]) * (x0 - x[1]) + ((y[1] - y[0]) / (x[1] - x[0])) * (x0 - x[1]) + y[1])

        delta_star = self._reduced_dipole_moment(a, b)
        for P in polynomials:
            assert len(P) == 7, len(P)
        table = table[interp_start_index - I0:][:3]
        assert (len(table) == 3)
        # P[:6]: Reproduces Cantera truncated polynomial mistake
        if delta_star > 0.:
            y = [_evaluate_polynomial(P[:6], delta_star) for P in polynomials]
        else:
            y = [row[0] for row in table]
        return _quadratic_interpolation(header_lnT_star_slice, y, lnT_star)

    def _omega_star_22(self, a, b, T):
        return self._collision_integral(1, const.collision_integrals_Omega_star_22, const.Omega_star_22, a, b, T)

    def _omega_star_11(self, a, b, T):
        return (self._omega_star_22(a, b, T) /
                self._collision_integral(0, const.collision_integrals_A_star, const.A_star, a, b, T))

    def _viscosity(self, a, T):
        (molar_masses, diameter) = (self.molar_masses, self._diameter)
        return (5. / 16. * sqrt(pi * molar_masses[a] / const.NA * const.kB * T) /
                (self._omega_star_22(a, a, T) * pi * sq(diameter[a])))

    def _conductivity(self, a, T):
        (molar_masses, thermodynamics, well_depth,
         rotational_relaxation, degrees_of_freedom) = (
            self.molar_masses, self.thermodynamics, self._well_depth,
            self._rotational_relaxation, self._degrees_of_freedom)
        f_internal = (molar_masses[a] / const.NA / (const.kB * T) * self._diffusivity(a, a, T) /
                      self._viscosity(a, T))
        T_star = self._T_star(a, a, T)

        def _fz(T_star):
            return (1. + pow(pi, 3. / 2.) / sqrt(T_star) *
                    (1. / 2. + 1. / T_star) + (1. / 4. * sq(pi) + 2.) / T_star)

        # Scaling factor for temperature dependence of rotational relaxation:
        # Kee, Coltrin [2003:12.112, 2017:11.115]
        c1 = (2. / pi * (5. / 2. - f_internal) /
              (rotational_relaxation[a] * _fz(298. * const.kB / well_depth[a]) /
               _fz(T_star) + 2. / pi * (5. / 3. * degrees_of_freedom[a] + f_internal)))
        f_trans = 5. / 2. * (1. - c1 * degrees_of_freedom[a] / (3. / 2.))
        f_rot = f_internal * (1. + c1)
        Cv = (thermodynamics[a].molar_heat_capacity_R(T) - 5. / 2. - degrees_of_freedom[a])
        return ((self._viscosity(a, T) / (molar_masses[a] / const.NA)) * const.kB *
                (f_trans * 3. / 2. + f_rot * degrees_of_freedom[a] + f_internal * Cv))

    def _reduced_mass(self, a, b):
        molar_masses = self.molar_masses
        return (molar_masses[a] / const.NA * molar_masses[b] / const.NA /
                (molar_masses[a] / const.NA + molar_masses[b] / const.NA))

    def _xi(self, a, b):
        (dipole_moment, polarizability, diameter, well_depth) = (
            self._dipole_moment, self._polarizability, self._diameter, self._well_depth)
        if (dipole_moment[a] > 0.) == (dipole_moment[b] > 0.):
            return 1.
        (polar, non_polar) = (a, b) if dipole_moment[a] != 0. else (b, a)
        return (1. + 1. / 4. * polarizability[non_polar] / cube(diameter[non_polar]) *
                sq(dipole_moment[polar] /
                   sqrt(4. * pi * const.epsilon0 * well_depth[polar] * cube(diameter[polar]))) *
                sqrt(well_depth[polar] / well_depth[non_polar]))

    def _reduced_diameter(self, a, b):
        diameter = self._diameter
        return (diameter[a] + diameter[b]) / 2. * pow(self._xi(a, b), -1. / 6.)

    # p*Djk
    def _diffusivity(self, a, b, T):
        return (3. / 16. * sqrt(2. * pi / self._reduced_mass(a, b)) * pow(const.kB * T, 3. / 2.) /
                (pi * sq(self._reduced_diameter(a, b)) * self._omega_star_11(a, b, T)))

    def transport_polynomials(self, T0):
        T_rng = linspace(300., 3000., 50)

        class TransportPolynomials:
            pass

        transport_polynomials = TransportPolynomials()
        transport_polynomials.viscosity = [
            polynomial_regression(ln(T_rng / T0), [sqrt(self._viscosity(a, T) / sqrt(T)) for T in T_rng])
            for a in range(self._sp_len)]
        transport_polynomials.conductivity = [
            polynomial_regression(ln(T_rng / T0), [self._conductivity(a, T) / sqrt(T) for T in T_rng])
            for a in range(self._sp_len)]
        transport_polynomials.diffusivity = [
            [polynomial_regression(ln(T_rng / T0), [(self._diffusivity(a, b, T)) / (T * sqrt(T)) for T in T_rng])
             for b in range(self._sp_len)] for a in range(self._sp_len)]
        return transport_polynomials


def get_species_from_model(species):
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
        nasa7 = NASA7(pieces, temperature_split)
        return nasa7

    thermodynamics = p(lambda s: from_model(s['thermo']))
    # Check if species have two sets of thermodynamic coefficients
    for idx, specie in enumerate(thermodynamics):
        try:
            data = specie.pieces[1][0]
        except IndexError:
            raise SystemExit(
                f'Specie {sp_names[idx]} has only one set of thermodynamic coefficients. Check input yaml file!')
    degrees_of_freedom = p(lambda s: {'atom': 0, 'linear': 1, 'nonlinear': 3 / 2}[s['transport']['geometry']])
    well_depth = p(lambda s: s['transport']['well-depth'] * const.kB)
    diameter = p(lambda s: s['transport']['diameter'] * 1e-10)  # Å
    dipole_moment = p(lambda s: s['transport'].get('dipole', 0) * const.Cm_per_Debye)
    polarizability = p(lambda s: s['transport'].get('polarizability', 0) * 1e-30)  # Å³
    rotational_relaxation = p(lambda s: float(s['transport'].get('rotational-relaxation', 0)))

    species = Species(sp_names, molar_masses, thermodynamics,
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
    if version_info.minor >= 9:
        # print(r['equation'])
        [reaction.reactants, reaction.products] = [
            [sum([c for (s, c) in side if s == specie]) for specie in sp_names] for side in
            [[(s.split(' ')[1], int(s.split(' ')[0])) if ' ' in s else (s, 1) for s in
              [s.strip() for s in side.removesuffix('(+ M)').removesuffix('+ M').removesuffix('(+M)').split(' + ')]]
             for side in [s.strip() for s in re.split('<?=>', r['equation'])]
             ]
        ]
    else:
        [reaction.reactants, reaction.products] = [
            [sum([c for (s, c) in side if s == specie]) for specie in sp_names] for side in
            [[(s.split(' ')[1], int(s.split(' ')[0])) if ' ' in s else (s, 1) for s in
              [s.strip() for s in re.sub('\\+ M$', '', re.sub('\\(\\+M\\)$', '', side)).split(' + ')]]
             for side in [s.strip() for s in re.split('<?=>', r['equation'])]
             ]
        ]
    reaction.net = [-reactant + product for reactant, product in zip(reaction.reactants, reaction.products)]
    reaction.sum_net = sum(reaction.net)

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
        if r['type'] != 'pressure-dependent-Arrhenius': print(f"expected P-log: {r}")
        if len(r['rate-constants']) != 1:
            exit(f"unimplemented P-log: always using first rate constant {r['rate-constants'][0]} for {r} instead")

    if r.get('type') == None or r.get('type') == 'elementary':
        if re.search('[^<]=>', r['equation']):
            reaction.type = 'irreversible'
        elif '<=>' in r['equation'] or '= ' in r['equation']:
            # Keep the space after = ('= ') here. or let '[^<]=>' match first
            assert (r.get('reversible', True))
            reaction.type = 'elementary'
        else:
            exit(r)
    elif r.get('type') == 'three-body':
        reaction.type = 'three-body'
    elif r.get('type') == 'falloff' and not r.get('Troe') and not r.get('SRI'):
        reaction.type = 'pressure-modification'
        reaction.k0 = rate_constant(r['low-P-rate-constant'], reactants)
    elif r.get('type') == 'falloff' and r.get('Troe'):
        reaction.type = 'Troe'
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

    return code([f"{f'{out}{specie}]' if '[' in out else f'{si}cfloat {out}{specie}'} "
                 f"= {expression(sp_thermo[specie].pieces[p])};" for specie in sp_indices])


def write_energy(out, length, expression, sp_thermo):
    """
    Write the evaluation of the energy log polynomial for a given species.
    """

    temperature_splits = {}
    for index, specie in enumerate(sp_thermo[:length]):
        temperature_splits.setdefault(specie.temp_split, []).append(index)
    return code([f'{si}if (T <= {f(temperature_split)}) {{\n'
                 f'{write_thermo_piece(out, species, sp_thermo, expression, 0)}\n'
                 f'{si}}} else {{\n'
                 f'{write_thermo_piece(out, species, sp_thermo, expression, 1)} '
                 f'\n{si}}}'
                 for temperature_split, species in temperature_splits.items()])


def compute_k_rev_unroll(r):
    """
    Calculate the reverse rate constant for a given reaction for the unrolled code.
    """

    pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
    gibbs_terms = []
    gibbs_terms_div = []
    inv_gibbs_terms = []
    for j, net in enumerate(r.net):
        if net > 0:
            for o in range(net):
                gibbs_terms.append(f"gibbs0_RT[{j}]")
    inv_gibbs_terms.append(f"1./(")
    for k, net in enumerate(r.net):
        if net < 0:
            for n in range(abs(net)):
                gibbs_terms_div.append(f"gibbs0_RT[{k}]")
    inv_gibbs_terms.append(f"{'*'.join(gibbs_terms_div)}")
    inv_gibbs_terms.append(f")")
    gibbs_terms.append(''.join(inv_gibbs_terms))
    k_rev = f"{'*'.join(gibbs_terms)}{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};"
    return k_rev


def write_reaction(idx, r, loop_gibbsexp):
    """
    Write reaction for the unrolled code.
    """

    lines = []
    lines.append(f"{si}//{idx + 1}: {r.description}")

    if hasattr(r, 'efficiencies'):
        efficiency = [f'{f(efficiency - 1)}*Ci[{specie}]' if efficiency != 2 else f'Ci[{specie}]'
                      for specie, efficiency in enumerate(r.efficiencies) if efficiency != 1]
        lines.append(f'''{si}eff = Cm{f"+{'+'.join(efficiency)}" if efficiency else ''};''')

    def arrhenius(rc):
        A, beta, E = (
            rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
        expression = (
            "__NEKRK_EXP__("
            f"{f'{f(-E)}*rcpT+' if E != 0 else ''}"
            f"{f'{f(beta)}*lnT+' if beta != 0 else ''}"
            f"{f(ln(A))})"
            if A != 0 and (beta != 0 or E != 0)
            else f"{f(A)}"
        )
        return expression

    def arrhenius_diff(rc):
        A_inf, beta_inf, E_inf = (
            rc.preexponential_factor, rc.temperature_exponent, rc.activation_temperature)
        A0, beta0, E0 = r.k0.preexponential_factor, r.k0.temperature_exponent, r.k0.activation_temperature
        expression = (
            f"__NEKRK_EXP__("
            f"{f'{f(-E0 + E_inf)}*rcpT+' if (E0 - E_inf) != 0 else ''}"
            f"{f'{f(beta0 - beta_inf)}*lnT+' if (beta0 - beta_inf) != 0 else ''}{f(ln(A0) - ln(A_inf))})"
            if (A0 - A_inf) != 0 and ((beta0 - beta_inf) != 0 or (E0 - E_inf) != 0) else f'{f(A0 / A_inf)}'
        )
        return expression

    if r.type == 'elementary' or r.type == 'irreversible':
        lines.append(f'{si}k = {arrhenius(r.rate_constant)};')
    elif r.type == 'three-body':
        lines.append(f"{si}k = {arrhenius(r.rate_constant)}{f'* eff' if hasattr(r, 'efficiencies') else '* Cm'};")
    elif r.type == 'pressure-modification':
        lines.append(f"{si}k_inf = {arrhenius(r.rate_constant)};")
        lines.append(f"{si}Pr = {arrhenius_diff(r.rate_constant)}"
                     f"{f'* eff' if hasattr(r, 'efficiencies') else '* Cm'};")
        lines.append(f"{si}k = k_inf * Pr/(1 + Pr);")
    elif r.type == 'Troe':
        lines.append(f"{si}k_inf = {arrhenius(r.rate_constant)};")
        lines.append(f"{si}Pr = {arrhenius_diff(r.rate_constant)}"
                     f"{f'* eff' if hasattr(r, 'efficiencies') else '* Cm'};")
        lines.append(f"{si}logPr = __NEKRK_LOG10__(Pr + CFLOAT_MIN);")
        lines.append(f"{si}logFcent = __NEKRK_LOG10__({1 - r.troe.A}*exp({-1. / r.troe.T3}*T) + "
                     f"{r.troe.A}*exp({-1. / r.troe.T1}*T)"
                     f"{f' + exp({-r.troe.T2}*rcpT)' if r.troe.T2 < float('inf') else ''});")
        lines.append(f"{si}troe_c = -.4 - .67 * logFcent;")
        lines.append(f"{si}troe_n = .75 - 1.27 * logFcent;")
        lines.append(f"{si}troe = (troe_c + logPr)/(troe_n - .14*(troe_c + logPr));")
        lines.append(f"{si}F = __NEKRK_POW__(10, logFcent/(1.0 + troe*troe));")
        lines.append(f"{si}k = k_inf * Pr/(1 + Pr) * F;")
    elif r.type == 'SRI':
        lines.append(f"{si}k_inf = {arrhenius(r.rate_constant)};")
        lines.append(f"{si}Pr = {arrhenius_diff(r.rate_constant)}"
                     f"{f'* eff' if hasattr(r, 'efficiencies') else '* Cm'};")
        lines.append(f"{si}logPr = log10(Pr);")
        lines.append(f"{si}F = {r.sri.D}*pow({r.sri.A}*exp({-r.sri.B}*rcpT)+"
                     f"exp({-1. / r.sri.C}*T), 1./(1.+logPr*logPr))*pow(T, {r.sri.E});")
        lines.append(f"{si}k = k_inf * Pr/(1 + Pr) * F;")
    else:
        exit(r.type)

    phase_space = lambda reagents: '*'.join(
        '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
        coefficient != 0.)
    Rf = phase_space(r.reactants)
    lines.append(f"{si}Rf= {Rf};")
    if r.type == 'irreversible':
        lines.append(f"{si}cR = k * Rf;")
    else:
        pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
        if loop_gibbsexp:
            lines.append(f"{si}k_rev = {compute_k_rev_unroll(r)}")
        else:
            lines.append(f"{si}k_rev = __NEKRK_EXP_OVERFLOW__("
                         f"{'+'.join(imul(net, f'gibbs0_RT[{k}]') for k, net in enumerate(r.net) if net != 0)})"
                         f"{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};")
        lines.append(f"{si}Rr = k_rev * {phase_space(r.products)};")
        lines.append(f"{si}cR = k * (Rf - Rr);")
    lines.append(f"#ifdef DEBUG")
    lines.append(f'{si}printf("{idx + 1}: %+.15e\\n", cR);')
    lines.append(f"#endif")
    lines.append(f"""{code(
        f"{si}rates[{specie}] += {imul(net, 'cR')};" for specie, net in enumerate(r.net) if net != 0)}""")
    lines.append(f"")

    return f'{code(lines)}'


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
    lines = []
    if static:
        if target.__eq__('c++17'):
            vtype = f'alignas({align_width}) static constexpr'
        else:
            vtype = f'const'
    else:
        vtype = f""
    if type(var_str) is list and hasattr(var_str, '__iter__'):
        assert len(var_str) == len(var)
        for i in range(len(var_str)):
            lines.append(f"{si}{vtype} cfloat {var_str[i]}[{len(var[i])}] = {{{f_sci_not(var[i])}}};")
            lines.append(f"")
    else:
        lines.append(f"{si}{vtype} cfloat {var_str}[{len(var)}] = {{{f_sci_not(var)}}};")
    return f'{code(lines)}'


def get_energy_coefficients(sp_thermo, sp_len):
    """
    Extract energy coefficients, temperature thersholds and reordering indices
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
    return a0, a1, a2, a3, a4, a5, a6, ids_thermo_new, len_unique_temp_splits, unique_temp_split


def get_thermo_prop(thermo_prop, unique_temp_split, len_unique_temp_splits):
    """
    Compute the polynomial evaluation of a specific thermodynamic property.
    """
    lines = []
    lines.append(f"{si}unsigned int offset;")
    lines.append(f"{si}unsigned int i_off;")
    for i in range(len(unique_temp_split)):
        if len_unique_temp_splits[i] > 1:
            lines.append(
                f"{si}offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + "
                f"(T>{unique_temp_split[i]})*{len_unique_temp_splits[i]};")
        else:
            lines.append(
                f"{si}offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + (T>{unique_temp_split[i]});")
        lines.append(f"{si}for(unsigned int i=0; i<{len_unique_temp_splits[i]}; ++i)")
        lines.append(f"{si}{{")
        lines.append(f"{di}i_off = i + offset;")
        if thermo_prop == 'cp_R':
            lines.append(
                f"{di}cp_R[i+{sum_of_list(len_unique_temp_splits[:i])}] = a0[i_off] + a1[i_off]*T + a2[i_off]*T2 + "
                f"a3[i_off]*T3+ a4[i_off]*T4;")
        elif thermo_prop == 'h_RT':
            lines.append(
                f"{di}h_RT[i+{sum_of_list(len_unique_temp_splits[:i])}] = a0[i_off] + {f(0.5)}*a1[i_off]*T + "
                f"{f(1. / 3.)}*a2[i_off]*T2 + {f(0.25)}*a3[i_off]*T3+ {f(0.2)}*a4[i_off]*T4 + a5[i_off]*rcpT;")
        lines.append(f"{si}}}")
        lines.append(f"")
    return f"{code(lines)}"


def reorder_thermo_prop(thermo_prop, unique_temp_split, ids_thermo_prop_new, sp_len):
    """
    Reorder thermodynamic property data according to new indices.
    """
    lines = []
    if len(unique_temp_split) > 1:
        lines.append(f"{si}//Reorder thermodynamic properties")
        lines.append(f"{si}cfloat tmp[{sp_len}];")
        lines.append(f"{si}for(unsigned i=0; i<{sp_len}; ++i)")
        if thermo_prop == 'cp_R':
            lines.append(f"{di}tmp[i] = cp_R[i];")
        elif thermo_prop == 'h_RT':
            lines.append(f"{di}tmp[i] = h_RT[i];")
        for i in range(sp_len):
            if thermo_prop == 'cp_R':
                lines.append(f"{si}cp_R[{i}] = tmp[{ids_thermo_prop_new.index(i)}];")
            elif thermo_prop == 'h_RT':
                lines.append(f"{si}h_RT[{i}] = tmp[{ids_thermo_prop_new.index(i)}];")
    else:
        lines.append(f'')
    return f"{code(lines)}"


def write_file_mech(file_name, output_dir, sp_names, sp_len, active_sp_len, rxn_len, Mi):
    """
    Write the 'mech.h' file for the reaction mechanism data.
    """

    lines = []
    lines.append(f'#define n_species {sp_len}')
    lines.append(f'#define n_active_species {active_sp_len}')
    lines.append(f'#define species_names_length '
                 f'{reduce(lambda x, y: x + y, [len(i) for i in sp_names]) + len(sp_names)}')
    lines.append(f'#define n_reactions {rxn_len}')
    lines.append(f"""__NEKRK_CONST__ char species_names[species_names_length] = "{' '.join(sp_names)}";""")
    lines.append(f"__NEKRK_CONST__ cfloat nekrk_molar_mass[n_species] = {{{', '.join([repr(w) for w in Mi])}}};")
    lines.append(f"__NEKRK_CONST__ cfloat nekrk_rcp_molar_mass[n_species] = "
                 f"{{{', '.join([repr(1. / w) for w in Mi])}}};")
    lines.append(f'#define __NEKRK_NSPECIES__ n_species')
    lines.append(f'#define __NEKRK_NACTIVESPECIES__ n_active_species')

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_rates_unroll(file_name, output_dir, loop_gibbsexp, reactions, active_len, sp_len, sp_thermo):
    """
    Write the 'rates.inc'('frates.inc') file with unrolled loop specification.
    Loops are expanded by replicating their body multiple times, reducing the
    overhead of loop control which can lead to more efficient code execution,
    particularly beneficial for GPUs.
    """

    lines = []
    lines.append(f"#include <math.h>")
    lines.append(f"#define __NEKRK_EXP_OVERFLOW__(x) __NEKRK_MIN_CFLOAT(CFLOAT_MAX, __NEKRK_EXP__(x))")
    lines.append(f"__NEKRK_DEVICE__ __NEKRK_INLINE__ void nekrk_species_rates"
                 f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                 f"const cfloat rcpT, const cfloat Ci[], cfloat* rates) ")
    lines.append(f"{{")
    lines.append(f"{si}cfloat gibbs0_RT[{active_len}];")
    expression = lambda a: (f"{f(a[5])} * rcpT + {f(a[0] - a[6])} + {f(-a[0])} * lnT + "
                            f"{f(-a[1] / 2)} * T + {f((1. / 3. - 1. / 2.) * a[2])} * T2 + "
                            f"{f((1. / 4. - 1. / 3.) * a[3])} * T3 + {f((1. / 5. - 1. / 4.) * a[4])} * T4")
    lines.append(f'{write_energy(f"{di}gibbs0_RT[", active_len, expression, sp_thermo)}')
    if loop_gibbsexp:
        # lines.append(f"{si}cfloat rcp_gibbs0_RT[{active_len}];")
        lines.append(f"{si}for(unsigned int i=0; i<{active_len}; ++i)")
        lines.append(f"{si}{{")
        lines.append(f"{di}gibbs0_RT[i] = __NEKRK_EXP__(gibbs0_RT[i]);")
        # lines.append(f"{di}rcp_gibbs0_RT[i] = 1./gibbs0_RT[i];")
        lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f'{si}cfloat Cm = {"+".join([f"Ci[{specie}]" for specie in range(sp_len)])};')
    lines.append(f"{si}cfloat C0 = {f(const.one_atm / const.R)} * rcpT;")
    lines.append(f"{si}cfloat rcpC0 = {f(const.R / const.one_atm)} * T;")
    lines.append(f"{si}cfloat k, Rf, k_inf, Pr, logFcent, k_rev, Rr, cR;")
    lines.append(f"{si}cfloat eff;")
    lines.append(f"{si}cfloat logPr, F, troe, troe_c, troe_n;")
    lines.append(f"")
    for idx, r in enumerate(reactions):
        lines.append(f"{write_reaction(idx, r, loop_gibbsexp)}")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_enthalpy_unroll(file_name, output_dir, sp_len, sp_thermo):
    """
    Write the 'fenthalpy_RT.inc' file with unrolled loop specification.
    """
    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_enthalpy_RT"
                 f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                 f"const cfloat rcpT,cfloat* h_RT)")
    lines.append(f"{{")
    expression = lambda a: (f'{f(a[0])} + {f(a[1] / 2)} * T + {f(a[2] / 3)} * T2 + '
                            f'{f(a[3] / 4)} * T3 + {f(a[4] / 5)} * T4 + {f(a[5])} * rcpT')
    lines.append(f'{write_energy(f"{di}h_RT[", sp_len, expression, sp_thermo)}')
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_heat_capacity_unroll(file_name, output_dir, sp_len, sp_thermo):
    """
    Write the 'fheat_capacity_R.inc' file with unrolled loop specification.
    """
    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_molar_heat_capacity_R"
                 f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                 f"const cfloat rcpT,cfloat* cp_R)")
    lines.append(f"{{")
    expression = lambda a: f'{f(a[0])} + {f(a[1])} * T + {f(a[2])} * T2 + {f(a[3])} * T3 + {f(a[4])} * T4'
    lines.append(f'{write_energy(f"{di}cp_R[", sp_len, expression, sp_thermo)}')
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_conductivity_unroll(file_name, output_dir, transport_polynomials, sp_names):
    """
    Write the 'fconductivity.inc' file with unrolled loop specification.
    """

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivity"
                 f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[])")
    lines.append(f"{{")
    lines.append(f"{si}cfloat lambda_k, a = 0., b = 0.;")
    lines.append(f"")
    for k, P in enumerate(transport_polynomials.conductivity):
        lines.append(f"{si}//{sp_names[k]}")
        lines.append(f"{si}lambda_k = {evaluate_polynomial(P)};")
        lines.append(f"{si}a += nXi[{k}]*lambda_k;")
        lines.append(f"{si}b += nXi[{k}]/lambda_k;")
        lines.append(f"")
    lines.append(f"{si}return a/rcpMbar + rcpMbar/b;")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_viscosity_unroll(file_name, output_dir, transport_polynomials, sp_len, Mi):
    """
    Write the 'fviscosity.inc' file with unrolled loop specification.
    """

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat sq(cfloat x) {{ return x*x; }}")
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosity"
                 f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[]) ")
    lines.append(f"{{")
    for k, P in enumerate(transport_polynomials.viscosity):
        lines.append(f"{si}cfloat v{k} = {evaluate_polynomial(P)};")
    lines.append(f"{code(f'{si}cfloat sum_{k} = 0.;' for k in range(sp_len))} "
                 f"// same name is used to refer to the reciprocal evaluation "
                 f"explicitly interleaved into the last iteration")

    def sq_v(Va):
        return f'sq({f(Va)}+{f(Va * sqrt(sqrt(Mi[j] / Mi[k])))}*v{k}*r{j})'

    for j in range(sp_len - 1):
        lines.append(f"{si}{{")
        lines.append(f"{di}cfloat r{j} = {f(1.)}/v{j};")
        for k in range(sp_len):
            lines.append(f"{di}sum_{k} += nXi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))};")
        lines.append(f"{si}}}")
    for j in [sp_len - 1]:
        lines.append(f"{si}{{")
        lines.append(f"{di}cfloat r{j} = {f(1.)}/v{j};")
        for k in range(sp_len):
            lines.append(f"{di}sum_{k} += nXi[{j}]*{sq_v(sqrt(1 / sqrt(8) * 1 / sqrt(1. + Mi[k] / Mi[j])))}; "
                         f"/*rcp_*/sum_{k} = {f(1.)}/sum_{k};")
        lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"""{si}return {('+' + new_line).join(f"{ti if k > 0 else ' '}nXi[{k}]*sq(v{k}) * /*rcp_*/sum_{k}"
                                                       for k in range(sp_len))};""")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_diffusivity_unroll(file_name, output_dir, transport_polynomials, sp_len, Mi):
    """
    Write the 'fdiffusivity.inc' file with unrolled loop specification.
    """

    lines = []

    S = [''] * sp_len

    def mut(y, i, v):
        S[i] = v
        return y

    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                 f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                 f"cfloat nXi[], dfloat* out, unsigned int stride) ")
    lines.append(f"{{")
    for k in range(sp_len):
        for j in range(k):
            lines.append(
                f"{si}cfloat R{k}_{j} = 1/({evaluate_polynomial(transport_polynomials.diffusivity[k][j])});")
            lines.append(f"{si}cfloat S{k}_{j} = {mut(f'{S[k]}+' if S[k] else '', k, f'S{k}_{j}')}nXi[{j}]*R{k}_{j};")
            lines.append(f"{si}cfloat S{j}_{k} = {mut(f'{S[j]}+' if S[j] else '', j, f'S{j}_{k}')}nXi[{k}]*R{k}_{j};")
    for k in range(sp_len):
        lines.append(f"{si}out[{k}*stride+id] = scale * (1.0f - {Mi[k]}f*nXi[{k}])/{S[k]};")
    lines.append(f"}}")
    write_module(output_dir, file_name, f'{code(lines)}')
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
        lines = []
        if len(ids_E0B0) > 0:
            lines.append(f'{si}// {len(ids_E0B0)} rate constants with E_R = 0 and beta = 0 ')
            lines.append(f'{si}for(unsigned int i=0; i<{len(ids_E0B0)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(f'{di}k[i+{len(ids_EB)}] = A[i+{len(ids_EB)}];')
            lines.append(f'{si}}}')
        if len(ids_E0Bneg2) > 0:
            start_ids_E0Bneg2 = len(ids_EB) + len(ids_E0B0)
            lines.append(f'{si}// {len(ids_E0Bneg2)} rate constants with E_R = 0 and beta = -2 ')
            lines.append(f'{si}cfloat rcpT_2 = rcpT*rcpT;')
            lines.append(f'{si}for(unsigned int i=0; i<{len(ids_E0Bneg2)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(f'{di}k[i+{start_ids_E0Bneg2}] = A[i+{start_ids_E0Bneg2}]*rcpT_2;')
            lines.append(f'{si}}}')
        if len(ids_E0Bneg1) > 0:
            start_ids_E0Bneg1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2)
            lines.append(f'{si}// {len(ids_E0Bneg1)} rate constants with E_R = 0 and beta = -1 ')
            lines.append(f'{si}for(unsigned int i=0; i<{len(ids_E0Bneg1)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(f'{di}k[i+{start_ids_E0Bneg1}] = A[i+{start_ids_E0Bneg1}]*rcpT;')
            lines.append(f'{si}}}')
        if len(ids_E0B1) > 0:
            start_ids_E0B1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1)
            lines.append(f'{si}// {len(ids_E0B1)} rate constants with E_R = 0 and beta = 1 ')
            lines.append(f'{si}for(unsigned int i=0; i<{len(ids_E0B1)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(f'{di}k[i+{start_ids_E0B1}] = A[i+{start_ids_E0B1}]*T;')
            lines.append(f'{si}}}')
        if len(ids_E0B2) > 0:
            start_ids_E0B2 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1) + len(ids_E0B1)
            lines.append(f'{si}// {len(ids_E0B2)} rate constants with E_R = 0 and beta = 2 ')
            lines.append(f'{si}cfloat T_2 = T*T;')
            lines.append(f'{si}for(unsigned int i=0; i<{len(ids_E0B1)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(f'{di}k[i+{start_ids_E0B2}] = A[i+{start_ids_E0B2}]*T_2;')
            lines.append(f'{si}}}')
        if len(ids_ErBr) > 0:
            start_ids_ErBr = len(ids_new) - len(ids_ErBr)
            lines.append(
                f'{si}// {len(ids_ErBr)} rate constants with E_R and '
                f'beta the same as for other rate constants which have already been computed ')
            lines.append(f'{si}for(unsigned int i=0; i<{len(unique_pos_ids_ErBr)}; ++i)')
            lines.append(f'{si}{{')
            lines.append(
                f'{di}k[i+{start_ids_ErBr}] = A[i+{start_ids_ErBr}]*k[i+{len(ids_EB) - len(unique_pos_ids_ErBr)}];')
            lines.append(f'{si}}}')
            if (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) > 0:
                for i in range(len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)):
                    start_ids_ErBr_rep = len(ids_new) - (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) + i
                    ids_k_rep = ids_EB.index(ids_ErBr_values[len(unique_pos_ids_ErBr) + i])
                    lines.append(f'{si}k[{start_ids_ErBr_rep}]=A[{start_ids_ErBr_rep}]*k[{ids_k_rep}];')
        return f'{code(lines)}'

    ids_eff = []
    dic_unique_eff = {}

    def reodrder_eff(r):
        lines = []
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
                lines.append(f"{si}cfloat eff{i} = Cm + {'+'.join(ci)};")

            return f'{code(lines)}'
        else:
            lines.append(f'')
            return f'{code(lines)}'

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
        tb_correction = []
        if len(ids_3b_rxn) > 0:
            tb_correction.append(f"{si}//Correct k for three-body reactions")
            for i in ids_3b_rxn:
                if hasattr(r[i], 'efficiencies'):
                    tb_correction.append(
                        f"{si}k[{ids_new.index(i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};")
                else:
                    tb_correction.append(
                        f"{si}k[{ids_new.index(i)}] *= Cm;")

        # Pressure-dependent reactions
        pd_correction = []
        if len(ids_pd_rxn) > 0:
            pd_correction.append(f"{si}//Correct k for pressure-dependent reactions")
            for i in ids_pd_rxn:
                if hasattr(r[i], 'efficiencies'):
                    pd_correction.append(
                        f"{si}k[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};")
                else:
                    pd_correction.append(
                        f"{si}k[{ids_new.index(rxn_len + i)}] *= Cm;")
                pd_correction.append(
                    f"{si}k[{ids_new.index(rxn_len + i)}] /= "
                    f"(1+ k[{ids_new.index(rxn_len + i)}]/(k[{ids_new.index(i)}]+ CFLOAT_MIN));")

        # Troe reactions
        troe_correction = []
        if len(ids_troe_rxn) > 0:
            troe_correction.append(f"{si}//Correct k for troe reactions")
            for i in ids_troe_rxn:
                if hasattr(r[i], 'efficiencies'):
                    troe_correction.append(
                        f"{si}k[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};")
                else:
                    troe_correction.append(
                        f"{si}k[{ids_new.index(rxn_len + i)}] *= Cm;")
                troe_correction.append(
                    f"{si}k[{ids_new.index(rxn_len + i)}]/= (k[{ids_new.index(i)}] + CFLOAT_MIN);")
            troe_A, rcp_troe_T1, troe_T2, rcp_troe_T3 = [], [], [], []
            for i in ids_troe_rxn:
                troe_A.append(r[i].troe.A)
                rcp_troe_T1.append(1 / r[i].troe.T1)
                troe_T2.append(r[i].troe.T2)
                rcp_troe_T3.append(1 / r[i].troe.T3)
            ids_troe = []
            for i in ids_troe_rxn:
                ids_troe.append(ids_new.index(rxn_len + i))
            troe_correction.append(f"")
            troe_correction.append(
                f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} "
                f"cfloat troe_A[{len(ids_troe_rxn)}] = {{{f_sci_not(troe_A)}}};")
            troe_correction.append(
                f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} "
                f"cfloat rcp_troe_T1[{len(ids_troe_rxn)}] = {{{f_sci_not(rcp_troe_T1)}}};")
            troe_correction.append(
                f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} "
                f"cfloat troe_T2[{len(ids_troe_rxn)}] = {{{f_sci_not(troe_T2)}}};" if not (
                    any([i == float('inf') for i in troe_T2])) else '')
            troe_correction.append(
                f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} "
                f"cfloat rcp_troe_T3[{len(ids_troe_rxn)}] = {{{f_sci_not(rcp_troe_T3)}}};")
            troe_correction.append(f"")
            troe_correction.append(
                f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} int "
                f"ids_troe[{len(ids_troe_rxn)}] = {str(ids_troe).replace('[', '{').replace(']', '}')};")
            troe_correction.append(
                f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                f"logFcent[{len(ids_troe_rxn)}];")
            troe_correction.append(f"{si}for(unsigned int i = 0; i<{len(ids_troe_rxn)}; ++i)")
            troe_correction.append(f"{si}{{")
            troe_correction.append(
                f"{di}logFcent[i] = __NEKRK_LOG10__(({f(1.)} - troe_A[i])*exp(-T*rcp_troe_T3[i]) + "
                f"troe_A[i]*exp(-T*rcp_troe_T1[i]){f'+ exp(-troe_T2[i]*rcpT)' if troe_T2[0] < float('inf') else ''});")
            troe_correction.append(f"{si}}}")
            troe_correction.append(f"{si}for(unsigned int i = 0; i<{len(ids_troe_rxn)}; ++i)")
            troe_correction.append(f"{si}{{")
            troe_correction.append(
                f"{di}cfloat troe_c = {f(-0.4)} - {f(0.67)} * logFcent[i];")
            troe_correction.append(
                f"{di}cfloat troe_n = {f(0.75)} - {f(1.27)} * logFcent[i];")
            troe_correction.append(
                f"{di}cfloat logPr = __NEKRK_LOG10__(k[ids_troe[i]] + CFLOAT_MIN);")
            troe_correction.append(
                f"{di}cfloat troe = (troe_c + logPr)/(troe_n - {f(0.14)}*(troe_c + logPr)+CFLOAT_MIN);")
            troe_correction.append(
                f"{di}cfloat F = __NEKRK_POW__(10, logFcent[i]/({f(1.0)} + troe*troe));")
            troe_correction.append(f"{di}k[ids_troe[i]] /= ({f(1.)}+k[ids_troe[i]]);")
            troe_correction.append(f"{di}k[ids_troe[i]] *= F;")
            troe_correction.append(f"{si}}}")
            for i in ids_troe_rxn:
                troe_correction.append(
                    f"{si}k[{ids_new.index(rxn_len + i)}]*= k[{ids_new.index(i)}];")

        # Combine all reactions
        all_reactions = tb_correction + pd_correction + troe_correction
        return f"{code(all_reactions)}"

    # Reorder reactions back to original
    def reorder_k(r):
        lines = []
        for i in range(len(r)):
            if hasattr(rxn[i], 'k0'):
                lines.append(f"{si}k[{i}] = tmp[{ids_new.index(rxn_len + i)}];")
            else:
                lines.append(f"{si}k[{i}] = tmp[{ids_new.index(i)}];")
        return f"{code(lines)}"

    # Compute the gibbs energy
    (a0, a1, a2, a3, a4, a5, a6,
     ids_gibbs_new, len_unique_temp_splits, unique_temp_split) = get_energy_coefficients(sp_thermo, sp_len)

    def gibbs_energy():
        lines = []
        for i in range(len(unique_temp_split)):
            if len_unique_temp_splits[i] > 1:
                lines.append(
                    f"{si}offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + "
                    f"(T>{unique_temp_split[i]})*{len_unique_temp_splits[i]};")
            else:
                lines.append(
                    f"{si}offset = {2 * sum_of_list(len_unique_temp_splits[:i])} + (T>{unique_temp_split[i]});")
            lines.append(f"{si}for(unsigned int i=0; i<{len_unique_temp_splits[i]}; ++i)")
            lines.append(f"{si}{{")
            lines.append(f"{di}i_off = i + offset;")
            lines.append(
                f"{di}gibbs0_RT[i+{sum_of_list(len_unique_temp_splits[:i])}] = "
                f"a0[i_off]*(1-lnT) - a1[i_off]*{f(0.5)}*T {f(1. / 3. - 0.5)}*a2[i_off]*T2 "
                f"{f(0.25 - 1. / 3.)}*a3[i_off]*T3 {f(0.2 - 0.25)}*a4[i_off]*T4 + a5[i_off]*rcpT - a6[i_off];")
            lines.append(f"{si}}}")
            lines.append(f"")
        return f"{code(lines)}"

    # Reciprocal gibbs energy
    repeated_rcp_gibbs = [0] * len(sp_thermo)
    for i in range(rxn_len):
        for j, net in enumerate(rxn[i].net):
            if net < 0:
                repeated_rcp_gibbs[j] += 1
    ids_rcp_gibbs = [i for i, x in enumerate(repeated_rcp_gibbs) if x > 2]
    ids_rcp_gibbs_reordered = [ids_gibbs_new.index(i) for i in ids_rcp_gibbs]

    # Compute reverse rates
    def compute_k_rev(r):
        lines = []
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
            lines.append(
                f"{si}k_rev[{i}] = {'*'.join(gibbs_terms)}{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};")
        return f"{code(lines)}"

    # Compute reaction rates
    def compute_rates(r):
        phaseSpace = lambda reagents: '*'.join(
            '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
            coefficient != 0.)

        lines = []
        for i in range(len(r)):
            lines.append(f"{si}//{i + 1}: {r[i].description}")
            if r[i].type == 'irreversible':
                if hasattr(r[i], 'k0'):
                    lines.append(f"{si}cR = k[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)});")
                else:
                    lines.append(f"{si}cR = k[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)});")
            else:
                if hasattr(r[i], 'k0'):
                    lines.append(
                        f"{si}cR = k[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)}-"
                        f"k_rev[{i}]*{phaseSpace(r[i].products)});")
                else:
                    lines.append(
                        f"{si}cR = k[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)}-"
                        f"k_rev[{i}]*{phaseSpace(r[i].products)});")
            lines.append(f"#ifdef DEBUG")
            lines.append(f"{si}printf(\"{i + 1}: %+.15e\\n\", cR);")
            if hasattr(r[i], 'k0'):
                lines.append(
                    f"{si}printf(\"{i + 1}: %+.15e, %+.15e\\n\", k[{ids_new.index(rxn_len + i)}], k_rev[{i}]);")
            else:
                lines.append(
                    f"{si}printf(\"{i + 1}: %+.15e, %+.15e\\n\", k[{ids_new.index(i)}], k_rev[{i}]);")
            lines.append(f"#endif")
            lines.append(code(
                f"{si}rates[{specie}] += {imul(net, 'cR')};" for specie, net in enumerate(r[i].net) if
                net != 0))
            lines.append(f"")
        return f"{code(lines)}"

    #############
    # Write file
    #############

    lines = []
    lines.append(f'#include <math.h>')
    lines.append(f"#define __NEKRK_EXP_OVERFLOW__(x) __NEKRK_MIN_CFLOAT(CFLOAT_MAX, __NEKRK_EXP__(x)")
    lines.append(f'__NEKRK_DEVICE__ __NEKRK_INLINE__ void nekrk_species_rates'
                 f'(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4,'
                 f' const cfloat rcpT, const cfloat Ci[], cfloat* rates) ')
    lines.append(f'{{')
    lines.append(f"{si}// Regrouping of rate constants to eliminate redundant operations")
    var_str = ['A', 'beta', 'E_R']
    var = [A_new, beta_new_reduced, E_R_new_reduced]
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"k[{len(ids_new)}];")
    lines.append(f"{si}// Compute the {len(ids_EB)} rate constants for which an evaluation is necessary")
    lines.append(f"{si}for(unsigned int i=0; i<{len(ids_EB)}; ++i)")
    lines.append(f"{si}{{")
    lines.append(f"{di}cfloat blogT = beta[i]*lnT;")
    lines.append(f"{di}cfloat E_RT = E_R[i]*rcpT;")
    lines.append(f"{di}cfloat diff = blogT - E_RT;")
    lines.append(f"{di}k[i] = __NEKRK_EXP__(A[i] + diff);")
    lines.append(f"{si}}}")
    lines.append(f"{set_k()}")
    lines.append(f"")
    lines.append(f"{si}// Correct rate constants based on reaction type")
    lines.append(f"{si}cfloat Cm = 0;")
    lines.append(f"{si}for(unsigned int i=0; i<{sp_len}; ++i)")
    lines.append(f"{di}Cm += Ci[i];")
    lines.append(f"")
    lines.append(f"{reodrder_eff(rxn)}")
    lines.append(f"")
    lines.append(f"{corr_k(rxn)}")
    lines.append(f"")
    lines.append(f"{si}// Compute the gibbs energy")
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    var = [a0, a1, a2, a3, a4, a5, a6]
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"gibbs0_RT[{sp_len}];")
    lines.append(f"{si}unsigned int offset;")
    lines.append(f"{si}unsigned int i_off; ")
    lines.append(f"")
    lines.append(f"{gibbs_energy()}")
    lines.append(f"{si}// Group the gibbs exponentials")
    lines.append(f"{si}for(unsigned int i=0; i<{sp_len}; ++i)")
    lines.append(f"{di}gibbs0_RT[i] = __NEKRK_EXP__(gibbs0_RT[i]);")
    lines.append(f"")
    lines.append(f"{si}// Compute the reciprocal of the gibbs exponential")
    lines.append(f"{si}{f'alignas({align_width}) static constexpr' if target.__eq__('c++17') else 'const'} "
                 f"int ids_rcp_gibbs[{len(ids_rcp_gibbs_reordered)}] = "
                 f"{str(ids_rcp_gibbs_reordered).replace('[', '{').replace(']', '}')};")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"rcp_gibbs0_RT[{len(ids_rcp_gibbs_reordered)}];")
    lines.append(f"{si}for(unsigned int i=0; i<{len(ids_rcp_gibbs_reordered)}; ++i)")
    lines.append(f"{di}rcp_gibbs0_RT[i] = {f(1.)}/gibbs0_RT[ids_rcp_gibbs[i]];")
    lines.append(f"")
    lines.append(f"{si}// Compute reverse rates")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"k_rev[{rxn_len}]; ")
    lines.append(f"{si}cfloat C0 = {f(const.one_atm / const.R)} * rcpT;")
    lines.append(f"{si}cfloat rcpC0 = {f(const.R / const.one_atm)} * T;")
    lines.append(f"{compute_k_rev(rxn)}")
    lines.append(f"")
    lines.append(f"{si}// Compute the reaction rates")
    lines.append(f"{si}cfloat cR;")
    lines.append(f"{compute_rates(rxn)}")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_enthalpy_roll(file_name, output_dir, align_width, target, sp_thermo, sp_len):
    """
    Write the 'fenthalpy_RT.inc' file with rolled loop specification.
    """

    (a0, a1, a2, a3, a4, a5, a6,
     ids_thermo_new, len_unique_temp_splits, unique_temp_split) = get_energy_coefficients(sp_thermo, sp_len)
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
    var = [a0, a1, a2, a3, a4, a5]

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_enthalpy_RT(const cfloat lnT, const cfloat T, "
                 f"const cfloat T2, const cfloat T3, const cfloat T4, const cfloat rcpT,cfloat h_RT[]) ")
    lines.append(f"{{")
    lines.append(f"{si}//Integration coefficients")
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{get_thermo_prop('h_RT', unique_temp_split, len_unique_temp_splits)}")
    lines.append(f"{reorder_thermo_prop('h_RT', unique_temp_split, ids_thermo_new, sp_len)}")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_heat_capacity_roll(file_name, output_dir, align_width, target, sp_thermo, sp_len):
    """
    Write the 'fheat_capacity_R.inc' file with rolled loop specification.
    """

    (a0, a1, a2, a3, a4, a5, a6,
     ids_thermo_new, len_unique_temp_splits, unique_temp_split) = get_energy_coefficients(sp_thermo, sp_len)
    var_str = ['a0', 'a1', 'a2', 'a3', 'a4']
    var = [a0, a1, a2, a3, a4]

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_molar_heat_capacity_R"
                 f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                 f"const cfloat rcpT,cfloat cp_R[]) ")
    lines.append(f"{{")
    lines.append(f"{si}//Integration coefficients")
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{get_thermo_prop('cp_R', unique_temp_split, len_unique_temp_splits)}")
    lines.append(f"{reorder_thermo_prop('cp_R', unique_temp_split, ids_thermo_new, sp_len)}")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
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

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_conductivity"
                 f"(cfloat rcpMbar, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[])")
    lines.append(f"{{")
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{si}cfloat lambda_k, sum1=0., sum2=0.;")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}lambda_k = b0[k] + b1[k]*lnT + b2[k]*lnT2 + b3[k]*lnT3 + b4[k]*lnT4;")
    lines.append(f"{di}sum1 += nXi[k]*lambda_k;")
    lines.append(f"{di}sum2 += nXi[k]/lambda_k;")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{si}return sum1/rcpMbar + rcpMbar/sum2;")
    lines.append(f"")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
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

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ cfloat nekrk_viscosity"
                 f"(cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, cfloat nXi[]) ")
    lines.append(f"{{")
    lines.append(f"{write_const_expression(align_width, target, True, var1_str, var1)}")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} mue[{sp_len}];")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} rcp_mue[{sp_len}];")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}mue[k] = a0[k] + a1[k]*lnT + a2[k]*lnT2 + a3[k]*lnT3 + a4[k]*lnT4;")
    lines.append(f"{di}rcp_mue[k] = 1./mue[k];")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{write_const_expression(align_width, target, True, var2_str, var2)}")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"sums[{sp_len}]={{0.}};")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}for(unsigned int j=0; j<{sp_len}; j++)")
    lines.append(f"{di}{{")
    lines.append(f"{ti}unsigned int idx = {sp_len}*k+j;")
    lines.append(f"{ti}cfloat sqrt_Phi_kj = C1[idx] + C2[idx]*mue[k]*rcp_mue[j];")
    lines.append(f"{ti}sums[k] += nXi[j]*sqrt_Phi_kj*sqrt_Phi_kj;")
    lines.append(f"{di}}}")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{si}cfloat vis = 0.;")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}vis += nXi[k]*mue[k]*mue[k]/sums[k];")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{si}return vis;")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def write_file_diffusivity_roll(file_name, output_dir, align_width, target, transport_polynomials, sp_len, Mi):
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

    lines = []
    lines.append(f"__NEKRK_DEVICE__  __NEKRK_INLINE__ void nekrk_density_diffusivity"
                 f"(unsigned int id, cfloat scale, cfloat lnT, cfloat lnT2, cfloat lnT3, cfloat lnT4, "
                 f"cfloat nXi[], dfloat* out, unsigned int stride) ")
    lines.append(f"{{")
    lines.append(f"{write_const_expression(align_width, target, True, var_str, var)}")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"sums1[{sp_len}]={{0.}};")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} "
                 f"sums2[{sp_len}]={{0.}};")
    lines.append(f"{si}for(unsigned int k=1; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}for(unsigned int j=0; j<k; j++)")
    lines.append(f"{di}{{")
    lines.append(f"{ti}unsigned int idx = k*(k-1)/2+j;")
    lines.append(f"{ti}cfloat rcp_Dkj = 1/(d0[idx] + d1[idx]*lnT + d2[idx]*lnT2 + d3[idx]*lnT3 + d4[idx]*lnT4);")
    lines.append(f"{ti}sums1[k] += nXi[j]*rcp_Dkj;")
    lines.append(f"{ti}sums2[j] += nXi[k]*rcp_Dkj;")
    lines.append(f"{di}}}")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{si}{f'alignas({align_width}) cfloat' if target.__eq__('c++17') else 'cfloat'} sums[{sp_len}];")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}sums[k] = sums1[k] + sums2[k];")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"{write_const_expression(align_width, target, True, 'Wi', Mi)}")
    lines.append(f"")
    lines.append(f"{si}for(unsigned int k=0; k<{sp_len}; k++)")
    lines.append(f"{si}{{")
    lines.append(f"{di}unsigned int idx = k*stride+id;")
    lines.append(f"{di}out[idx] = scale * (1.0f - Wi[k]*nXi[k])/sums[k];")
    lines.append(f"{si}}}")
    lines.append(f"")
    lines.append(f"}}")

    write_module(output_dir, file_name, f'{code(lines)}')
    return 0


def generate_files(mech_file=None, output_dir=None,
                   pressure_ref=101325.0, temperature_ref=1000.0,
                   mole_fractions_ref=1.0, length_ref=1.0, velocity_ref=1.0,
                   header_only=False, unroll_loops=False,
                   align_width=64, target=None,
                   loop_gibbsexp=False, transport=True
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
    species = get_species_from_model(active_sp+inert_sp)
    species_names = species.species_names
    reactions = [get_reaction_from_model(species_names, model['units'], reaction)
                 for reaction in model['reactions']]
    species_len = len(species_names)
    active_sp_len = len(active_sp)
    reactions_len = len(reactions)

    # Reference quantities
    Mi = species.molar_masses
    p_ref = float(pressure_ref)
    T_ref = float(temperature_ref)
    l_ref = float(length_ref)
    u_ref = float(velocity_ref)
    mole_proportions = [1. for _ in range(species_len)] \
        if args.moleFractionsRef == 1. else [float(x) for x in mole_fractions_ref.split(',')]
    Xi_ref = [x / sum(mole_proportions) for x in mole_proportions]
    M_tot = sum([n * M for n, M in zip(mole_proportions, Mi)])
    Yi_ref = [n * M / M_tot for n, M in zip(mole_proportions, Mi)]
    M_bar = sum((m * x) for m, x in zip(Mi, Xi_ref))
    rho_ref = M_bar * p_ref / (const.R * T_ref)
    cp_ref = const.R / M_bar * sum(a.molar_heat_capacity_R(T_ref) * x for a, x in zip(species.thermodynamics, Xi_ref))
    cv_ref = cp_ref - const.R / M_bar
    rho_cp_ref = rho_ref * cp_ref
    assert len(Xi_ref) == species_len

    # Load transport polynomials
    if transport and not header_only:
        transport_polynomials = species.transport_polynomials(T_ref)
        # need to multiply by dimensional sqrt(T): multiply by sqrt(T_ref) here and later on by sqrt(T_nondim)
        transport_polynomials.viscosity = [[sqrt(sqrt(T_ref)) * p for p in P] for P in
                                                transport_polynomials.viscosity]
        transport_polynomials.conductivity = [[(sqrt(T_ref) / 2) * p for p in P] for P in
                                                 transport_polynomials.conductivity]
        transport_polynomials.diffusivity = [
            [[(sqrt(T_ref) / const.R) * p for p in P]
             for k, P in enumerate(row)] for row in transport_polynomials.diffusivity]

    #########################
    # Write subroutine files
    #########################

    # File names
    mech_file = 'mech.h'
    rates_file = 'rates.inc'
    enthalpy_file = 'fenthalpy_RT.inc'
    heat_capacity_file = 'fheat_capacity_R.inc'
    conductivity_file = 'fconductivity.inc'
    viscosity_file = 'fviscosity.inc'
    diffusivity_file = 'fdiffusivity.inc'

    if header_only:
        write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
    else:
        write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
        if unroll_loops:  # Unrolled code
            precisions = [32, 64]
            for p in precisions:
                set_precision(p)
                if p == 32:
                    new_rates_file = 'f' + rates_file
                else:
                    new_rates_file = rates_file
                write_file_rates_unroll(new_rates_file, output_dir, loop_gibbsexp,
                                        reactions, active_sp_len, species_len, species.thermodynamics)
            set_precision(32)
            write_file_enthalpy_unroll(enthalpy_file, output_dir,
                                       species_len, species.thermodynamics)
            write_file_heat_capacity_unroll(heat_capacity_file, output_dir,
                                            species_len, species.thermodynamics)
            if transport:
                write_file_conductivity_unroll(conductivity_file, output_dir,
                                               transport_polynomials, species_names)
                write_file_viscosity_unroll(viscosity_file, output_dir,
                                            transport_polynomials, species_len, Mi)
                write_file_diffusivity_unroll(diffusivity_file, output_dir,
                                              transport_polynomials, species_len, Mi)
        else:  # Rolled code
            precisions = [32, 64]
            for p in precisions:
                set_precision(p)
                if p == 32:
                    new_rates_file = 'f' + rates_file
                else:
                    new_rates_file = rates_file
                write_file_rates_roll(new_rates_file, output_dir, align_width, target,
                                      species.thermodynamics, species_len, reactions, reactions_len)
            set_precision(32)
            write_file_enthalpy_roll(enthalpy_file, output_dir, align_width, target,
                                     species.thermodynamics, species_len)
            write_file_heat_capacity_roll(heat_capacity_file, output_dir, align_width, target,
                                          species.thermodynamics, species_len)
            if transport:
                write_file_conductivity_roll(conductivity_file, output_dir, align_width, target,
                                             transport_polynomials, species_len)
                write_file_viscosity_roll(viscosity_file, output_dir, align_width, target,
                                          transport_polynomials, species_len, Mi)
                write_file_diffusivity_roll(diffusivity_file, output_dir, align_width, target,
                                            transport_polynomials, species_len, Mi)

    return 0


if __name__ == "__main__":
    args = get_parser()

    generate_files(mech_file=args.mechanism,
                   output_dir=args.output,
                   pressure_ref=args.pressureRef,
                   temperature_ref=args.temperatureRef,
                   mole_fractions_ref=args.moleFractionsRef,
                   length_ref=args.lengthRef,
                   velocity_ref=args.velocityRef,
                   header_only=args.header_only,
                   unroll_loops=args.unroll_loops,
                   align_width=args.align_width,
                   target=args.target,
                   loop_gibbsexp=args.loop_gibbsexp,
                   transport=args.transport)
