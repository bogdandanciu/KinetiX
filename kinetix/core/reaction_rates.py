"""
Module containing routines for reaction rates computation.
"""

# Standard library imports
import math
import re
from sys import version_info

# Third-party library imports
from numpy import log as ln

# Local imports
from . import constants as const
from ..utils.general_utils import (
    FLOAT_MIN,
    f,
    f_sci_not,
    imul
)
from ..utils.write_utils import (
    CodeGenerator,
    get_thermo_coeffs,
    get_thermo_prop,
    write_const_expression,
    write_energy
)


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

    def rate_constant(self, cm3_factor_exp):

        class RateConstant:
            pass

        rate_constant = RateConstant()
        rate_constant.preexponential_factor = self['A'] * pow(1e-6, cm3_factor_exp)
        rate_constant.temperature_exponent = self['b']
        Ea = self['Ea']
        if units['activation-energy'] == 'K':
            rate_constant.activation_temperature = Ea
        elif units['activation-energy'] == 'cal/mol':
            rate_constant.activation_temperature = Ea * const.J_per_cal / const.R
        else:
            exit('Error: Unknown units for activation-energy')
        return rate_constant

    reactants = sum(reaction.reactants)
    if r.get('type') == "three-body":
        reactants += 1

    if r.get('rate-constant') or r.get('high-P-rate-constant'):
        reaction.rate_constant = rate_constant(r.get('rate-constant', r.get('high-P-rate-constant')), reactants - 1)
    else:
        reaction.rate_constant = rate_constant(r['rate-constants'][0], reactants - 1)

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
            exit(f"Error: unknown reaction: '{r}'")
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
        reaction.type = 'P-log'
        reaction.plog_k = {}
        reaction.num_P_conditions = len(r['rate-constants'])
        pressure_conditon_pattern = r"[-+]?\d*\.?\d+([eE][-+]?\d+)?"
        for idx in range(reaction.num_P_conditions):
            rate_constant_plog = rate_constant(r['rate-constants'][idx], reactants - 1)
            pressure_condition = r['rate-constants'][idx]['P']
            A = rate_constant_plog.preexponential_factor
            beta = rate_constant_plog.temperature_exponent
            E = rate_constant_plog.activation_temperature
            match = re.search(pressure_conditon_pattern, pressure_condition)
            pressure = float(match.group())*const.one_atm
            # Takes into account multiple rate constants for the same pressure.
            if (idx !=0) and (pressure == last_pressure):
                reaction.plog_k[pressure].append([A, beta, E, rate_constant_plog])
            else:
                reaction.plog_k[pressure] = [[A, beta, E, rate_constant_plog]]
            last_pressure = pressure 
        if '<=>' in r['equation'] or '=' in r['equation']:
            assert (r.get('reversible', True))  
            reaction.direction = 'reversible'
        elif re.search('[^<]=>', r['equation']):
            reaction.direction = 'irreversible'
        else:
            exit(f"Error: unknown reaction: '{r}'")
    else:
        exit(f"Error: unknown reaction: '{r}'")

    if r.get('efficiencies'):
        reaction.efficiencies = [
            r['efficiencies'].get(specie, r.get('default-efficiency', 1)) for specie in sp_names]
    return reaction


def compute_kr_unroll(r):
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
    kr = f"{'*'.join(gibbs_terms)}{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};"
    return kr


def arrhenius(rc):
    """
    Generates a string expression for an Arrhenius rate equation based on rate constant parameters.
    """
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


def arrhenius_diff(r):
    """
    Generates a string expression for the ratio of two Arrhenius rate constants (k0/k_inf).
    """
    A_inf, beta_inf, E_inf = (
        r.rate_constant.preexponential_factor,
        r.rate_constant.temperature_exponent,
        r.rate_constant.activation_temperature)
    A0, beta0, E0 = r.k0.preexponential_factor, r.k0.temperature_exponent, r.k0.activation_temperature
    expression = (
        f"exp("
        f"{f'{f(-E0 + E_inf)}*rcpT+' if (E0 - E_inf) != 0 else ''}"
        f"{f'{f(beta0 - beta_inf)}*lnT+' if (beta0 - beta_inf) != 0 else ''}{f(ln(A0) - ln(A_inf))})"
        if (A0 - A_inf) != 0 and ((beta0 - beta_inf) != 0 or (E0 - E_inf) != 0) else f'{f(A0 / A_inf)}'
    )
    return expression


def write_multiple_plog_rates_unroll(cg, kf_vals, num=None):
    """
    Writes rate constant computation in the case of multiple rates for a single pressure.
    Here "num" takes values None, 1, or 2:
        None - Pressure specified by user is equal to one of the pressure range bounds/limits.
        1    - Lower pressure value for that specific range.
        2    - Higher pressure value for that specific range.
    """
    if num == None:
        if len(kf_vals) == 1:
            cg.add_line(f"kf = {arrhenius(kf_vals[0][-1])};", 2)
        else:
            kf_str = []
            for j in range(len(kf_vals)):
                kf_str.append(f'{arrhenius(kf_vals[j][-1])}')
                if j == (len(kf_vals) - 1):
                    cg.add_line(f"kf = " + " + ".join(kf_str) + ";", 2)
    elif (num == 1) or (num == 2):
        if len(kf_vals) == 1:
            cg.add_line(f"cfloat kf{num} = {arrhenius(kf_vals[0][-1])};", 2)
        else:
            kf_str = []
            for j in range(len(kf_vals)):
                kf_str.append(f"{arrhenius(kf_vals[j][-1])}")
                if j == (len(kf_vals) - 1):
                    cg.add_line(f"cfloat kf{num} = " + " + ".join(kf_str) + ";", 2)
    return 0


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

    if r.type == 'elementary' or r.type == 'irreversible':
        cg.add_line(f'kf = {arrhenius(r.rate_constant)};', 1)
    elif r.type == 'three-body':
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"kf = {arrhenius(r.rate_constant)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"kf = {arrhenius(r.rate_constant)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"kf = {arrhenius(r.rate_constant)} * Cm;", 1)
    elif r.type == 'pressure-modification':
        cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
        cg.add_line(f"kf *= Pr/(1 + Pr);", 1)
    elif r.type == 'Troe':
        cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
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
        cg.add_line(f"kf *= Pr/(1 + Pr) * F;", 1)
    elif r.type == 'SRI':
        cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
        if hasattr(r, 'efficiencies'):
            cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
        elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
        else:
            cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
        cg.add_line(f"logPr = log10(Pr);", 1)
        cg.add_line(f"F = {r.sri.D}*pow({r.sri.A}*exp({-r.sri.B}*rcpT)+"
                    f"exp({-1. / (r.sri.C + FLOAT_MIN)}*T), 1./(1.+logPr*logPr))*pow(T, {r.sri.E});", 1)
        cg.add_line(f"kf *= Pr/(1 + Pr) * F;", 1)
    elif r.type == 'P-log':
        plog_vals = list(r.plog_k.keys())
        for plog_index in range(len(plog_vals) - 1):
            p1, p2 = plog_vals[plog_index], plog_vals[plog_index+1]
            kf1_vals, kf2_vals = list(r.plog_k[p1]), list(r.plog_k[p2])
            lnp1, lnp2 = math.log(p1), math.log(p2)
            lnpdiff = lnp2 - lnp1
            rcp_lnpdiff = 1/lnpdiff
            if (plog_index == 0):
                cg.add_line(f"if((P > {p1}) && (P < {p2}))", 1)
                cg.add_line(f"{{", 1)
            else:
                cg.add_line(f"}} else if ((P > {p1}) && (P < {p2})){{", 1)
            
            write_multiple_plog_rates_unroll(cg, kf1_vals, num=1)
            write_multiple_plog_rates_unroll(cg, kf2_vals, num=2)
            cg.add_line(f"kf = exp("
                        f" log(kf1) + ( log(kf2) - log(kf1) )*(lnP - {f(lnp1)})*{f(rcp_lnpdiff)});", 2)
            # Takes into account pressures outside the given range similar to Cantera.
            if (plog_index == 0):
                cg.add_line(f"}} else if (P <= {p1}){{", 1)
                write_multiple_plog_rates_unroll(cg, kf1_vals)
            elif (plog_index == (len(plog_vals) - 2)):
                cg.add_line(f"}} else if (P == {p1}){{", 1)
                write_multiple_plog_rates_unroll(cg, kf1_vals)
                cg.add_line(f"}} else if (P >= {p2}){{", 1)
                write_multiple_plog_rates_unroll(cg, kf2_vals)
                cg.add_line(f"}}", 1)
            else:
                cg.add_line(f"}} else if (P == {p1}){{", 1)
                write_multiple_plog_rates_unroll(cg, kf1_vals)
    else:
        exit(f"Error: --unroll-loops option doesn't support reaction type: '{r.type}'")

    phase_space = lambda reagents: '*'.join(
        '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
        coefficient != 0.)
    Rf = phase_space(r.reactants)
    Rr = phase_space(r.products)
    if r.type == 'irreversible' or r.direction == 'irreversible':
        cg.add_line(f"cR = kf * {Rf};", 1)
    else:
        pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
        if loop_gibbsexp:
            cg.add_line(f"kr = {compute_kr_unroll(r)}", 1)
        else:
            cg.add_line(f"kr = exp("
                        f"{'+'.join(imul(net, f'gibbs0_RT[{k}]') for k, net in enumerate(r.net) if net != 0)})"
                        f"{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};", 1)
        cg.add_line(f"cR = kf * ({Rf} - kr * {Rr});", 1)
    cg.add_line(f"#ifdef DEBUG")
    cg.add_line(f'printf("{idx + 1}: %+.15e\\n", cR);', 1)
    cg.add_line(f"#endif")
    for specie, net in enumerate(r.net):
        if net != 0:
            cg.add_line(f"wdot[{specie}] += {imul(net, 'cR')};", 1)
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

        if r.type == 'elementary' or r.type == 'irreversible':
            if idx == first_idx:
                cg.add_line(f'kf = {arrhenius(r.rate_constant)};', 1)
            else:
                cg.add_line(f'kf *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
        elif r.type == 'three-body':
            if idx == first_idx:
                cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f"kf *= "
                            f"{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};",
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"k_corr = kf * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"k_corr = kf * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"k_corr = kf * Cm;", 1)
        elif r.type == 'pressure-modification':
            if idx == first_idx:
                cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'kf *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
            cg.add_line(f"k_corr = kf * Pr/(1 + Pr);", 1)
        elif r.type == 'Troe':
            if idx == first_idx:
                cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'kf *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
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
            cg.add_line(f"k_corr = kf * Pr/(1 + Pr) * F;", 1)
        elif r.type == 'SRI':
            if idx == first_idx:
                cg.add_line(f"kf = {arrhenius(r.rate_constant)};", 1)
            else:
                cg.add_line(f'kf *= '
                            f'{r.rate_constant.preexponential_factor/previous_r.rate_constant.preexponential_factor};',
                            1)
            if hasattr(r, 'efficiencies'):
                cg.add_line(f"Pr = {arrhenius_diff(r)} * eff;", 1)
            elif not hasattr(r, 'efficiencies') and r.third_body_index >= 0:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Ci[{r.third_body_index}];", 1)
            else:
                cg.add_line(f"Pr = {arrhenius_diff(r)} * Cm;", 1)
            cg.add_line(f"logPr = log10(Pr);", 1)
            cg.add_line(f"F = {r.sri.D}*pow({r.sri.A}*exp({-r.sri.B}*rcpT)+"
                        f"exp({-1. / (r.sri.C + FLOAT_MIN)}*T), 1./(1.+logPr*logPr))*pow(T, {r.sri.E});", 1)
            cg.add_line(f"k_corr = kf * Pr/(1 + Pr) * F;", 1)
        else:
            exit(f"Error: --group-rxnUnroll option doesn't support reaction type: '{r.type}'")

        phase_space = lambda reagents: '*'.join(
            '*'.join([f'Ci[{specie}]'] * coefficient) for specie, coefficient in enumerate(reagents) if
            coefficient != 0.)
        Rf = phase_space(r.reactants)
        Rr = phase_space(r.products)
        if r.type == 'irreversible' or r.direction == 'irreversible':
            cg.add_line(f"cR = kf * {Rf};", 1)
        else:
            pow_C0_sum_net = '*'.join(["C0" if r.sum_net < 0 else 'rcpC0'] * abs(-r.sum_net))
            if loop_gibbsexp:
                cg.add_line(f"kr = {compute_kr_unroll(r)}", 1)
            else:
                cg.add_line(f"kr = exp("
                            f"{'+'.join(imul(net, f'gibbs0_RT[{k}]') for k, net in enumerate(r.net) if net != 0)})"
                            f"{f' * {pow_C0_sum_net}' if pow_C0_sum_net else ''};", 1)
            if r.type == 'elementary' or r.type == 'irreversible':
                cg.add_line(f"cR = kf * ({Rf} - kr * {Rr});", 1)
            else:
                cg.add_line(f"cR = k_corr * ({Rf} - kr * {Rr});", 1)
        cg.add_line(f"#ifdef DEBUG")
        cg.add_line(f'printf("{idx + 1}: %+.15e\\n", cR);', 1)
        cg.add_line(f"#endif")
        for specie, net in enumerate(r.net):
            if net != 0:
                cg.add_line(f"wdot[{specie}] += {imul(net, 'cR')};", 1)
        cg.add_line(f"")

    return cg.get_code()


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
    cg.add_line(f"__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_species_rates"
                f"(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4, "
                f"const cfloat rcpT, const cfloat P, const cfloat lnP, const cfloat* Ci, cfloat* wdot) ")
    cg.add_line(f"{{")
    cg.add_line(f"cfloat gibbs0_RT[{active_len}];", 1)
    expression = lambda a: (f"{f(a[5])} * rcpT + {f(a[0] - a[6])} + {f(-a[0])} * lnT + "
                            f"{f(-a[1] / 2)} * T + "
                            f"{f((1. / 3. - 1. / 2.) * a[2])} * T2 + "
                            f"{f((1. / 4. - 1. / 3.) * a[3])} * T3 + "
                            f"{f((1. / 5. - 1. / 4.) * a[4])} * T4")
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
        cg.add_line(f"cfloat kf, k_corr, kr, cR;", 1)
    else:
        cg.add_line(f"cfloat kf, kr, cR;", 1)
    cg.add_line(f"cfloat eff, Pr, logPr, F, logFcent, troe, troe_c, troe_n;", 1)
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
    pressure_plog = []
    for i in range(rxn_len):
        ids_old.append(i)
        A.append(rxn[i].rate_constant.preexponential_factor)
        beta.append(rxn[i].rate_constant.temperature_exponent)
        E_R.append(rxn[i].rate_constant.activation_temperature)
        if beta[-1] == 0 and E_R[-1] == 0:
            ids_E0B0.append(i)
        elif beta[-1] == -2 and E_R[-1] == 0:
            ids_E0Bneg2.append(i)
        elif beta[-1] == -1 and E_R[-1] == 0:
            ids_E0Bneg1.append(i)
        elif beta[-1] == 1 and E_R[-1] == 0:
            ids_E0B1.append(i)
        elif beta[-1] == 2 and E_R[-1] == 0:
            ids_E0B2.append(i)
        else:
            match_found = False
            for j in range(len(ids_EB)):
                if beta[-1] == beta[ids_old.index(ids_EB[j])] and E_R[-1] == E_R[ids_old.index(ids_EB[j])]:
                    ids_ErBr.append(i)
                    pos_ids_ErBr.append(j)
                    match_found = True
                    break
            if not match_found:
                ids_EB.append(i)
        # Process k0 if available
        # Indices for k0 are number of reactions + reaction index
        if hasattr(rxn[i], 'k0'):
            k0_ids = rxn_len + i
            ids_old.append(k0_ids)
            A.append(rxn[i].k0.preexponential_factor)
            beta.append(rxn[i].k0.temperature_exponent)
            E_R.append(rxn[i].k0.activation_temperature)
            if beta[-1] == 0 and E_R[-1] == 0:
                ids_E0B0.append(k0_ids)
            elif beta[-1] == -2 and E_R[-1] == 0:
                ids_E0Bneg2.append(k0_ids)
            elif beta[-1] == -1 and E_R[-1] == 0:
                ids_E0Bneg1.append(k0_ids)
            elif beta[-1] == 1 and E_R[-1] == 0:
                ids_E0B1.append(k0_ids)
            elif beta[-1] == 2 and E_R[-1] == 0:
                ids_E0B2.append(k0_ids)
            else:
                match_found = False
                for j in range(len(ids_EB)):
                    if beta[-1] == beta[ids_old.index(ids_EB[j])] and E_R[-1] == E_R[ids_old.index(ids_EB[j])]:
                        ids_ErBr.append(k0_ids)
                        pos_ids_ErBr.append(j)
                        match_found = True
                        break
                if not match_found:
                    ids_EB.append(k0_ids)
        # Process p-log if available
        # Indices for ks are number of reactions * k_idx + reaction index
        elif rxn[i].type == 'P-log':
            plog_ids = i
            for idx, key in enumerate(rxn[i].plog_k):                
                if key not in pressure_plog:
                    pressure_plog.append(key)
                # Adding multiple rates per single pressure functionality.
                num_plog_k = len(rxn[i].plog_k[key])
                for k in range(num_plog_k):
                    ids_old.append(plog_ids)
                    A.append(rxn[i].plog_k[key][k][0])
                    beta.append(rxn[i].plog_k[key][k][1])
                    E_R.append(rxn[i].plog_k[key][k][2])
                    if beta[-1] == 0 and E_R[-1] == 0:
                        ids_E0B0.append(plog_ids)
                    elif beta[-1] == -2 and E_R[-1] == 0:
                        ids_E0Bneg2.append(plog_ids)
                    elif beta[-1] == -1 and E_R[-1] == 0:
                        ids_E0Bneg1.append(plog_ids)
                    elif beta[-1] == 1 and E_R[-1] == 0:
                        ids_E0B1.append(plog_ids)
                    elif beta[-1] == 2 and E_R[-1] == 0:
                        ids_E0B2.append(plog_ids)
                    else:
                        match_found = False
                        for j in range(len(ids_EB)):
                            if beta[-1] == beta[ids_old.index(ids_EB[j])] and E_R[-1] == E_R[ids_old.index(ids_EB[j])]:
                                ids_ErBr.append(plog_ids)
                                pos_ids_ErBr.append(j)
                                match_found = True
                                break
                        if not match_found:
                            ids_EB.append(plog_ids)

                    plog_ids += rxn_len 

    # Group repeated constants for better vectorization
    unique_pos_ids_ErBr = list(dict.fromkeys(pos_ids_ErBr))

    # Move repeated constants that need to be calculated at the end
    ids_ErBr_values = [ids_EB[i] for i in pos_ids_ErBr]
    new_ids_EB_s = [ids_EB[i] for i in range(len(ids_EB)) if i not in unique_pos_ids_ErBr]
    new_ids_EB_e = [ids_EB[i] for i in unique_pos_ids_ErBr]
    ids_EB = new_ids_EB_s + new_ids_EB_e

    ids_rep = [i for i in range(len(pos_ids_ErBr)) if pos_ids_ErBr[i] in pos_ids_ErBr[:i]]
    new_pos_ids_ErBr_s = [pos_ids_ErBr[i] for i in range(len(pos_ids_ErBr)) if i not in ids_rep]
    new_pos_ids_ErBr_e = [pos_ids_ErBr[i] for i in ids_rep]
    new_ids_ErBr_values_s = [ids_ErBr_values[i] for i in range(len(ids_ErBr_values)) if i not in ids_rep]
    new_ids_ErBr_values_e = [ids_ErBr_values[i] for i in ids_rep]

    pos_ids_ErBr = new_pos_ids_ErBr_s + new_pos_ids_ErBr_e
    ids_ErBr_values = new_ids_ErBr_values_s + new_ids_ErBr_values_e
    new_ids_ErBr_s = [ids_ErBr[i] for i in range(len(ids_ErBr)) if i not in ids_rep]
    new_ids_ErBr_e = [ids_ErBr[i] for i in ids_rep]
    ids_ErBr = new_ids_ErBr_s + new_ids_ErBr_e

    # Group indices
    ids_E0Bsmall = ids_E0Bneg2 + ids_E0Bneg1 + ids_E0B1 + ids_E0B2
    ids_new = ids_EB + ids_E0B0 + ids_E0Bsmall + ids_ErBr
    assert len(ids_old) == len(ids_new)

    # Rearrange constants
    A_new = [A[ids_old.index(i)] for i in ids_new]
    beta_new = [beta[ids_old.index(i)] for i in ids_new]
    E_R_new = [E_R[ids_old.index(i)] for i in ids_new]

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
            cg.add_line(f'kf[i+{len(ids_EB)}] = A[i+{len(ids_EB)}];', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0Bneg2) > 0:
            start_ids_E0Bneg2 = len(ids_EB) + len(ids_E0B0)
            cg.add_line(f'// {len(ids_E0Bneg2)} rate constants with E_R = 0 and beta = -2 ', 1)
            cg.add_line(f'cfloat rcpT_2 = rcpT*rcpT;', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0Bneg2)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'kf[i+{start_ids_E0Bneg2}] = A[i+{start_ids_E0Bneg2}]*rcpT_2;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0Bneg1) > 0:
            start_ids_E0Bneg1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2)
            cg.add_line(f'// {len(ids_E0Bneg1)} rate constants with E_R = 0 and beta = -1 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0Bneg1)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'kf[i+{start_ids_E0Bneg1}] = A[i+{start_ids_E0Bneg1}]*rcpT;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0B1) > 0:
            start_ids_E0B1 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1)
            cg.add_line(f'// {len(ids_E0B1)} rate constants with E_R = 0 and beta = 1 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0B1)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'kf[i+{start_ids_E0B1}] = A[i+{start_ids_E0B1}]*T;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_E0B2) > 0:
            start_ids_E0B2 = len(ids_EB) + len(ids_E0B0) + len(ids_E0Bneg2) + len(ids_E0Bneg1) + len(ids_E0B1)
            cg.add_line(f'// {len(ids_E0B2)} rate constants with E_R = 0 and beta = 2 ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(ids_E0B2)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(f'kf[i+{start_ids_E0B2}] = A[i+{start_ids_E0B2}]*T2;', 2)
            cg.add_line(f'}}', 1)
        if len(ids_ErBr) > 0:
            start_ids_ErBr = len(ids_new) - len(ids_ErBr)
            cg.add_line(
                f'// {len(ids_ErBr)} rate constants with E_R and '
                f'beta the same as for other rate constants already computed ', 1)
            cg.add_line(f'for(unsigned int i=0; i<{len(unique_pos_ids_ErBr)}; ++i)', 1)
            cg.add_line(f'{{', 1)
            cg.add_line(
                f'kf[i+{start_ids_ErBr}] = A[i+{start_ids_ErBr}]*kf[i+{len(ids_EB) - len(unique_pos_ids_ErBr)}];', 2)
            cg.add_line(f'}}', 1)
            if (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) > 0:
                for i in range(len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)):
                    start_ids_ErBr_rep = len(ids_new) - (len(pos_ids_ErBr) - len(unique_pos_ids_ErBr)) + i
                    ids_k_rep = ids_EB.index(ids_ErBr_values[len(unique_pos_ids_ErBr) + i])
                    cg.add_line(f'kf[{start_ids_ErBr_rep}]=A[{start_ids_ErBr_rep}]*kf[{ids_k_rep}];', 1)
        return cg.get_code()

    ids_eff = []
    dic_unique_eff = {}

    def reorder_eff(r):
        cg = CodeGenerator()
        dic_eff = {}
        count = 0

        for i, ri in enumerate(r):
            if hasattr(ri, 'efficiencies'):
                ids_eff.append(i)
                dic_eff[count] = ri.efficiencies
                count += 1

        if count > 0:
            unique_ids_eff = []
            unique_count = 0

            for i in range(len(ids_eff)):
                test = True
                for j in range(i):
                    if dic_eff[j] == dic_eff[i]:
                        dic_unique_eff[i] = dic_unique_eff[j]
                        test = False
                        break
                if test:
                    dic_unique_eff[i] = unique_count
                    unique_ids_eff.append(ids_eff[i])
                    unique_count += 1

            for i in range(unique_count):
                ci = []
                efficiencies = r[unique_ids_eff[i]].efficiencies
                for specie, efficiency in enumerate(efficiencies):
                    if efficiency != 1.0:
                        if efficiency == 2:
                            ci.append(f"Ci[{specie}]")
                        elif efficiency == 0:
                            ci.append(f"-Ci[{specie}]")
                        else:
                            ci.append(f"{efficiency - 1.0}*Ci[{specie}]")
                cg.add_line(f"cfloat eff{i} = Cm + {' + '.join(ci)};", 1)

        return cg.get_code()

    # Correct k based on reaction type
    ids_er_rxn, ids_3b_rxn, ids_pd_rxn, ids_troe_rxn, ids_sri_rxn, ids_plog_rxn = [], [], [], [], [], []
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
        elif rxn[i].type == 'P-log':
            ids_plog_rxn.append(i)
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
                        f"kf[{ids_new.index(i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_tb.add_line(
                        f"kf[{ids_new.index(i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_tb.add_line(
                        f"kf[{ids_new.index(i)}] *= Cm;", 1)
            cg_tb.add_line("")

        # Pressure-dependent reactions
        cg_pd = CodeGenerator()
        if len(ids_pd_rxn) > 0:
            cg_pd.add_line("")
            cg_pd.add_line(f"//Correct k for pressure-dependent reactions", 1)
            for i in ids_pd_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_pd.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_pd.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_pd.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_pd.add_line(
                    f"kf[{ids_new.index(rxn_len + i)}] /= "
                    f"(1+ kf[{ids_new.index(rxn_len + i)}]/(kf[{ids_new.index(i)}]+ CFLOAT_MIN));", 1)
            cg_pd.add_line("")

        # Troe reactions
        cg_troe = CodeGenerator()
        if len(ids_troe_rxn) > 0:
            cg_troe.add_line("")
            cg_troe.add_line(f"//Correct k for troe reactions", 1)
            for i in ids_troe_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_troe.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_troe.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_troe.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_troe.add_line(
                    f"kf[{ids_new.index(rxn_len + i)}] /= (kf[{ids_new.index(i)}] + CFLOAT_MIN);", 1)
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
                f"cfloat logPr = log10(kf[ids_troe[i]] + CFLOAT_MIN);", 2)
            cg_troe.add_line(
                f"cfloat troe = (troe_c + logPr)/(troe_n - {f(0.14)}*(troe_c + logPr)+CFLOAT_MIN);", 2)
            cg_troe.add_line(
                f"cfloat F = pow(10, logFcent[i]/({f(1.0)} + troe*troe));", 2)
            cg_troe.add_line(f"kf[ids_troe[i]] /= ({f(1.)}+kf[ids_troe[i]]);", 2)
            cg_troe.add_line(f"kf[ids_troe[i]] *= F;", 2)
            cg_troe.add_line(f"}}", 1)
            for i in ids_troe_rxn:
                cg_troe.add_line(
                    f"kf[{ids_new.index(rxn_len + i)}] *= kf[{ids_new.index(i)}];", 1)
            cg_troe.add_line("")

        # SRI reaction
        cg_sri = CodeGenerator()
        if len(ids_sri_rxn) > 0:
            cg_sri.add_line("")
            cg_sri.add_line(f"//Correct k for SRI reactions", 1)
            for i in ids_sri_rxn:
                if hasattr(r[i], 'efficiencies'):
                    cg_sri.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= eff{dic_unique_eff[ids_eff.index(i)]};", 1)
                elif not hasattr(r[i], 'efficiencies') and r[i].third_body_index >= 0:
                    cg_sri.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Ci[{r[i].third_body_index}];", 1)
                else:
                    cg_sri.add_line(
                        f"kf[{ids_new.index(rxn_len + i)}] *= Cm;", 1)
                cg_sri.add_line(
                    f"kf[{ids_new.index(rxn_len + i)}] /= (kf[{ids_new.index(i)}] + CFLOAT_MIN);", 1)
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
                f"cfloat logPr = log10(kf[ids_sri[i]] + CFLOAT_MIN);", 2)
            cg_sri.add_line(f"cfloat F = sri_D[i]*pow(T, sri_E[i])*"
                                  f"pow(sri_A[i]*exp(-sri_B[i]*rcpT)+exp(-rcp_sri_C[i]*T), 1./(1.+logPr*logPr));", 2)
            cg_sri.add_line(f"kf[ids_sri[i]] /= (1.+kf[ids_sri[i]]);", 2)
            cg_sri.add_line(f"kf[ids_sri[i]] *= F;", 2)
            cg_sri.add_line(f"}}", 1)
            for i in ids_sri_rxn:
                cg_sri.add_line(
                    f"kf[{ids_new.index(rxn_len + i)}]*= kf[{ids_new.index(i)}];", 1)
            cg_sri.add_line("")

        # P-log reaction
        cg_plog = CodeGenerator()
        if len(ids_plog_rxn) > 0:

            def write_multiple_plog_rates_roll(p_idx, pressure, rxn_idx):
                num_k = len(rxn[rxn_idx].plog_k[pressure])
                if num_k == 1:
                    return f"kf[{ids_new.index(p_idx)}]"
                elif num_k > 1:
                    kf_str = ""
                    for k in range(num_k):  
                        if k > 0:
                            kf_str += " + "
                        kf_str += f"kf[{ids_new.index(p_idx+k*rxn_len)}]"
                    return kf_str

            cg_plog.add_line("")
            cg_plog.add_line(f"//Correct k for pressure-dependent Arrenhius (P-log) reactions", 1)

            plog_vals_rxns = {}
            lnplog_vals_rxns = {}
            rcp_lnpdiff_vals_rxns = {}
            lnplog_high_rxns = {}
            lnplog_low_rxns  = {}

            # Precomputing the values required in C++ kernels.
            for i in ids_plog_rxn:
                plog_vals = list(r[i].plog_k.keys())
                lnplog_vals = list(map(math.log, plog_vals))
                rcp_lnpdiff_vals = [1/(lnp2 - lnp1) for lnp1, lnp2 in zip(lnplog_vals[:-1], lnplog_vals[1:])]

                plog_vals_rxns[i] = plog_vals
                lnplog_vals_rxns[i] = lnplog_vals
                rcp_lnpdiff_vals_rxns[i] = rcp_lnpdiff_vals
                lnplog_low_rxns[i]  = min(lnplog_vals)
                lnplog_high_rxns[i] = max(lnplog_vals)

            for j1 in range(len(pressure_plog)):
                # First nested loop is to treat pressures within range.
                for j2 in range(1, len(pressure_plog)):
                    if (j1 == (len(pressure_plog) - 1)):
                        break
                    count = 0
                    rxn_count = 0
                    for i in ids_plog_rxn:
                        for plog_index in range(len(plog_vals_rxns[i]) - 1):
                            p1, p2 = plog_vals_rxns[i][plog_index], plog_vals_rxns[i][plog_index+1]
                            rcp_lnpdiff = rcp_lnpdiff_vals_rxns[i][plog_index]
                            num_k_p1 = len(r[i].plog_k[p1])
                            if plog_index == 0:
                                prev_num_k = 0
                            else:
                                p1_prev = plog_vals_rxns[i][plog_index-1]
                                prev_num_k += len(r[i].plog_k[p1_prev])
                            if pressure_plog[j1] == p1 and pressure_plog[j2] == p2:
                                if (j1 == 0 and j2 == 1) and count == 0:
                                    cg_plog.add_line(f"if((P > {pressure_plog[j1]}) && (P <= {pressure_plog[j2]}))", 1)
                                    cg_plog.add_line(f"{{", 1)
                                    count+=1
                                elif (j1 != 0 or j2 != 1) and count == 0:
                                    cg_plog.add_line(f"}}", 1)
                                    cg_plog.add_line(f"if ((P > {pressure_plog[j1]}) && "
                                                     f"(P <= {pressure_plog[j2]})){{", 1)
                                    count+=1

                                p1_idx = rxn_len*prev_num_k + i 
                                p2_idx = p1_idx + rxn_len*num_k_p1

                                cg_plog.add_line(f"kf[{ids_new.index(i)}] = exp("
                                                 f" log({write_multiple_plog_rates_roll(p1_idx, p1, i)}) + "
                                                 f"( log({write_multiple_plog_rates_roll(p2_idx, p2, i)}) - "
                                                 f"log({write_multiple_plog_rates_roll(p1_idx, p1, i)}) )*"
                                                 f"(lnP - {f(lnplog_vals_rxns[i][plog_index])})*{f(rcp_lnpdiff)});", 2)
                                break
                        rxn_count+=1
                                
            cg_plog.add_line(f"}}", 1)
            for j1 in range(len(pressure_plog)):
                rxn_count = 0 
                count_low  = 0
                count_high = 0
                # Now we treat pressures outside range.
                for i in ids_plog_rxn:
                    if math.log(pressure_plog[j1]) == lnplog_low_rxns[i]:
                        if count_low == 0:
                                cg_plog.add_line(f"if (P <= {pressure_plog[j1]}){{", 1)
                                count_low=1
                        p1_idx = i
                        cg_plog.add_line(f"kf[{ids_new.index(i)}]"
                                         f"= {write_multiple_plog_rates_roll(p1_idx, pressure_plog[j1], i)};", 2)
                        
                    elif math.log(pressure_plog[j1]) == lnplog_high_rxns[i]:
                        num_plog_k_prev = sum(len(value) for value in list(rxn[i].plog_k.values())[:-1])
                        if count_high == 0:
                                cg_plog.add_line(f"if (P > {pressure_plog[j1]}){{", 1)
                                count_high=1
                        p2_idx = rxn_len*num_plog_k_prev + i 
                        cg_plog.add_line(f"kf[{ids_new.index(i)}]"
                                         f"= {write_multiple_plog_rates_roll(p2_idx, pressure_plog[j1], i)};", 2)
                    
                    rxn_count+=1
                if count_low !=0 or count_high!=0:
                    cg_plog.add_line(f"}}", 1)

        # Combine all reactions
        reaction_corr = (cg_tb.get_code()  + cg_pd.get_code() + cg_troe.get_code() +
                         cg_sri.get_code() + cg_plog.get_code())
        return reaction_corr

    # Reorder reactions back to original
    def reorder_k(r):
        cg = CodeGenerator()
        for i in range(len(r)):
            if hasattr(rxn[i], 'k0'):
                cg.add_line(f"kf[{i}] = tmp[{ids_new.index(rxn_len + i)}];", 1)
            else:
                cg.add_line(f"kf[{i}] = tmp[{ids_new.index(i)}];", 1)
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

    # Compute kr only for reversible reactions
    ids_kr_nz = []
    for i in range(rxn_len):
        if rxn[i].direction != 'irreversible':
            ids_kr_nz.append(i)

    # Compute reverse rates
    def compute_kr(r):
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
                    f"kr[{ids_kr_nz.index(i)}] = "
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
                    cg.add_line(f"cR = kf[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)});", 1)
                else:
                    cg.add_line(f"cR = kf[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)});", 1)
            else:
                if hasattr(r[i], 'k0'):
                    cg.add_line(
                        f"cR = kf[{ids_new.index(rxn_len + i)}]*({phaseSpace(r[i].reactants)}-"
                        f"kr[{ids_kr_nz.index(i)}]*{phaseSpace(r[i].products)});", 1)
                else:
                    cg.add_line(
                        f"cR = kf[{ids_new.index(i)}]*({phaseSpace(r[i].reactants)}-"
                        f"kr[{ids_kr_nz.index(i)}]*{phaseSpace(r[i].products)});", 1)
            cg.add_line(f"#ifdef DEBUG")
            cg.add_line(f"printf(\"{i + 1}: %+.15e\\n\", cR);", 1)
            cg.add_line(f"#endif")
            for specie, net in enumerate(r[i].net):
                if net != 0:  # Only generate code for non-zero net changes
                    cg.add_line(f"wdot[{specie}] += {imul(net, 'cR')};", 1)
            cg.add_line(f"")
        return cg.get_code()

    #############
    # Write file
    #############

    cg = CodeGenerator()
    cg.add_line(f'#include <math.h>')
    cg.add_line(f'__KINETIX_DEVICE__ __KINETIX_INLINE__ void kinetix_species_rates'
                 f'(const cfloat lnT, const cfloat T, const cfloat T2, const cfloat T3, const cfloat T4,'
                 f' const cfloat rcpT, const cfloat P, const cfloat lnP, const cfloat* Ci, cfloat* wdot) ')
    cg.add_line(f'{{')
    cg.add_line(f"// Regrouping of rate constants to eliminate redundant operations", 1)
    var_str = ['A', 'beta', 'E_R']
    var = [A_new, beta_new_reduced, E_R_new_reduced]
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{f'alignas({align_width}) cfloat' if target=='c++17' else 'cfloat'} "
                 f"kf[{len(ids_new)}];", 1)
    cg.add_line(f"// {len(ids_EB)} rate constants for which an evaluation is necessary", 1)
    cg.add_line(f"for(unsigned int i=0; i<{len(ids_EB)}; ++i)", 1)
    cg.add_line(f"{{", 1)
    cg.add_line(f"cfloat blogT = beta[i]*lnT;", 2)
    cg.add_line(f"cfloat E_RT = E_R[i]*rcpT;", 2)
    cg.add_line(f"cfloat diff = blogT - E_RT;", 2)
    cg.add_line(f"kf[i] = exp(A[i] + diff);", 2)
    cg.add_line(f"}}", 1)
    cg.add_line(f"{set_k()}")
    cg.add_line("")
    cg.add_line(f"// Correct rate constants based on reaction type", 1)
    cg.add_line(f"cfloat Cm = 0;", 1)
    cg.add_line(f"for(unsigned int i=0; i<{sp_len}; ++i)", 1)
    cg.add_line(f"Cm += Ci[i];", 2)
    cg.add_line("")
    cg.add_line(f"{reorder_eff(rxn)}")
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
                 f"kr[{len(ids_kr_nz)}]; ", 1)
    cg.add_line(f"cfloat C0 = {f(const.one_atm / const.R)} * rcpT;", 1)
    cg.add_line(f"cfloat rcpC0 = {f(const.R / const.one_atm)} * T;", 1)
    cg.add_line(f"{compute_kr(rxn)}")
    cg.add_line("")
    cg.add_line(f"// Compute the reaction rates", 1)
    cg.add_line(f"cfloat cR;", 1)
    cg.add_line(f"{compute_rates(rxn)}")
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0
