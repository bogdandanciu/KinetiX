#!/usr/bin/env python3

"""
Generate source code files for computing chemical reaction rates,
thermodynamic properties, and mixture-averaged transport coefficients.
"""

# Standard library imports
import argparse

# Local imports
from mechanism import read_mechanism_yaml, write_file_mech
from mix_transport import (
    get_species_from_model,
    write_file_conductivity_roll,
    write_file_conductivity_unroll,
    write_file_diffusivity_nonsym_roll,
    write_file_diffusivity_nonsym_unroll,
    write_file_diffusivity_roll,
    write_file_diffusivity_unroll,
    write_file_viscosity_roll,
    write_file_viscosity_unroll,
)
from reaction_rates import (
    get_reaction_from_model,
    write_file_rates_roll,
    write_file_rates_unroll,
)
from thermodynamics import (
    write_file_enthalpy_roll,
    write_file_enthalpy_unroll,
    write_file_heat_capacity_roll,
    write_file_heat_capacity_unroll,
)
from utils.general_utils import partition, set_precision


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
