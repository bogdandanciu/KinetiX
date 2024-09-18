"""
Generate source code files for computing chemical reaction rates,
thermodynamic properties, and mixture-averaged transport coefficients.
"""

# Local import
from . import mechanism as mech
from . import mix_transport as trans
from . import reaction_rates as reac
from . import thermodynamics as thermo
from ..utils import general_utils as gutils


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
    model = mech.read_mechanism_yaml(mech_file)

    # Initialize reactions to find inert species
    species_names_init = [i['name'] for i in model['species']]
    reactions_init = [reac.get_reaction_from_model(species_names_init, model['units'], reaction)
                      for reaction in model['reactions']]
    (inert_sp, active_sp) = gutils.partition(
        lambda specie: any([reaction.net[species_names_init.index(specie['name'])] != 0
                            for reaction in reactions_init]), model['species'])
    # Load species and reactions with new indexing (inert species at the end)
    species = trans.get_species_from_model(active_sp+inert_sp, rcp_diffcoeffs)
    species_names = species.sp_names
    reactions = [reac.get_reaction_from_model(species_names, model['units'], reaction)
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
        gutils.set_precision('FP32')
        rates_file = 'frates.inc'
        enthalpy_file = 'fenthalpy_RT.inc'
        heat_capacity_file = 'fheat_capacity_R.inc'
        conductivity_file = 'fconductivity.inc'
        viscosity_file = 'fviscosity.inc'
        diffusivity_file = 'fdiffusivity.inc'
    else:
        gutils.set_precision('FP64')
        rates_file = 'rates.inc'
        enthalpy_file = 'enthalpy_RT.inc'
        heat_capacity_file = 'heat_capacity_R.inc'
        conductivity_file = 'conductivity.inc'
        viscosity_file = 'viscosity.inc'
        diffusivity_file = 'diffusivity.inc'

    if header_only:
        mech.write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
    else:
        mech.write_file_mech(mech_file, output_dir, species_names, species_len, active_sp_len, reactions_len, Mi)
        if unroll_loops:  # Unrolled code
            reac.write_file_rates_unroll(rates_file, output_dir, loop_gibbsexp, group_rxnunroll,
                                         reactions, active_sp_len, species_len, species.thermo)
            thermo.write_file_enthalpy_unroll(enthalpy_file, output_dir,
                                              species_len, species.thermo)
            thermo.write_file_heat_capacity_unroll(heat_capacity_file, output_dir,
                                            species_len, species.thermo)
            if transport:
                trans.write_file_conductivity_unroll(conductivity_file, output_dir,
                                                     transport_polynomials, species_names)
                trans.write_file_viscosity_unroll(viscosity_file, output_dir, group_vis,
                                                  transport_polynomials, species_names, species_len, Mi)
                if nonsymDij:
                    trans.write_file_diffusivity_nonsym_unroll(diffusivity_file, output_dir, rcp_diffcoeffs,
                                                               transport_polynomials, species_names, species_len, Mi)
                else:
                    trans.write_file_diffusivity_unroll(diffusivity_file, output_dir, rcp_diffcoeffs,
                                                        transport_polynomials, species_len, Mi)
        else:  # Rolled code
            reac.write_file_rates_roll(rates_file, output_dir, align_width, target,
                                       species.thermo, species_len, reactions, reactions_len)
            thermo.write_file_enthalpy_roll(enthalpy_file, output_dir, align_width, target,
                                            species.thermo, species_len)
            thermo.write_file_heat_capacity_roll(heat_capacity_file, output_dir, align_width, target,
                                                 species.thermo, species_len)
            if transport:
                trans.write_file_conductivity_roll(conductivity_file, output_dir, align_width, target,
                                                   transport_polynomials, species_len)
                trans.write_file_viscosity_roll(viscosity_file, output_dir, align_width, target,
                                                transport_polynomials, species_len, Mi)
                if nonsymDij:
                    trans.write_file_diffusivity_nonsym_roll(diffusivity_file, output_dir,
                                                             align_width, target, rcp_diffcoeffs,
                                                             transport_polynomials, species_len, Mi)
                else:
                    trans.write_file_diffusivity_roll(diffusivity_file, output_dir,
                                                      align_width, target, rcp_diffcoeffs,
                                                      transport_polynomials, species_len, Mi)

    return 0

