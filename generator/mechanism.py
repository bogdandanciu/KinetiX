"""
Module for reading and generating the reaction mechanism file header.
"""

# Standard library imports
from functools import reduce

# Third-party imports
from ruamel.yaml import YAML

# Local imports
from utils.write_utils import CodeGenerator


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
