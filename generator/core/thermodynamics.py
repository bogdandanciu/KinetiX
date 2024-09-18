"""
Module containing routines for computing enthalpy and specific heat at constant pressure.
"""

# Local imports
from ..utils.general_utils import f
from ..utils.write_utils import (
    CodeGenerator,
    get_thermo_coeffs,
    get_thermo_prop,
    reorder_thermo_prop,
    write_const_expression,
    write_energy
)


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
                f"const cfloat T2, const cfloat T3, const cfloat T4, const cfloat rcpT, cfloat* h_RT) ")
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
                f"const cfloat rcpT, cfloat* cp_R) ")
    cg.add_line(f"{{")
    cg.add_line(f"//Integration coefficients", 1)
    cg.add_line(f"{write_const_expression(align_width, target, True, var_str, var)}")
    cg.add_line(f"{get_thermo_prop('cp_R', unique_temp_split, len_unique_temp_splits)}")
    cg.add_line(f"{reorder_thermo_prop('cp_R', unique_temp_split, ids_thermo_new, sp_len)}")
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
                f"const cfloat rcpT, cfloat* h_RT)")
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
                f"const cfloat rcpT, cfloat* cp_R)")
    cg.add_line(f"{{")
    expression = lambda a: (f'{f(a[0])} + {f(a[1])} * T + {f(a[2])} * T2 + '
                            f'{f(a[3])} * T3 + {f(a[4])} * T4')
    cg.add_line(f'{write_energy(f"cp_R[", sp_len, expression, sp_thermo)}')
    cg.add_line(f"}}")

    cg.write_to_file(output_dir, file_name)
    return 0
