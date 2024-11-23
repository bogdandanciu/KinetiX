"""
Utility functions for code generation.
"""

# Standard library imports
import os
from pathlib import Path

# Local imports
from . import general_utils as gutils


class CodeGenerator:
    """
    A utility class for generating and writing structured code to files.
    """
    def __init__(self, indent='  '):
        self.lines = []
        self.si = indent  # single indent
        self.di = 2 * indent  # double indent
        self.ti = 3 * indent  # triple indent
        self.new_line = '\n'  # new line

    def add_line(self, line, level=0):
        """Add a line of code with optional indentation."""
        indented_line = f"{self.si * level}{line}"
        self.lines.append(indented_line)

    def get_code(self):
        """Return the complete code as a single string."""
        return '\n'.join(self.lines)

    def write_to_file(self, output_dir, module_name):
        """Write the generated code to a specified file."""
        content = self.get_code()
        # Create the output directory if it doesn't already exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_path = os.path.join(output_dir, module_name)
        with open(file_path, 'w') as file:
            updated_content = (content.replace('- -', '+ ')
                                      .replace('+ -', '- ')
                                      .replace('+-', '-')
                                      .replace('exp(', '__KINETIX_EXP__(')
                                      .replace('pow(', '__KINETIX_POW__(')
                                      .replace('log10(', '__KINETIX_LOG10__(')
                                      .replace('log(', '__KINETIX_LOG__('))
            file.write(updated_content)


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
            cg.add_line(f"{vtype} cfloat {var_str[i]}[{len(var[i])}] = {{{gutils.f_sci_not(var[i])}}};", 1)
            cg.add_line("")
    else:
        cg.add_line(f"{vtype} cfloat {var_str}[{len(var)}] = {{{gutils.f_sci_not(var)}}};", 1)

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


def sum_of_list(lt):
    """
    Calculate the sum of all elements in a list.
    """

    total = 0
    for val in lt:
        total = total + val
    return total


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
