"""
General utility functions and variables.
"""

# Standard library imports
import argparse
from itertools import (
    filterfalse,
    repeat,
    tee)

# Third-party imports
from numpy import (
    isfinite as is_finite,
    polynomial,
    single
)


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


def cube(x):
    """
    Calculate the cube of a given value.
    """
    return x * x * x


def flatten_list(t):
    """
    Flatten a nested list into a single list.
    """
    return [item for sublist in t for item in sublist]


def partition(pred, iterable):
    """
    Partition the elements of an iterable into two lists based on the given
    predicate.
    """

    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))


""" Variable representing the floating point precision of generated files."""
precision = 'FP64'


def set_precision(new_value):
    """
    Set the floating point precision value for output files.
    """
    global precision
    precision = new_value


def f(c):
    """
    Convert numerical value to a string representation, depending
    on the global precision setting.
    """

    global precision
    if precision == 'FP64':
        if c == 0.:
            return '0.'
        if c == 1.:
            return '1.'
        return f'{c}'
    elif precision == 'FP32':
        if not is_finite(single(c)):
            print(f'/!\\ WARNING: {c} constant already overflows single precision')
        return f'{c}f'
    else:
        raise ValueError


def f_sci_not(x):
    """
    Convert numerical values to their string representation in a scientific
    notation format, taking into account the global precision setting.
    """

    global precision
    if precision == 'FP64':
        if hasattr(x, '__iter__'):
            return ', '.join(['{:.16e}'.format(i) for i in x])
        else:
            return "{:.16e}".format(x)
    elif precision == 'FP32':
        if hasattr(x, '__iter__'):
            return ', '.join(['{:.16e}f'.format(i) for i in x])
        else:
            return "{:.16e}f".format(x)
    else:
        raise ValueError


def imul(c, v):
    """
    Compute the product of a coefficient and a variable expression
    and returns a formatted string representation.
    """

    if c == 0:
        return None
    else:
        return f"{'' if c == 1 else '-' if c == -1 else f'{c}*'}{v}"


def prod_of_exp(c, v):
    """
    Calculate the product of exponentiations involving a variable expression
    `v` raised to integer powers specified by coefficients in the list `c`.
    """

    for _c in c:
        assert (_c == int(_c))
    c = [int(c) for c in c]
    (div, num) = partition(lambda c: c[1] > 0, [(i, c) for i, c in enumerate(c) if c != 0])
    num = '*'.join(flatten_list([repeat(f'{v}[{i}]', max(0, c)) for i, c in num]))
    div = '*'.join(flatten_list([repeat(f'{v}[{i}]', max(0, -c)) for i, c in div]))
    if (num == '') and (div == ''):
        return '1.'
    elif div == '':
        return num
    elif num == '':
        return f'{f(1.)}/({div})'
    else:
        return f'{num}/({div})'


def prod_of_exp_rcp(c, v, rcp):
    """
    Calculate the reciprocal for the product of exponentiations involving
    variables.
    """

    for _c in c:
        assert (_c == int(_c))
    c = [int(c) for c in c]
    (div, num) = partition(lambda c: c[1] > 0, [(i, c) for i, c in enumerate(c) if c != 0])
    num = '*'.join(flatten_list([repeat(f'{v}{i}', max(0, c)) for i, c in num]))
    div = '*'.join(flatten_list([repeat(f'{rcp}{i}', max(0, -c)) for i, c in div]))
    if (num == '') and (div == ''):
        return '1.'
    elif div == '':
        return num
    elif num == '':
        return div
    else:
        return f'{num}*{div}'
    
    
def polynomial_regression(X, Y, degree=4, weights=None):
    """
    Perform polynomial regression to fit a polynomial curve to data points.
    """

    if weights is None:
        weights = [1 / abs(y) for y in Y]
    return polynomial.polynomial.polyfit(X, Y, deg=degree, w=weights)


""" Maximum and minimum floating point values"""
FLOAT_MAX = 1e300
FLOAT_MIN = 1e-300

