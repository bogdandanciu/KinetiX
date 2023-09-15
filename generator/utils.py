"""
Module containing utility functions and variables.
"""

# Standard libraries
from numpy import polynomial
from numpy import square as sq
from numpy import isfinite as is_finite
from numpy import single
from pathlib import Path
from itertools import tee, filterfalse, repeat


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


def sum_of_list(lt):
    """
    Calculate the sum of all elements in a list.
    """

    total = 0
    for val in lt:
        total = total + val
    return total


def partition(pred, iterable):
    """
    Partition the elements of an iterable into two lists based on the given
    predicate.
    """

    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))


""" Variable representing the floating point precision of generated files."""
precision = 64


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
    if precision == 64:
        if c == 0.:
            return '0.'
        if c == 1.:
            return '1.'
        return repr(c)
    elif precision == 32:
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
    if precision == 64:
        if hasattr(x, '__iter__'):
            return ', '.join(['{:.16e}'.format(i) for i in x])
        else:
            return "{:.16e}".format(x)
    elif precision == 32:
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


def polynomial_regression(X, Y, degree=4):
    """
    Perform polynomial regression to fit a polynomial curve to data points.
    """

    return polynomial.polynomial.polyfit(X, Y, deg=degree, w=[1 / sq(y) for y in Y])


def write_module(output_dir, module_name, content):
    """
    Write a module file to the specified output directory.
    """
    Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)
    open(f'{output_dir}/{module_name}', 'w').write(
        content.replace('- -', '+ ').replace('+ -', '- ').replace('+-', '-').replace('exp(', '__NEKRK_EXP__('))


def code(lines):
    """
    Join a list of code lines into a single code block.
    """
    return '\n'.join(lines)


""" Constant representing a new line character."""
new_line = '\n'

""" Indentation levels for output code."""
si = '    '  # single indentation
di = si+si  # double indentation
ti = si+si+si  # triple indentation
qi = si+si+si+si  # quadruple indentation
