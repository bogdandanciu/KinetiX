#!/bin/env python3.9
from sys import argv, stderr
from cantera import Solution
gas = Solution(argv[1])
from ruamel.yaml import YAML
state = YAML().load(open(argv[2]))
temperature = state['temperature']
pressure = state['pressure']
X = state['X']
gas.TPX = (temperature, pressure, X)
#print(gas.TPX)
T = temperature
kB = 1.380649e-23 #* J/kelvin
NA = 6.02214076e23 #/mole
R = kB*NA
print('\n'.join([f'{i}: {v*1000:.17e}' for (i ,v) in enumerate(gas.net_rates_of_progress)])) # kmol -> mol
#print('\n'.join([f'{w*1000} {h/1000/(R*T):.17e}' for (w ,h) in zip(gas.net_production_rates, gas.partial_molar_enthalpies)])) # /kmol -> /mol
#print(gas.heat_release_rate/(R*T))
