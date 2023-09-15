#!/bin/env python3.9
from sys import argv, stderr
from cantera import Solution
gas = Solution(argv[1])
if True:
    from ruamel.yaml import YAML
    state = YAML().load(open(argv[2]))
    temperature = state['temperature']
    pressure = state['pressure']
    X = state['X']
    print(temperature, pressure, X, file=stderr)
else:
    from random import uniform
    temperature = uniform(1000, 2800)
    pressure = uniform(1, 101325)
    X = [uniform(0, 1) for name in gas.species_names]
#print([name for X, name in zip(X, gas.species_names) if X != 0.], file=stderr)
gas.TPX = (temperature, pressure, X)
#print(' '.join(gas.species_names))
#print('\n'.join([f'{i}: {e}' for i,e in enumerate(gas.net_rates_of_progress)]))
#print(' '.join(f'{r}' for r in gas.net_production_rates))
#print(f"{gas.viscosity} {gas.thermal_conductivity} {' '.join([f'{x:e}' for x in gas.mix_diff_coeffs])}")
if False:
    print(', '.join([f'{name}: {rate:+.3f}' for (name, rate) in zip(gas.species_names, gas.net_production_rates) if rate != 0]+[f'HRR: {gas.heat_release_rate:.3e}']))
    #print(f"μ: {gas.viscosity:.4}, λ: {gas.thermal_conductivity:.4}, D: {' '.join(f'{x:.3e}' for x in gas.mix_diff_coeffs)}")
else:
    # Unlabeled data is error prone but simpler to parse in raw C++
    #species_names, species_molar_mass #kg/kmol, temperature, pressure, amount_proportions, molar_mass, density, mean_molar_heat_capacity, molar_heat_capacity, net_production_rates, volumetric_heat_release_rate, conductivity, viscosity, density_diffusivity
    print(f"""{' '.join(gas.species_names)}
{' '.join(repr(w*1e-3) for w in gas.molecular_weights)}
{repr(temperature)}
{repr(pressure)}
{' '.join(repr(X[specie]) for specie in gas.species_names)}
{repr(gas.density_mass)}
{repr(gas.cp_mole*1e-3)}
{' '.join(repr(c*1e-3) for c in gas.partial_molar_cp)}
{' '.join(repr(d*1e3) for d in gas.net_production_rates)}
{repr(gas.heat_release_rate)}
{repr(gas.thermal_conductivity)}
{repr(gas.viscosity)}
{' '.join(repr(gas.density_mass*diffusivity) for diffusivity in gas.mix_diff_coeffs)}""")
