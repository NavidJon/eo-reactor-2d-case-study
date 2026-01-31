import cupy as cp

class Reactor:
    reactor_scale = 10
    radius = 0.001 * reactor_scale
    height = 1.0 * reactor_scale
    void_fraction = 0.40
    heat_transfer_coefficient = 200.0 # [W/m^3/K]
    coolant_temperature = 300.0 # [K]

class Main_Reaction:
    KE1 = 6.5
    k01 = 1.33e5
    n1 = 0.58
    Ea1 = 60.7e3
    dH1 = -105e3

class Side_Reaction:
    KE2 = 4.33
    k02 = 1.80e6
    n2  = 0.30
    Ea2 = 73.2e3
    dH2 = -1327e3

class Catalyst:
    particle_diameter = 3.0e-2
    density = 1200.0
    weight  = density * (1.0) - Reactor.void_fraction
    specific_heat_capacity = 900.0

class Gas:
    ideal_gas_constant = 8.314
    dynamic_viscosity = 3.0e-5
    inlet_pressure = 10.0e5
    inlet_temperature = 450  # [K]
    inlet_superficial_velocity = 0.5
    density = 1.0
    specific_heat_capacity = 1000.0
    thermal_conductivity = 0.1
    thermal_diffusivity = thermal_conductivity / (density * specific_heat_capacity)  # [m^2/s]
    mass_diffusivity = 1.5e-5   # [m^2/s]

class Species:
    names = [
        'Ethylene',
        'Oxygen',
        'Ethylene Oxide',
        'Water',
        'Carbon Dioxide',
        'Methane'
    ]
    count = len(names)
    initial_concentration = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0])
    inlet_concentrations = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0])
    molecular_weight = cp.asarray([0.028, 0.032, 0.044, 0.018, 0.044, 0.016])
