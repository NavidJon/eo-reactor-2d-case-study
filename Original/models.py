import cupy as cp
from config import data_type

class Reactor:
    reactor_scale = 10
    radius = 0.001 * reactor_scale
    height = 1.0 * reactor_scale
    void_fraction = data_type(0.40)
    heat_transfer_coefficient = data_type(200.0) # [W/m^3/K]
    coolant_temperature = data_type(300.0) # [K]

class Main_Reaction:
    KE1 = data_type(6.5)
    k01 = data_type(1.33e5)
    n1 = data_type(0.58)
    Ea1 = data_type(60.7e3)
    dH1 = data_type(-105e3); 

class Side_Reaction:
    KE2 = data_type(4.33)
    k02 = data_type(1.80e6)
    n2  = data_type(0.30)
    Ea2 = data_type(73.2e3)
    dH2 = data_type(-1327e3)

class Catalyst:
    particle_diameter = data_type(3.0e-2)
    density = data_type(1200.0)
    weight  = density * (data_type(1.0) - Reactor.void_fraction)
    specific_heat_capacity = data_type(900.0)

class Gas:
    ideal_gas_constant = data_type(8.314)
    dynamic_viscosity = data_type(3.0e-5)
    inlet_pressure = data_type(10.0e5)
    inlet_temperature = data_type(450)  # [K]
    inlet_superficial_velocity = data_type(0.5)
    density = data_type(1.0)
    specific_heat_capacity = data_type(1000.0)
    thermal_conductivity = data_type(0.1)
    thermal_diffusivity = data_type(thermal_conductivity / (density * specific_heat_capacity))  # [m^2/s]
    mass_diffusivity = data_type(1.5e-5)   # [m^2/s]

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
    initial_concentration = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=data_type)
    inlet_concentrations = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=data_type)
    molecular_weight = cp.asarray([0.028, 0.032, 0.044, 0.018, 0.044, 0.016], dtype=data_type)
