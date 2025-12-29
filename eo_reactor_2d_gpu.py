from functools import partial
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time as pytime

# -----------------------------
# Toggle time check prints
# -----------------------------
PROFILE = False

class GPUTimer:
    def __init__(self): 
        self.t = {}

    def start_timing(self, section_name):
        if not PROFILE: 
            return
        start_event = cp.cuda.Event()
        stop_event = cp.cuda.Event()

        start_event.record()
        self.t[section_name] = [start_event, stop_event]

    def stop_timing(self, section_name, accumulated_time):
        if not PROFILE: 
            return
        start_event, stop_event = self.t[section_name]
        stop_event.record()
        stop_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, stop_event)  # milliseconds
        accumulated_time[section_name] = accumulated_time.get(section_name, 0.0) + elapsed_time

class NoOperationTimer:
    def start_timing(self, *a, **k):
        pass
    def stop_timing(self, *a, **k):
        pass

timer = GPUTimer() if PROFILE else NoOperationTimer()
elapsed_time_accumulator = {}

# =========================
# Precision for GPU calculations 
# =========================
data_type = cp.float32
cp.cuda.runtime.setDevice(0)
if data_type == cp.float32:
    input_data_type = 'float32'   # for Elementwise/RawKernel signatures
    cuda_data_type  = 'float'     # for CUDA device code
elif data_type == cp.float64:
    input_data_type = 'float64'
    cuda_data_type  = 'double'
else:
    raise ValueError("Must be 32 or 64")

# =========================
# Reactor Tube Geometry
# =========================
class Reactor:
    reactor_scale = 10
    radius = 0.001 * reactor_scale
    height = 1.0 * reactor_scale
    void_fraction = data_type(0.40)
    heat_transfer_coefficient = data_type(200.0)  # [W/m^3/K]

# =========================
# Grid Mesh
# =========================
mesh_scale = 40

number_radial_cells = 1 * mesh_scale
number_axial_cells = 10 * mesh_scale

radial_cell_size = data_type(Reactor.radius / number_radial_cells)
axial_cell_size = data_type(Reactor.height / number_axial_cells)

radial_cell_center_position = cp.linspace(
    radial_cell_size / 2, 
    Reactor.radius - radial_cell_size / 2, 
    number_radial_cells,
    data_type=data_type
)
axial_cell_center_position = cp.linspace(
    axial_cell_size / 2,
    Reactor.height - axial_cell_size / 2,
    number_axial_cells,
    data_type=data_type
)

# =========================
# Time Stepping
# =========================
time_final = 20
number_of_time_steps = 20000
time_step_size = data_type(time_final / number_of_steps)

# =========================
# Mass transport properties
# =========================
diffusivity = data_type(1.5e-5)   # [m^2/s]
pressure = data_type(1.0e5)    # [Pa]

# =========================
# Reaction properties
# =========================
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

# =========================
# Catalyst parameters
# =========================
class Catalyst:
    particle_diameter = data_type(3.0e-2)
    density = data_type(1200)
    weight  = density * (1-Reactor.void_fraction)
    specific_heat_capacity = data_type(900.0)

# =========================
# Gas parameters
# =========================
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

# =========================
# Energy balance parameters
# =========================
coolant_temperature = data_type(300.0)                   # [K]


# =========================
# Precompute geometric factors
# =========================
interior_radial_cell_center_position = radial_cell_center_position[1:-1][None, :, None]          # (1, number_radial_cells-2, 1)
inverse_interior_radial_cell_center_position = data_type(1.0) / interior_radial_cell_center_position          # (1, number_radial_cells-2, 1)
inverse_interior_radial_cell_center_position_line = inverse_interior_radial_cell_center_position[0]              # (number_radial_cells-2, 1)

# =========================
# Species setup
# =========================
class Species:
    names = ['Ethylene','Oxygen','Ethylene Oxide','Water','Carbon Dioxide','Methane']
    count = len(names)
    initial_concentration = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=data_type)
    inlet_concentrations = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=data_type)
    molecular_weight = cp.asarray([0.028, 0.032, 0.044, 0.018, 0.044, 0.016], dtype=data_type)

species_concentration_field = cp.zeros(
    (
     Species.count,
     number_radial_cells,
     number_axial_cells
    ),
    dtype=data_type
)
species_concentration_field += Species.initial_concentration[:, None, None]
temperature_field = cp.full(
    (
     number_radial_cells,
     number_axial_cells
    ),
    data_type(Gas.inlet_temperature),
    dtype=data_type
)

# =========================
# Precomputed energy constants
# =========================
volumetric_heat_capacity = data_type(
    Reactor.void_fraction * Gas.density * Gas.specific_heat_capacity 
    + Catalyst.weight * Catalyst.specific_heat_capacity
    )

# =========================
# Fused reaction (for end-of-run selectivity plots)
# =========================
@cp.fuse()
def reaction_terms_T(
    concentration_ethylene,
    concentration_oxygen,
    temperature,
    KE1, KE2,
    n1, n2,
    k01, k02,
    Ea1, Ea2,
    ideal_gas_constant,
    catalyst_weight
):
    ethylene_partial_pressure = cp.maximum(
        (concentration_ethylene * ideal_gas_constant * temperature)/data_type(1e5),
        data_type(0.0)
    )

    oxygen_partial_pressure = cp.maximum(
        (concentration_oxygen  * ideal_gas_constant * temperature)/data_type(1e5),
        data_type(0.0)
    )

    k1 = k01 * cp.exp(-Ea1 / (ideal_gas_constant * temperature))
    k2 = k02 * cp.exp(-Ea2 / (ideal_gas_constant * temperature))

    main_reaction_denominator = cp.maximum(
        cp.square(data_type(1.0) + KE1 * ethylene_partial_pressure),
        data_type(1e-20)
    )

    side_reaction_denominator = cp.maximum(
        cp.square(data_type(1.0) + KE2 * ethylene_partial_pressure),
        data_type(1e-20),
    )

    main_reaction_rate = (
        k1 
        * ethylene_partial_pressure 
        * (oxygen_partial_pressure ** n1) 
        * catalyst_weight 
        / main_reaction_denominator
    )

    side_reaction_rate = (
        k2
        * ethylene_partial_pressure
        * (oxygen_partial_pressure ** n2)
        * catalyst_weight
        / side_reaction_denominator
    )

    ethylene_source_term = - (main_reaction_rate + side_reaction_rate)
    oxygen_source_term = -(data_type(0.5)*main_reaction_rate + data_type(3.0)*side_reaction_rate)
    ethylene_oxide_source_term = main_reaction_rate
    water_source_term = data_type(2.0)*side_reaction_rate
    carbon_dioxide_source_term = data_type(2.0)*side_reaction_rate

    return ethylene_source_term, oxygen_source_term, ethylene_oxide_source_term, water_source_term, carbon_dioxide_source_term, main_reaction_rate, side_reaction_rate

# =========================
# Ergun function
# =========================
def ergun_pressure_velocity_profile(
        species_concentration_field,
        inlet_pressure,
        bed_void_fraction,
        catalyst_particle_diameter,
        dynamic_viscosity_profile,
        temperature_profile_density,
        ideal_gas_constant,
        species_molecular_weight,
        inlet_concentrations,
        axial_cell_size,
        inlet_velocity,
        iterations=1,
        minimum_pressure=data_type(0.5e5)
):
    number_of_species, number_radial_cells, number_axial_cells = species_concentration_field.shape

    cross_section_average_concentration_profile = cp.mean(
        species_concentration_field,
        axis=1,
        dtype=data_type
    )

    total_concentration_profile = (
        cp.sum(cross_section_average_concentration_profile, axis=0, dtype=data_type)
        + data_type(1e-30)
    )

    mole_fraction_profile = (
        cross_section_average_concentration_profile
        / total_concentration_profile[None, :]
    )

    mixture_molecular_weight_profile = cp.sum(
        mole_fraction_profile * species_molecular_weight[:, None],
        axis=0,
        dtype=data_type
    )

    # inlet mixture props
    inlet_mole_fractions = (
        inlet_concentrations
        / (cp.sum(inlet_concentrations, dtype=data_type) + data_type(1e-30))
    )

    inlet_mixture_molecular_weight = cp.sum(
        inlet_mole_fractions * species_molecular_weight,
        dtype=data_type
    )

    inlet_temperature_density = (
        temperature_profile_density[0] 
        if hasattr(temperature_profile_density, "ndim") and temperature_profile_density.ndim == 1
        else temperature_profile_density
    )

    inlet_gas_density = (
        (inlet_pressure * inlet_mixture_molecular_weight)
        / (ideal_gas_constant * inlet_temperature_density)
    )

    # mass flux G fixed by inlet velocity
    inlet_mass_flux = inlet_gas_density * inlet_velocity

    # Ergun coefficients
    viscous_ergun_coefficient = (
        data_type(150.0)
        * (data_type(1.0) - bed_void_fraction) ** 2
        * dynamic_viscosity_profile
        / (bed_void_fraction**3 * catalyst_particle_diameter**2)
    )
    inertial_ergun_coefficient = (
        data_type(1.75)
        * (data_type(1.0) - bed_void_fraction)
        / (bed_void_fraction**3 * catalyst_particle_diameter)
    )

    # initial guess
    gas_density_profile = cp.maximum(
        (inlet_pressure * mixture_molecular_weight_profile) 
        / (ideal_gas_constant * temperature_profile_density),
        data_type(1e-20)
    )

    superficial_velocity_profile = (
        inlet_mass_flux / gas_density_profile
    )

    # simple Picard
    for _ in range(iterations):
        # dpdz
        axial_pressure_gradient_profile = (
            viscous_ergun_coefficient * superficial_velocity_profile
            + inertial_ergun_coefficient * gas_density_profile * cp.square(superficial_velocity_profile)
        )

        cumulative_pressure_drop_profile = cp.empty_like(axial_pressure_gradient_profile)

        if number_axial_cells > 0:
            cumulative_pressure_drop_profile[0] = (
                data_type(0.5) * axial_pressure_gradient_profile[0] * axial_cell_size
            )

        if number_axial_cells > 1:
            trapezoid_midpoint = (
                data_type(0.5)
                * (axial_pressure_gradient_profile[1:] + axial_pressure_gradient_profile[:-1])
                * axial_cell_size
            )
            cumulative_pressure_drop_profile[1:] = (
                cumulative_pressure_drop_profile[0]
                + cp.cumsum(trapezoid_midpoint, dtype=data_type)
            )

        pressure_profile = cp.maximum(
            inlet_pressure - cumulative_pressure_drop_profile,
            minimum_pressure
        )
        
        updated_gas_density_profile = cp.maximum(
            (pressure_profile * mixture_molecular_weight_profile)
            / (ideal_gas_constant * temperature_profile_density),
            data_type(1e-20)
        )

        # Damping step to prevent oscillations
        relaxation_factor = data_type(0.4)
        gas_density_profile = (
            (data_type(1.0) - relaxation_factor) * gas_density_profile
            + relaxation_factor * updated_gas_density_profile
        )

        superficial_velocity_profile = inlet_mass_flux / gas_density_profile

    return pressure_profile, superficial_velocity_profile

# =========================
# Macro-step fused kernel in CUDA C++ kernel code
# =========================
macro_step_source_code = f'''
extern "C" __global__
void macro_step(
    const int number_of_species,
    const int number_of_radial_cells,
    const int number_of_axial_cells,
    const int number_of_substeps,

    const {cuda_data_type}* __restrict__ species_concentration_input,
          {cuda_data_type}* __restrict__ species_concentration_output,

    const {cuda_data_type}* __restrict__ temperature_input,
          {cuda_data_type}* __restrict__ temperature_output,

    const {cuda_data_type}* __restrict__ inverse_interior_radial_cell_center_positions,
    const {cuda_data_type}* __restrict__ left_face_velocity,
    const {cuda_data_type}* __restrict__ right_face_velocity,
    const {cuda_data_type}* __restrict__ interior_cell_center_pressure,

    const {cuda_data_type} radial_cell_size,
    const {cuda_data_type} axial_cell_size,
    const {cuda_data_type} substep_time_step,

    const {cuda_data_type} thermal_diffusivity_time_over_radial_cell_size_squared,
    const {cuda_data_type} half_thermal_diffusivity_time_over_radial_cell_size,
    const {cuda_data_type} thermal_diffusivity_time_over_axial_cell_size_squared,
    const {cuda_data_type} substep_time_over_axial_cell_size,

    const {cuda_data_type} heat_transfer_sink_coefficient_time,

    const {cuda_data_type} reference_mass_diffusivity,
    const {cuda_data_type} reference_temperature,
    const {cuda_data_type} reference_pressure,

    const {cuda_data_type} adsorption_equilibrium_constant_main_reaction,
    const {cuda_data_type} adsorption_equilibrium_constant_side_reaction,
    const {cuda_data_type} oxygen_reaction_order_main_reaction,
    const {cuda_data_type} oxygen_reaction_order_side_reaction,
    const {cuda_data_type} pre_exponential_factor_main_reaction,
    const {cuda_data_type} pre_exponential_factor_side_reaction,
    const {cuda_data_type} activation_energy_main_reaction,
    const {cuda_data_type} activation_energy_side_reaction,

    const {cuda_data_type} ideal_gas_constant,
    const {cuda_data_type} catalyst_mass_per_volume,

    const {cuda_data_type} reaction_enthalpy_main_reaction,
    const {cuda_data_type} reaction_enthalpy_side_reaction,

    const {cuda_data_type} volumetric_heat_capacity,
    const {cuda_data_type} coolant_temperature
)
{{
    // Interior cell indices (skip boundary cells)
    const int axial_cell_index  = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int radial_cell_index = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (radial_cell_index >= number_of_radial_cells - 1 || axial_cell_index >= number_of_axial_cells - 1) return;

    const int axial_stride_for_row_major = number_of_axial_cells;
    const int species_stride_for_row_major = number_of_radial_cells * number_of_axial_cells;

    const int interior_axial_index  = axial_cell_index - 1;   // 0 .. number_of_axial_cells-3
    const int interior_radial_index = radial_cell_index - 1;  // 0 .. number_of_radial_cells-3

    const {cuda_data_type} inverse_radial_position =
        inverse_interior_radial_cell_center_positions[interior_radial_index];

    const {cuda_data_type} left_face_velocity_value  = left_face_velocity[interior_axial_index];
    const {cuda_data_type} right_face_velocity_value = right_face_velocity[interior_axial_index];
    const {cuda_data_type} cell_center_pressure_value = interior_cell_center_pressure[interior_axial_index];

    const {cuda_data_type} zero_value = ({cuda_data_type})0.0;
    const {cuda_data_type} half_value = ({cuda_data_type})0.5;
    const {cuda_data_type} two_value  = ({cuda_data_type})2.0;
    const {cuda_data_type} tiny_value = ({cuda_data_type})1e-30;

    const {cuda_data_type} pressure_scaling_factor = ({cuda_data_type})1e-5;

    const {cuda_data_type}* species_concentration_read = species_concentration_input;
          {cuda_data_type}* species_concentration_write = species_concentration_output;

    const {cuda_data_type}* temperature_read = temperature_input;
          {cuda_data_type}* temperature_write = temperature_output;

    for (int substep_index = 0; substep_index < number_of_substeps; ++substep_index) {{

        // ------------------------------------------------------------
        // 1) Advection + diffusion update for all species concentrations
        // ------------------------------------------------------------
        const int temperature_center_index =
            radial_cell_index * axial_stride_for_row_major + axial_cell_index;

        const {cuda_data_type} temperature_center_for_transport =
            temperature_read[temperature_center_index];

        const {cuda_data_type} local_mass_diffusivity =
            reference_mass_diffusivity
            * pow(temperature_center_for_transport / reference_temperature, ({cuda_data_type})1.75)
            * (reference_pressure / (cell_center_pressure_value + tiny_value));

        for (int species_index = 0; species_index < number_of_species; ++species_index) {{

            const int center_index =
                species_index * species_stride_for_row_major
                + radial_cell_index * axial_stride_for_row_major
                + axial_cell_index;

            const int radial_plus_index =
                species_index * species_stride_for_row_major
                + (radial_cell_index + 1) * axial_stride_for_row_major
                + axial_cell_index;

            const int radial_minus_index =
                species_index * species_stride_for_row_major
                + (radial_cell_index - 1) * axial_stride_for_row_major
                + axial_cell_index;

            const int axial_plus_index =
                species_index * species_stride_for_row_major
                + radial_cell_index * axial_stride_for_row_major
                + (axial_cell_index + 1);

            const int axial_minus_index =
                species_index * species_stride_for_row_major
                + radial_cell_index * axial_stride_for_row_major
                + (axial_cell_index - 1);

            const {cuda_data_type} concentration_center = species_concentration_read[center_index];
            const {cuda_data_type} concentration_radial_plus = species_concentration_read[radial_plus_index];
            const {cuda_data_type} concentration_radial_minus = species_concentration_read[radial_minus_index];
            const {cuda_data_type} concentration_axial_plus = species_concentration_read[axial_plus_index];
            const {cuda_data_type} concentration_axial_minus = species_concentration_read[axial_minus_index];

            const {cuda_data_type} radial_laplacian =
                (concentration_radial_plus - two_value * concentration_center + concentration_radial_minus)
                / (radial_cell_size * radial_cell_size)
                + (concentration_radial_plus - concentration_radial_minus)
                * (half_value / radial_cell_size)
                * inverse_radial_position;

            const {cuda_data_type} axial_laplacian =
                (concentration_axial_plus - two_value * concentration_center + concentration_axial_minus)
                / (axial_cell_size * axial_cell_size);

            const {cuda_data_type} diffusion_increment =
                substep_time_step * local_mass_diffusivity * (radial_laplacian + axial_laplacian);

            const {cuda_data_type} left_face_flux =
                (left_face_velocity_value > zero_value)
                    ? left_face_velocity_value * concentration_axial_minus
                    : left_face_velocity_value * concentration_center;

            const {cuda_data_type} right_face_flux =
                (right_face_velocity_value > zero_value)
                    ? right_face_velocity_value * concentration_center
                    : right_face_velocity_value * concentration_axial_plus;

            const {cuda_data_type} advection_increment =
                -substep_time_over_axial_cell_size * (right_face_flux - left_face_flux);

            species_concentration_write[center_index] =
                concentration_center + diffusion_increment + advection_increment;
        }}

        // ------------------------------------------------------------
        // 2) Reaction source terms and heat release
        //    (assumes species order: 0 Ethylene, 1 Oxygen, 2 Ethylene Oxide, 3 Water, 4 Carbon Dioxide)
        // ------------------------------------------------------------
        const int ethylene_species_index = 0;
        const int oxygen_species_index = 1;

        const int ethylene_center_index =
            ethylene_species_index * species_stride_for_row_major
            + radial_cell_index * axial_stride_for_row_major
            + axial_cell_index;

        const int oxygen_center_index =
            oxygen_species_index * species_stride_for_row_major
            + radial_cell_index * axial_stride_for_row_major
            + axial_cell_index;

        {cuda_data_type} ethylene_concentration = species_concentration_write[ethylene_center_index];
        {cuda_data_type} oxygen_concentration   = species_concentration_write[oxygen_center_index];

        const {cuda_data_type} temperature_center_for_reactions =
            temperature_read[temperature_center_index];

        {cuda_data_type} reaction_temperature_increment = zero_value;

        if (ethylene_concentration > tiny_value && oxygen_concentration > tiny_value) {{

            const {cuda_data_type} ethylene_partial_pressure_scaled =
                (ethylene_concentration * ideal_gas_constant * temperature_center_for_reactions) * pressure_scaling_factor;

            const {cuda_data_type} oxygen_partial_pressure_scaled =
                (oxygen_concentration * ideal_gas_constant * temperature_center_for_reactions) * pressure_scaling_factor;

            const {cuda_data_type} main_reaction_rate_constant =
                pre_exponential_factor_main_reaction
                * exp(-activation_energy_main_reaction / (ideal_gas_constant * temperature_center_for_reactions));

            const {cuda_data_type} side_reaction_rate_constant =
                pre_exponential_factor_side_reaction
                * exp(-activation_energy_side_reaction / (ideal_gas_constant * temperature_center_for_reactions));

            {cuda_data_type} main_reaction_denominator =
                ({cuda_data_type})1.0 + adsorption_equilibrium_constant_main_reaction * ethylene_partial_pressure_scaled;
            main_reaction_denominator *= main_reaction_denominator;

            {cuda_data_type} side_reaction_denominator =
                ({cuda_data_type})1.0 + adsorption_equilibrium_constant_side_reaction * ethylene_partial_pressure_scaled;
            side_reaction_denominator *= side_reaction_denominator;

            const {cuda_data_type} main_reaction_rate =
                main_reaction_rate_constant
                * ethylene_partial_pressure_scaled
                * pow(oxygen_partial_pressure_scaled, oxygen_reaction_order_main_reaction)
                * catalyst_mass_per_volume
                / main_reaction_denominator;

            const {cuda_data_type} side_reaction_rate =
                side_reaction_rate_constant
                * ethylene_partial_pressure_scaled
                * pow(oxygen_partial_pressure_scaled, oxygen_reaction_order_side_reaction)
                * catalyst_mass_per_volume
                / side_reaction_denominator;

            // Species concentration updates
            species_concentration_write[ethylene_center_index] +=
                substep_time_step * ( -main_reaction_rate - side_reaction_rate );

            species_concentration_write[oxygen_center_index] +=
                substep_time_step * ( -({cuda_data_type})0.5 * main_reaction_rate - ({cuda_data_type})3.0 * side_reaction_rate );

            species_concentration_write[2 * species_stride_for_row_major + radial_cell_index * axial_stride_for_row_major + axial_cell_index] +=
                substep_time_step * ( main_reaction_rate );

            species_concentration_write[3 * species_stride_for_row_major + radial_cell_index * axial_stride_for_row_major + axial_cell_index] +=
                substep_time_step * ( ({cuda_data_type})2.0 * side_reaction_rate );

            species_concentration_write[4 * species_stride_for_row_major + radial_cell_index * axial_stride_for_row_major + axial_cell_index] +=
                substep_time_step * ( ({cuda_data_type})2.0 * side_reaction_rate );

            const {cuda_data_type} volumetric_heat_release =
                - (reaction_enthalpy_main_reaction * main_reaction_rate + reaction_enthalpy_side_reaction * side_reaction_rate);

            reaction_temperature_increment =
                substep_time_step * (volumetric_heat_release / volumetric_heat_capacity);
        }}

        // ------------------------------------------------------------
        // 3) Temperature update (conduction + advection + sink + reaction)
        // ------------------------------------------------------------
        const int temperature_radial_plus_index =
            (radial_cell_index + 1) * axial_stride_for_row_major + axial_cell_index;

        const int temperature_radial_minus_index =
            (radial_cell_index - 1) * axial_stride_for_row_major + axial_cell_index;

        const int temperature_axial_plus_index =
            radial_cell_index * axial_stride_for_row_major + (axial_cell_index + 1);

        const int temperature_axial_minus_index =
            radial_cell_index * axial_stride_for_row_major + (axial_cell_index - 1);

        const {cuda_data_type} temperature_center = temperature_read[temperature_center_index];
        const {cuda_data_type} temperature_radial_plus = temperature_read[temperature_radial_plus_index];
        const {cuda_data_type} temperature_radial_minus = temperature_read[temperature_radial_minus_index];
        const {cuda_data_type} temperature_axial_plus = temperature_read[temperature_axial_plus_index];
        const {cuda_data_type} temperature_axial_minus = temperature_read[temperature_axial_minus_index];

        const {cuda_data_type} radial_conduction_increment =
            (temperature_radial_plus - two_value * temperature_center + temperature_radial_minus)
            * thermal_diffusivity_time_over_radial_cell_size_squared
            + (temperature_radial_plus - temperature_radial_minus)
            * half_thermal_diffusivity_time_over_radial_cell_size
            * inverse_radial_position;

        const {cuda_data_type} axial_conduction_increment =
            (temperature_axial_plus - two_value * temperature_center + temperature_axial_minus)
            * thermal_diffusivity_time_over_axial_cell_size_squared;

        const {cuda_data_type} left_face_temperature_flux =
            (left_face_velocity_value > zero_value)
                ? left_face_velocity_value * temperature_axial_minus
                : left_face_velocity_value * temperature_center;

        const {cuda_data_type} right_face_temperature_flux =
            (right_face_velocity_value > zero_value)
                ? right_face_velocity_value * temperature_center
                : right_face_velocity_value * temperature_axial_plus;

        const {cuda_data_type} temperature_advection_increment =
            -substep_time_over_axial_cell_size * (right_face_temperature_flux - left_face_temperature_flux);

        const {cuda_data_type} temperature_intermediate =
            temperature_center
            + radial_conduction_increment
            + axial_conduction_increment
            + temperature_advection_increment
            + reaction_temperature_increment;

        temperature_write[temperature_center_index] =
            (temperature_intermediate + heat_transfer_sink_coefficient_time * coolant_temperature)
            / (({cuda_data_type})1.0 + heat_transfer_sink_coefficient_time);

        // ------------------------------------------------------------
        // 4) Clamp species concentrations to non-negative values
        // ------------------------------------------------------------
        for (int species_index = 0; species_index < number_of_species; ++species_index) {{
            const int center_index =
                species_index * species_stride_for_row_major
                + radial_cell_index * axial_stride_for_row_major
                + axial_cell_index;

            const {cuda_data_type} concentration_value = species_concentration_write[center_index];
            if (concentration_value < zero_value) species_concentration_write[center_index] = zero_value;
        }}

        // ------------------------------------------------------------
        // Ping-pong buffers
        // ------------------------------------------------------------
        const {cuda_data_type}* temporary_species_pointer = species_concentration_read;
        species_concentration_read = species_concentration_write;
        species_concentration_write = ({cuda_data_type}*)temporary_species_pointer;

        const {cuda_data_type}* temporary_temperature_pointer = temperature_read;
        temperature_read = temperature_write;
        temperature_write = ({cuda_data_type}*)temporary_temperature_pointer;
    }}
}}
'''

macro_step = cp.RawKernel(macro_step_source_code, "macro_step", options=("--use_fast_math",))

# =========================
# Macro-step loop setup
# =========================
SUB_STEPS   = 1        
CHUNK_STEPS = 1          
assert SUB_STEPS % CHUNK_STEPS == 0

number_of_macro_steps = number_of_time_steps // SUB_STEPS
remaining_time_steps     = number_of_time_steps %  SUB_STEPS
if remaining_time_steps:
    raise ValueError("number_of_steps must be divisible by SUB_STEPS")

substep_time_step = data_type(time_step_size / SUB_STEPS)

thermal_diffusivity_time_over_radial_cell_size_squared = data_type(
    Gas.thermal_diffusivity * substep_time_step
    / (radial_cell_size*radial_cell_size)
)

thermal_diffusivity_time_over_axial_cell_size_squared = data_type(
    Gas.thermal_diffusivity * substep_time_step
    / (axial_cell_size*axial_cell_size)
)

half_thermal_diffusivity_time_over_radial_cell_size = (
    data_type(0.5)
    * data_type(Gas.thermal_diffusivity * substep_time_step / radial_cell_size)
)

substep_time_over_axial_cell_size = data_type(substep_time_step / axial_cell_size)

heat_transfer_sink_coefficient_time = data_type(
    (Reactor.heat_transfer_coefficient / volumetric_heat_capacity)
    * substep_time_step
)

# ping-pong buffers
species_concentration_buffer_field = cp.empty_like(species_concentration_field)

temperature_buffer_field = cp.empty_like(temperature_field)

# face arrays & 1/r flatten
face_velocity_profile = cp.empty(
    (number_axial_cells - 1),
    dtype=data_type
)

left_face_velocity_profile = cp.empty(
    (number_axial_cells - 2),
    dtype=data_type
)

right_face_velocity_profile = cp.empty(
    (number_axial_cells - 2),
    dtype=data_type
)

inverse_interior_radial_cell_center_position_line_contiguous = cp.ascontiguousarray(
    inverse_interior_radial_cell_center_position_line.ravel()
)

# initialize inlet BC
species_concentration_field[:, :, 0] = Species.inlet_concentrations[:, None]

# initial Ergun
axial_average_temperature_profile = cp.mean(
    temperature_field,
    axis=0,
    dtype=data_type
)

axial_dynamic_viscosity_profile = (
    data_type(Gas.dynamic_viscosity)
    * (axial_average_temperature_profile / data_type(Gas.inlet_temperature))
    **data_type(0.7)
)

pressure_profile, superficial_velocity_profile = ergun_pressure_velocity_profile(
    species_concentration_field,
    Gas.inlet_pressure,
    Reactor.void_fraction,
    Catalyst.particle_diameter,
    axial_dynamic_viscosity_profile,
    axial_average_temperature_profile,
    Gas.ideal_gas_constant,
    Species.molecular_weight,
    Species.inlet_concentrations,
    axial_cell_size,
    Gas.inlet_superficial_velocity,
    iterations=1,
    minimum_pressure=data_type(0.5e5)
)

# Kernel launch shape (interior region only)
threads_per_block = (32, 4, 1)
blocks_per_grid = (
    (number_axial_cells-2 + threads_per_block[0] - 1) // threads_per_block[0],
    (number_radial_cells-2 + threads_per_block[1] - 1) // threads_per_block[1],
    1
)

heat_transfer_sink_coefficient = Reactor.heat_transfer_coefficient / volumetric_heat_capacity

# -------------------------
# Macro-step main loop
# -------------------------
simulation_start_time = pytime.time()
LOG_EVERY = max(1, n_macro // 1)
RECOMP_TOL  = 1e-4  # recompute Ergun when Tz changes

previous_axial_average_temperature_profile = axial_average_temperature_profile

current_species_concentration_field, next_species_concentration_field = (
    species_concentration_field,
    species_concentration_buffer_field
)

current_temperature_field, next_temperature_field = (
    temperature_field,
    temperature_buffer_field
)

for macro_step_index in range(number_of_macro_steps):

    # Recompute Ergun each macro
    axial_average_temperature_profile = cp.mean(
        current_temperature_field,
        axis=0,
        dtype = data_type
    )

    relative_temperature_change = cp.max(
        cp.abs(
            (axial_average_temperature_profile - previous_axial_average_temperature_profile)
            / (axial_average_temperature_profile + data_type(1e-6))
        )
    )

    if relative_temperature_change > temperature_change_tolerance:
        axial_dynamic_viscosity_profile = (
            data_type(Gas.dynamic_viscosity)
            * (axial_average_temperature_profile / data_type(Gas.inlet_temperature))
            ** data_type(0.7)
        )

        pressure_profile, superficial_velocity_profile = ergun_pressure_velocity_profile(
            current_species_concentration_field,
            Gas.inlet_pressure,
            Reactor.void_fraction,
            Catalyst.particle_diameter,
            axial_dynamic_viscosity_profile,
            axial_average_temperature_profile,
            Gas.ideal_gas_constant,
            Species.molecular_weight,
            Species.inlet_concentrations,
            axial_cell_size,
            Gas.inlet_superficial_velocity,
            iterations=1,
            minimum_pressure=data_type(0.5e5),
        )

        previous_axial_average_temperature_profile = axial_average_temperature_profile

    # build face velocities: from the centre velocity profile
    cp.add(
        superficial_velocity_profile[:-1],
        superficial_velocity_profile[1:],
        out=face_velocity_profile
    )
    face_velocity_profile *= data_type(0.5)

    left_face_velocity_profile[...] = face_velocity_profile[:-1]
    right_face_velocity_profile[...] = face_velocity_profile[ 1:]

    interior_cell_center_pressure_profile = pressure_profile[1:-1]

    remaining_substeps = SUB_STEPS
    while remaining_substeps > 0:
        number_of_substeps_this_call = min(CHUNK_STEPS, remaining)

        macro_step(
            blocks_per_grid,
            threads_per_block,
            (
                np.int32(Species.count),
                np.int32(number_radial_cells),
                np.int32(number_axial_cells),
                np.int32(number_of_substeps_this_call),

                current_species_concentration_field,
                next_species_concentration_field,
                current_temperature_field,
                next_temperature_field,

                inverse_interior_radial_cell_center_position_line_contiguous,
                left_face_velocity_profile,
                right_face_velocity_profile,
                interior_cell_center_pressure_profile,

                radial_cell_size,
                axial_cell_size,
                substep_time_step,

                # substep-scaled transport coefficients
                thermal_diffusivity_time_over_radial_cell_size_squared,
                half_thermal_diffusivity_time_over_radial_cell_size,
                thermal_diffusivity_time_over_axial_cell_size_squared,
                substep_time_over_axial_cell_size,
                heat_transfer_sink_coefficient_time,

                # diffusion & kinetics constants
                Catalyst.weight,
                mass_diffusivity,
                Gas.inlet_temperature,
                Gas.inlet_pressure,
                Gas.ideal_gas_constant,

                Main_Reaction.KE1,
                Main_Reaction.n1,
                Main_Reaction.k01,
                Main_Reaction.Ea1,
                Main_Reaction.dH1,

                Side_Reaction.KE2,
                Side_Reaction.n2,
                Side_Reaction.k02,
                Side_Reaction.Ea2,
                Side_Reaction.dH2,
                # energy source and cooling pieces
                volumetric_heat_capacity,
                coolant_temperature
            )
        )

        # swap for next substep
        current_species_concentration_field, next_species_concentration_field = (
            next_species_concentration_field,
            current_species_concentration_field
        )

        current_temperature_field, next_temperature_field = (
            next_temperature_field,
            current_temperature_field
        )

        # Apply BC at each chunk
        current_species_concentration_field[:, 0, :]  = current_species_concentration_field[:, 1, :]
        current_species_concentration_field[:, -1, :] = current_species_concentration_field[:, -2, :]
        current_species_concentration_field[:, :,  0] = Species.initial_concentration[:, None]
        current_species_concentration_field[:, :, -1] = current_species_concentration_field[:, :, -2]

        current_temperature_field[0, :]  = current_temperature_field[1, :]
        current_temperature_field[-1, :] = current_temperature_field[-2, :]
        current_temperature_field[:,  0] = data_type(Gas.inlet_temperature)
        current_temperature_field[:, -1] = current_temperature_field[:, -2]

        remaining_substeps -= number_of_substeps_this_call

    # Logging per macro
    if (macro_step_index % LOG_EVERY) == 0:
        outlet_pressure_value = float(pressure_profile[-1])
        
        outlet_average_temperature_value = float(
            cp.mean(current_temperature_field[:, -1])
        )
        
        elapsed_simulation_time = pytime.time() - current_temperature_field
        
        completed_fraction = (macro_step_index + 1) / number_of_macro_steps
        
        estimated_remaining_time = (
            elapsed_simulation_time * (1 - completed_fraction)
            / max(completed_fraction, 1e-9)
        )
        
        print(
            f"Macro {macro_step_index + 1}/{number_of_macro_steps} "
            f"(sub={number_of_substeps_this_call}) "
            f"outlet_pressure={outlet_pressure_value / 1e5:.3f} "
            f"outlet_temperature~{outlet_average_temperature_value:.1f} "
            f"elapsed={elapsed_simulation_time:.1f}s "
            f"ETA={estimated_remaining_time:.1f}s"
        )

species_concentration_field = current_species_concentration_field
temperature_field = current_temperature_field

total_elapsed_time = pytime.time() - simulation_start_time
print(f"Simulation done in {total_elapsed_time:.1f}s")

# =========================
# CPU transfer and plots
# =========================
species_concentration_field_cpu = cp.asnumpy(species_concentration_field)
radial_cell_center_positions_cpu = cp.asnumpy(radial_cell_center_position)
axial_cell_center_positions_cpu = cp.asnumpy(axial_cell_center_position)

superficial_velocity_profile_cpu = cp.asnumpy(superficial_velocity_profile)
pressure_profile_cpu = cp.asnumpy(pressure_profile)
temperature_field_cpu = cp.asnumpy(temperature_field)

axial_grid_coordinates, radial_grid_coordinates = np.meshgrid(
    axial_cell_center_positions_cpu,
    radial_cell_center_positions_cpu,
)

# Instantaneous rates at final state for selectivity plots
(
    ethylene_source_term,
    oxygen_source_term,
    ethylene_oxide_source_term,
    water_source_term,
    carbon_dioxide_source_term,
    main_reaction_rate,
    side_reaction_rate,
) = reaction_terms_T(
    species_concentration_field[0],
    species_concentration_field[1],
    temperature_field,
    Main_Reaction.KE1,
    Side_Reaction.KE2,
    Main_Reaction.n1,
    Side_Reaction.n2,
    Main_Reaction.k01,
    Side_Reaction.k02,
    Main_Reaction.Ea1,
    Side_Reaction.Ea2,
    Gas.ideal_gas_constant,
    Catalyst.weight,
)

main_reaction_rate_cpu = cp.asnumpy(main_reaction_rate)
side_reaction_rate_cpu = cp.asnumpy(side_reaction_rate)

reaction_rate_sum = main_reaction_rate_cpu + side_reaction_rate_cpu
reaction_rate_sum_with_nan_for_zero = np.where(reaction_rate_sum > 1e-20, reaction_rate_sum, np.nan)

main_path_selectivity_fraction = main_reaction_rate_cpu / reaction_rate_sum_with_nan_for_zero
side_path_selectivity_fraction = side_reaction_rate_cpu / reaction_rate_sum_with_nan_for_zero

# Species concentration fields
figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.ravel()

for species_index, species_name in enumerate(Species.names):
    image_handle = axes[species_index].pcolormesh(
        axial_grid_coordinates,
        radial_grid_coordinates,
        species_concentration_field_cpu[species_index],
        shading="auto",
        cmap="plasma",
    )
    axes[species_index].set_title(f"Species: {species_name}")
    axes[species_index].set_xlabel("Axial position")
    axes[species_index].set_ylabel("Radial position")
    colorbar_handle = plt.colorbar(image_handle, ax=axes[species_index])
    colorbar_handle.set_label("Concentration")

plt.suptitle("Final concentration fields")
plt.show()

# Temperature field
plt.figure(figsize=(6, 4))
temperature_image_handle = plt.pcolormesh(
    axial_grid_coordinates,
    radial_grid_coordinates,
    temperature_field_cpu,
    shading="auto",
    cmap="inferno",
)
plt.colorbar(temperature_image_handle, label="Temperature")
plt.xlabel("Axial position")
plt.ylabel("Radial position")
plt.title("Final temperature field")
plt.show()

# Velocity profile
plt.figure(figsize=(10, 4))
plt.plot(axial_cell_center_positions_cpu, superficial_velocity_profile_cpu)
plt.grid(True, alpha=0.3)
plt.xlabel("Axial position")
plt.ylabel("Superficial velocity")
plt.title("Axial superficial velocity profile")
plt.show()

# Pressure profile
plt.figure(figsize=(10, 4))
plt.plot(axial_cell_center_positions_cpu, pressure_profile_cpu)
plt.grid(True, alpha=0.3)
plt.xlabel("Axial position")
plt.ylabel("Pressure")
plt.title("Pressure profile")
plt.show()

# Selectivity heatmap (main reaction path)
plt.figure(figsize=(12, 5))
selectivity_image_handle = plt.pcolormesh(
    axial_grid_coordinates,
    radial_grid_coordinates,
    main_path_selectivity_fraction,
    shading="auto",
    cmap="viridis",
)
plt.colorbar(selectivity_image_handle, label="Main path selectivity fraction")
plt.xlabel("Axial position")
plt.ylabel("Radial position")
plt.title("Instantaneous selectivity fraction to ethylene oxide")
plt.tight_layout()
plt.show()