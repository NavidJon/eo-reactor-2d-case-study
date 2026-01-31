import cupy as cp
from config import cuda_data_type

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

def get_macro_step_kernel():
    return macro_step