# simulation.py
import time as pytime
import cupy as cp
import numpy as np

from boundary_conditions import apply_boundary_conditions
from config import (
    CHUNK_STEPS,
    data_type,
    RECOMP_TOL,
    SUB_STEPS,
    number_axial_cells,
    number_radial_cells,
    number_of_time_steps,
    time_final,
)
from kernels import get_macro_step_kernel
from models import Catalyst, Gas, Main_Reaction, Side_Reaction, Reactor, Species
from physics import ergun_pressure_velocity_profile


def run_simulation(
        *,
        log_every=None,
        ergun_iterations=1,
        threads_per_block=(32, 4, 1)
):
    assert SUB_STEPS % CHUNK_STEPS == 0

    radial_cell_size = data_type(Reactor.radius / number_radial_cells)
    axial_cell_size = data_type(Reactor.height / number_axial_cells)

    number_of_macro_steps = number_of_time_steps // SUB_STEPS
    remaining_time_steps = number_of_time_steps % SUB_STEPS
    if remaining_time_steps:
        raise ValueError("number_of_steps must be divisible by SUB_STEPS")

    time_step_size = data_type(time_final / number_of_time_steps)
    substep_time_step = data_type(time_step_size / SUB_STEPS)

    volumetric_heat_capacity = data_type(
        Reactor.void_fraction * Gas.density * Gas.specific_heat_capacity
        + Catalyst.weight * Catalyst.specific_heat_capacity
    )

    thermal_diffusivity_time_over_radial_cell_size_squared = data_type(
        Gas.thermal_diffusivity * substep_time_step
        / (radial_cell_size * radial_cell_size)
    )

    thermal_diffusivity_time_over_axial_cell_size_squared = data_type(
        Gas.thermal_diffusivity * substep_time_step
        / (axial_cell_size * axial_cell_size)
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

    radial_cell_center_position = cp.linspace(
        radial_cell_size / data_type(2.0),
        Reactor.radius - radial_cell_size / data_type(2.0),
        number_radial_cells,
        dtype=data_type,
    )

    inverse_interior_radial_cell_center_position_line = data_type(1.0) / radial_cell_center_position[1:-1]
    inverse_interior_radial_cell_center_position_line_contiguous = cp.ascontiguousarray(
        inverse_interior_radial_cell_center_position_line.ravel()
    )

    species_concentration_field = cp.zeros(
        (Species.count, number_radial_cells, number_axial_cells),
        dtype=data_type
    )
    species_concentration_field += Species.initial_concentration[:, None, None]

    temperature_field = cp.full(
        (number_radial_cells, number_axial_cells),
        Gas.inlet_temperature,
        dtype=data_type
    )

    # ping-pong buffers
    species_concentration_buffer_field = cp.empty_like(species_concentration_field)
    temperature_buffer_field = cp.empty_like(temperature_field)

    # face arrays
    face_velocity_profile = cp.empty((number_axial_cells - 1), dtype=data_type)
    left_face_velocity_profile = cp.empty((number_axial_cells - 2), dtype=data_type)
    right_face_velocity_profile = cp.empty((number_axial_cells - 2), dtype=data_type)

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
        ** data_type(0.7)
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
        iterations=ergun_iterations,
        minimum_pressure=data_type(0.5e5)
    )

    macro_step = get_macro_step_kernel()

    # Kernel launch shape (interior region only)
    blocks_per_grid = (
        (number_axial_cells - 2 + threads_per_block[0] - 1) // threads_per_block[0],
        (number_radial_cells - 2 + threads_per_block[1] - 1) // threads_per_block[1],
        1
    )

    # -------------------------
    # Macro-step main loop
    # -------------------------
    simulation_start_time = pytime.time()

    if log_every is None:
        LOG_EVERY = max(1, number_of_macro_steps // 1)
    else:
        LOG_EVERY = int(log_every)

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

        axial_average_temperature_profile = cp.mean(
            current_temperature_field,
            axis=0,
            dtype=data_type
        )

        relative_temperature_change = cp.max(
            cp.abs(
                (axial_average_temperature_profile - previous_axial_average_temperature_profile)
                / (axial_average_temperature_profile + data_type(1e-6))
            )
        )

        if relative_temperature_change > RECOMP_TOL:
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
                iterations=ergun_iterations,
                minimum_pressure=data_type(0.5e5),
            )

            previous_axial_average_temperature_profile = axial_average_temperature_profile

        # build face velocities
        cp.add(
            superficial_velocity_profile[:-1],
            superficial_velocity_profile[1:],
            out=face_velocity_profile
        )
        face_velocity_profile *= data_type(0.5)

        left_face_velocity_profile[...] = face_velocity_profile[:-1]
        right_face_velocity_profile[...] = face_velocity_profile[1:]

        interior_cell_center_pressure_profile = pressure_profile[1:-1]

        remaining_substeps = SUB_STEPS
        while remaining_substeps > 0:
            number_of_substeps_this_call = min(CHUNK_STEPS, remaining_substeps)

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

                    thermal_diffusivity_time_over_radial_cell_size_squared,
                    half_thermal_diffusivity_time_over_radial_cell_size,
                    thermal_diffusivity_time_over_axial_cell_size_squared,
                    substep_time_over_axial_cell_size,
                    heat_transfer_sink_coefficient_time,

                    Gas.mass_diffusivity,
                    Gas.inlet_temperature,
                    Gas.inlet_pressure,

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

                    Main_Reaction.dH1,
                    Side_Reaction.dH2,

                    volumetric_heat_capacity,
                    Reactor.coolant_temperature,
                )
            )

            current_species_concentration_field, next_species_concentration_field = (
                next_species_concentration_field,
                current_species_concentration_field
            )

            current_temperature_field, next_temperature_field = (
                next_temperature_field,
                current_temperature_field
            )

            apply_boundary_conditions(current_species_concentration_field, current_temperature_field)

            remaining_substeps -= number_of_substeps_this_call

        if (macro_step_index % LOG_EVERY) == 0:
            outlet_pressure_value = float(pressure_profile[-1])
            outlet_average_temperature_value = float(cp.mean(current_temperature_field[:, -1]))

            elapsed_simulation_time = pytime.time() - simulation_start_time
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

    total_elapsed_time = pytime.time() - simulation_start_time
    print(f"Simulation done in {total_elapsed_time:.1f}s")

    return {
        "species_concentration_field": current_species_concentration_field,
        "temperature_field": current_temperature_field,
        "pressure_profile": pressure_profile,
        "superficial_velocity_profile": superficial_velocity_profile,
        "radial_cell_center_position": radial_cell_center_position,
        "axial_cell_size": axial_cell_size,
        "radial_cell_size": radial_cell_size,
        "time_step_size": time_step_size,
        "substep_time_step": substep_time_step,
        "volumetric_heat_capacity": volumetric_heat_capacity,
    }