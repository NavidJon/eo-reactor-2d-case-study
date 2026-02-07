# simulation.py

import time as pytime
import cupy as cp
import numpy as np

from boundary_conditions import apply_boundary_conditions
from config import (
    number_axial_cells,
    number_radial_cells,
    number_of_time_steps,
    time_final,
    log_every as default_log_every,
    steady_state_enabled,
    steady_state_tolerance,
    steady_state_check_every,
    steady_state_min_steps,
)
from kernels import get_macro_step_kernel
from models import Catalyst, Gas, Main_Reaction, Side_Reaction, Reactor, Species
from physics import ergun_pressure_velocity_profile


def run_simulation(
        *,
        log_every=None,
        ergun_iterations=1,
        threads_per_block=(32, 4, 1),
        steady_state=None,
        tolerance=None,
        check_every=None,
        min_steps=None,
        quiet=False,  # NEW: Suppress step logging
):
    """
    Run the 2D packed bed reactor simulation.
    
    Args:
        log_every: Print progress every N steps (default from config)
        ergun_iterations: Iterations for pressure-velocity coupling
        threads_per_block: CUDA thread configuration
        steady_state: Enable early stopping when converged (default from config)
        tolerance: Relative change threshold for convergence (default from config)
        check_every: Check convergence every N steps (default from config)
        min_steps: Minimum steps before checking convergence (default from config)
        quiet: If True, suppress step-by-step logging (for batch runs)
    
    Returns:
        Dictionary containing simulation results and convergence info.
    """
    
    # =========================================================================
    # Use config defaults if not specified
    # =========================================================================
    if log_every is None:
        log_every = default_log_every
    if steady_state is None:
        steady_state = steady_state_enabled
    if tolerance is None:
        tolerance = steady_state_tolerance
    if check_every is None:
        check_every = steady_state_check_every
    if min_steps is None:
        min_steps = steady_state_min_steps

    # =========================================================================
    # Calculate derived quantities
    # =========================================================================
    radial_cell_size = Reactor.radius / number_radial_cells
    axial_cell_size = Reactor.height / number_axial_cells
    time_step_size = time_final / number_of_time_steps

    volumetric_heat_capacity = (
        Reactor.void_fraction
        * Gas.density
        * Gas.specific_heat_capacity
        + Catalyst.weight
        * Catalyst.specific_heat_capacity
    )

    thermal_diffusivity_time_over_radial_cell_size_squared = (
        Gas.thermal_diffusivity
        * time_step_size
        / (radial_cell_size * radial_cell_size)
    )

    thermal_diffusivity_time_over_axial_cell_size_squared = (
        Gas.thermal_diffusivity
        * time_step_size
        / (axial_cell_size * axial_cell_size)
    )

    half_thermal_diffusivity_time_over_radial_cell_size = (
        0.5
        * Gas.thermal_diffusivity
        * time_step_size
        / radial_cell_size
    )

    time_over_axial_cell_size = time_step_size / axial_cell_size

    heat_transfer_sink_coefficient_time = (
        Reactor.heat_transfer_coefficient
        / volumetric_heat_capacity
        * time_step_size
    )

    # =========================================================================
    # Initialize grid positions
    # =========================================================================
    radial_cell_center_position = cp.linspace(
        radial_cell_size / 2.0,
        Reactor.radius - radial_cell_size / 2.0,
        number_radial_cells,
    )

    axial_cell_center_position = cp.linspace(
        axial_cell_size / 2.0,
        Reactor.height - axial_cell_size / 2.0,
        number_axial_cells,
    )

    inverse_interior_radial_cell_center_position = cp.ascontiguousarray(
        1.0 / radial_cell_center_position[1:-1]
    )

    # =========================================================================
    # Initialize fields
    # =========================================================================
    species_concentration_field = cp.zeros(
        (Species.count, number_radial_cells, number_axial_cells)
    )
    species_concentration_field += Species.initial_concentration[:, None, None]

    temperature_field = cp.full(
        (number_radial_cells, number_axial_cells),
        Gas.inlet_temperature
    )

    # Ping-pong buffers
    species_concentration_buffer = species_concentration_field.copy()
    temperature_buffer = temperature_field.copy()

    # Face velocity arrays
    face_velocity_profile = cp.empty(number_axial_cells - 1)
    left_face_velocity = cp.empty(number_axial_cells - 2)
    right_face_velocity = cp.empty(number_axial_cells - 2)

    # Initialize inlet boundary condition
    species_concentration_field[:, :, 0] = Species.inlet_concentrations[:, None]

    # =========================================================================
    # Initial Ergun calculation
    # =========================================================================
    axial_average_temperature = cp.mean(temperature_field, axis=0)

    axial_dynamic_viscosity = (
        Gas.dynamic_viscosity
        * (axial_average_temperature / Gas.inlet_temperature) ** 0.7
    )

    pressure_profile, superficial_velocity_profile = ergun_pressure_velocity_profile(
        species_concentration_field,
        Gas.inlet_pressure,
        Reactor.void_fraction,
        Catalyst.particle_diameter,
        axial_dynamic_viscosity,
        axial_average_temperature,
        Gas.ideal_gas_constant,
        Species.molecular_weight,
        Species.inlet_concentrations,
        axial_cell_size,
        Gas.inlet_superficial_velocity,
        iterations=ergun_iterations,
        minimum_pressure=0.5e5
    )

    # =========================================================================
    # Setup CUDA kernel
    # =========================================================================
    macro_step = get_macro_step_kernel()

    blocks_per_grid = (
        (number_axial_cells - 2 + threads_per_block[0] - 1) // threads_per_block[0],
        (number_radial_cells - 2 + threads_per_block[1] - 1) // threads_per_block[1],
        1
    )

    # =========================================================================
    # Simulation loop
    # =========================================================================
    simulation_start_time = pytime.time()

    current_concentration = species_concentration_field
    next_concentration = species_concentration_buffer

    current_temperature = temperature_field
    next_temperature = temperature_buffer

    # Steady-state detection variables
    converged = False
    final_step = number_of_time_steps
    prev_temperature = None
    prev_concentration = None

    for step_index in range(number_of_time_steps):

        # =====================================================================
        # Update pressure and velocity (Ergun equation)
        # =====================================================================
        axial_average_temperature = cp.mean(current_temperature, axis=0)

        axial_dynamic_viscosity = (
            Gas.dynamic_viscosity
            * (axial_average_temperature / Gas.inlet_temperature) ** 0.7
        )

        pressure_profile, superficial_velocity_profile = ergun_pressure_velocity_profile(
            current_concentration,
            Gas.inlet_pressure,
            Reactor.void_fraction,
            Catalyst.particle_diameter,
            axial_dynamic_viscosity,
            axial_average_temperature,
            Gas.ideal_gas_constant,
            Species.molecular_weight,
            Species.inlet_concentrations,
            axial_cell_size,
            Gas.inlet_superficial_velocity,
            iterations=ergun_iterations,
            minimum_pressure=0.5e5,
        )

        # =====================================================================
        # Build face velocities
        # =====================================================================
        cp.add(
            superficial_velocity_profile[:-1],
            superficial_velocity_profile[1:],
            out=face_velocity_profile
        )
        face_velocity_profile *= 0.5

        left_face_velocity[...] = face_velocity_profile[:-1]
        right_face_velocity[...] = face_velocity_profile[1:]

        interior_pressure = pressure_profile[1:-1]

        # =====================================================================
        # Execute main computation kernel
        # =====================================================================
        macro_step(
            blocks_per_grid,
            threads_per_block,
            (
                np.int32(Species.count),
                np.int32(number_radial_cells),
                np.int32(number_axial_cells),

                current_concentration,
                next_concentration,
                current_temperature,
                next_temperature,

                inverse_interior_radial_cell_center_position,
                left_face_velocity,
                right_face_velocity,
                interior_pressure,

                radial_cell_size,
                axial_cell_size,
                time_step_size,

                thermal_diffusivity_time_over_radial_cell_size_squared,
                half_thermal_diffusivity_time_over_radial_cell_size,
                thermal_diffusivity_time_over_axial_cell_size_squared,
                time_over_axial_cell_size,
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

        # =====================================================================
        # Swap buffers
        # =====================================================================
        current_concentration, next_concentration = next_concentration, current_concentration
        current_temperature, next_temperature = next_temperature, current_temperature

        # =====================================================================
        # Apply boundary conditions
        # =====================================================================
        apply_boundary_conditions(current_concentration, current_temperature)

        # =====================================================================
        # Steady-state check
        # =====================================================================
        if steady_state and step_index >= min_steps and step_index % check_every == 0:
            
            if prev_temperature is not None and prev_concentration is not None:
                # Calculate relative change in temperature
                T_diff = cp.abs(current_temperature - prev_temperature)
                T_scale = cp.abs(prev_temperature) + 1e-10
                T_rel_change = float(cp.max(T_diff / T_scale))
                
                # Calculate relative change in concentration
                C_diff = cp.abs(current_concentration - prev_concentration)
                C_scale = cp.abs(prev_concentration) + 1e-10
                C_rel_change = float(cp.max(C_diff / C_scale))
                
                max_change = max(T_rel_change, C_rel_change)
                
                if max_change < tolerance:
                    converged = True
                    final_step = step_index
                    break
            
            # Store current state for next comparison
            prev_temperature = current_temperature.copy()
            prev_concentration = current_concentration.copy()

        # =====================================================================
        # Progress logging (only if not quiet)
        # =====================================================================
        if not quiet and (step_index % log_every) == 0:
            outlet_pressure = float(pressure_profile[-1])
            outlet_temperature = float(cp.mean(current_temperature[:, -1]))

            elapsed = pytime.time() - simulation_start_time
            progress = (step_index + 1) / number_of_time_steps
            eta = elapsed * (1 - progress) / max(progress, 1e-9)

            print(
                f"Step {step_index + 1}/{number_of_time_steps} "
                f"P_out={outlet_pressure / 1e5:.3f} bar "
                f"T_out={outlet_temperature:.1f} K "
                f"[{elapsed:.1f}s, ETA {eta:.1f}s]"
            )

    # =========================================================================
    # Simulation complete
    # =========================================================================
    total_time = pytime.time() - simulation_start_time

    return {
        # Field data
        "species_concentration_field": current_concentration,
        "temperature_field": current_temperature,
        "pressure_profile": pressure_profile,
        "superficial_velocity_profile": superficial_velocity_profile,
        
        # Grid positions
        "radial_cell_center_position": radial_cell_center_position,
        "axial_cell_center_position": axial_cell_center_position,
        
        # Grid sizes
        "axial_cell_size": axial_cell_size,
        "radial_cell_size": radial_cell_size,
        "time_step_size": time_step_size,
        
        # Physical properties
        "volumetric_heat_capacity": volumetric_heat_capacity,
        
        # Convergence info
        "converged": converged,
        "final_step": final_step,
        "simulation_time": total_time,
    }