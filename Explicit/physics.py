import cupy as cp

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
        (concentration_ethylene * ideal_gas_constant * temperature) / 1e5,
        0.0
    )

    oxygen_partial_pressure = cp.maximum(
        (concentration_oxygen * ideal_gas_constant * temperature) / 1e5,
        0.0
    )

    k1 = k01 * cp.exp(-Ea1 / (ideal_gas_constant * temperature))
    k2 = k02 * cp.exp(-Ea2 / (ideal_gas_constant * temperature))

    main_reaction_denominator = cp.square(1.0 + KE1 * ethylene_partial_pressure)
    side_reaction_denominator = cp.square(1.0 + KE2 * ethylene_partial_pressure)

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

    ethylene_source_term = -(main_reaction_rate + side_reaction_rate)
    oxygen_source_term = -0.5 * main_reaction_rate - 3.0 * side_reaction_rate
    ethylene_oxide_source_term = main_reaction_rate
    water_source_term = 2.0 * side_reaction_rate
    carbon_dioxide_source_term = 2.0 * side_reaction_rate

    return (
        ethylene_source_term,
        oxygen_source_term,
        ethylene_oxide_source_term,
        water_source_term,
        carbon_dioxide_source_term,
        main_reaction_rate,
        side_reaction_rate,
    )


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
        minimum_pressure=0.5e5
):
    number_of_species, number_radial_cells, number_axial_cells = species_concentration_field.shape

    cross_section_average_concentration_profile = cp.mean(
        species_concentration_field,
        axis=1,
    )

    total_concentration_profile = cp.sum(cross_section_average_concentration_profile, axis=0)

    mole_fraction_profile = (
        cross_section_average_concentration_profile
        / total_concentration_profile[None, :]
    )

    mixture_molecular_weight_profile = cp.sum(
        mole_fraction_profile * species_molecular_weight[:, None],
        axis=0,
    )

    inlet_total_concentration = cp.sum(inlet_concentrations)
    inlet_mole_fractions = inlet_concentrations / inlet_total_concentration

    inlet_mixture_molecular_weight = cp.sum(
        inlet_mole_fractions * species_molecular_weight
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

    inlet_mass_flux = inlet_gas_density * inlet_velocity

    viscous_ergun_coefficient = (
        150.0
        * (1.0 - bed_void_fraction) ** 2
        * dynamic_viscosity_profile
        / (bed_void_fraction**3 * catalyst_particle_diameter**2)
    )
    inertial_ergun_coefficient = (
        1.75
        * (1.0 - bed_void_fraction)
        / (bed_void_fraction**3 * catalyst_particle_diameter)
    )

    gas_density_profile = (
        (inlet_pressure * mixture_molecular_weight_profile) 
        / (ideal_gas_constant * temperature_profile_density)
    )

    superficial_velocity_profile = inlet_mass_flux / gas_density_profile

    for _ in range(iterations):
        axial_pressure_gradient_profile = (
            viscous_ergun_coefficient 
            * superficial_velocity_profile
            + inertial_ergun_coefficient 
            * gas_density_profile 
            * cp.square(superficial_velocity_profile)
        )

        cumulative_pressure_drop_profile = cp.empty_like(axial_pressure_gradient_profile)

        if number_axial_cells > 0:
            cumulative_pressure_drop_profile[0] = (
                0.5 
                * axial_pressure_gradient_profile[0] 
                * axial_cell_size
            )

        if number_axial_cells > 1:
            trapezoid_midpoint = (
                0.5
                * (axial_pressure_gradient_profile[1:] + axial_pressure_gradient_profile[:-1])
                * axial_cell_size
            )
            cumulative_pressure_drop_profile[1:] = (
                cumulative_pressure_drop_profile[0]
                + cp.cumsum(trapezoid_midpoint)
            )

        pressure_profile = cp.maximum(
            inlet_pressure - cumulative_pressure_drop_profile,
            minimum_pressure
        )
        
        updated_gas_density_profile = (
            (pressure_profile * mixture_molecular_weight_profile)
            / (ideal_gas_constant * temperature_profile_density)
        )

        relaxation_factor = 0.4
        gas_density_profile = (
            (1.0 - relaxation_factor) * gas_density_profile
            + relaxation_factor * updated_gas_density_profile
        )
        superficial_velocity_profile = inlet_mass_flux / gas_density_profile

    return pressure_profile, superficial_velocity_profile