import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from models import Species
from physics import reaction_terms_T
from models import Main_Reaction, Side_Reaction, Gas, Catalyst

def plot_results(
    species_concentration_field,
    radial_cell_center_position,
    axial_cell_center_position,
    superficial_velocity_profile,
    pressure_profile,
    temperature_field,
):
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