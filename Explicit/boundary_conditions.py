from models import Gas, Species

def apply_boundary_conditions(species_concentration_field, temperature_field):
    # Radial (Neumann / symmetry)
    species_concentration_field[:, 0, :] = species_concentration_field[:, 1, :]
    species_concentration_field[:, -1, :] = species_concentration_field[:, -2, :]
    temperature_field[0, :] = temperature_field[1, :]
    temperature_field[-1, :] = temperature_field[-2, :]

    # Axial inlet (Dirichlet)
    species_concentration_field[:, :, 0] = Species.inlet_concentrations[:, None]
    temperature_field[:, 0] = Gas.inlet_temperature

    # Axial outlet (Neumann)
    species_concentration_field[:, :, -1] = species_concentration_field[:, :, -2]
    temperature_field[:, -1] = temperature_field[:, -2]
