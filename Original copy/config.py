import cupy as cp

# -----------------------------
# Mesh / time
# -----------------------------
mesh_scale = 40
number_radial_cells = 1 * mesh_scale
number_axial_cells = 10 * mesh_scale

time_final = 20.0
number_of_time_steps = 20000

# Kernel launch defaults
threads_per_block = (32, 4, 1)