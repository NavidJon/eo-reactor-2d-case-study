# config.py

import cupy as cp

# -----------------------------
# Mesh / time
# -----------------------------
mesh_scale = 100
number_radial_cells = 1 * mesh_scale
number_axial_cells = 10 * mesh_scale

time_final = 20.0
number_of_time_steps = 40000

# Kernel launch defaults
threads_per_block = (32, 4, 1)

log_every = 20000

# -----------------------------
# Steady-state detection
# -----------------------------
steady_state_enabled = True
steady_state_tolerance = 1e-3    # Relative change threshold
steady_state_check_every = 500   # Check every N steps
steady_state_min_steps = 10000   # Minimum steps before checking