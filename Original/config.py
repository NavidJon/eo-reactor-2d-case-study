import cupy as cp

# -----------------------------
# GPU / precision
# -----------------------------
DATA_TYPE = cp.float32  # or cp.float64

cp.cuda.runtime.setDevice(0)

if DATA_TYPE == cp.float32:
    CUDA_DATA_TYPE = "float"
elif DATA_TYPE == cp.float64:
    CUDA_DATA_TYPE = "double"
else:
    raise ValueError("DATA_TYPE must be cp.float32 or cp.float64")

data_type = DATA_TYPE
cuda_data_type = CUDA_DATA_TYPE

# -----------------------------
# Mesh / time
# -----------------------------
mesh_scale = 40
number_radial_cells = 1 * mesh_scale
number_axial_cells = 10 * mesh_scale

time_final = data_type(20.0)
number_of_time_steps = 20000

# Substepping (macro-step fusion)
SUB_STEPS = 1
CHUNK_STEPS = 1

# -----------------------------
# Solver options
# -----------------------------
RECOMP_TOL = data_type(1e-4)  # recompute Ergun when axial-average T changes

LOG_EVERY_FRAC = data_type(1.0)

# Kernel launch defaults
threads_per_block = (32, 4, 1)