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
    def __init__(self): self.t = {}
    def tick(self, key):
        if not PROFILE: return
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record(); self.t[key] = [s, e]
    def tock(self, key, acc):
        if not PROFILE: return
        s, e = self.t[key]
        e.record(); e.synchronize()
        ms = cp.cuda.get_elapsed_time(s, e)  # milliseconds
        acc[key] = acc.get(key, 0.0) + ms

class NoOpTimer:
    def tick(self, *a, **k): pass
    def tock(self, *a, **k): pass

timer = GPUTimer() if PROFILE else NoOpTimer()
tim_acc = {}

# =========================
# Precision for GPU calculations
# =========================
DTYPE = cp.float32
cp.cuda.runtime.setDevice(0)
if DTYPE == cp.float32:
    IN_DTYPE = 'float32'   # for Elementwise/RawKernel signatures
    CUDTYPE  = 'float'     # for CUDA device code
elif DTYPE == cp.float64:
    IN_DTYPE = 'float64'
    CUDTYPE  = 'double'
else:
    raise ValueError("DTYPE must be cp.float32 or cp.float64")

# =========================
# Reactor Tube Geometry
# =========================
reactor_scale = 10
radius = 0.001 * reactor_scale
height = 1 * reactor_scale

# =========================
# Grid Mesh
# =========================
mesh_scale = 40
nr, nz = 1*mesh_scale, 10*mesh_scale
dr, dz = DTYPE(radius / nr), DTYPE(height / nz)
r = cp.linspace(dr / 2, radius - dr / 2, nr, dtype=DTYPE)
z = cp.linspace(dz / 2, height - dz / 2, nz, dtype=DTYPE)

# =========================
# Time Stepping
# =========================
time_final = 20
number_of_steps = 20000
dt = DTYPE(time_final / number_of_steps)

# =========================
# Mass transport properties
# =========================
D_REF = DTYPE(1.5e-5)   # [m^2/s]
p_ref = DTYPE(1.0e5)    # [Pa]

# =========================
# Kinetic properties
# =========================
inlet_reference_temperature = DTYPE(450)  # [K]
ideal_gas_constant = DTYPE(8.314)
KE1, KE2 = DTYPE(6.5), DTYPE(4.33)
k01, k02 = DTYPE(1.33e5), DTYPE(1.80e6)
n1,  n2  = DTYPE(0.58),  DTYPE(0.30)
Ea1, Ea2 = DTYPE(60.7e3), DTYPE(73.2e3)

# =========================
# Ergun equation parameters
# =========================
void_fraction = DTYPE(0.40)
catalyst_particle_diameter = DTYPE(3.0e-2)
catalyst_density = DTYPE(1200)
catalyst_weight  = catalyst_density * (1-void_fraction)
mu_ref = DTYPE(3.0e-5)
pressure_inlet = DTYPE(10.0e5)
U_in  = DTYPE(0.5)

# Species molecular weights (kg/mol)
molecular_weight = cp.asarray([0.028, 0.032, 0.044, 0.018, 0.044, 0.016], dtype=DTYPE)

# =========================
# Energy balance parameters
# =========================
effective_gas_density = DTYPE(1.0)
effective_gas_specific_heat_capacity  = DTYPE(1000.0)
effective_gas_thermal_conductivity    = DTYPE(0.1)
alpha = DTYPE(effective_gas_thermal_conductivity /
              (effective_gas_density * effective_gas_specific_heat_capacity))  # [m^2/s]

volumetric_heat_transfer_coefficient = DTYPE(200.0)  # [W/m^3/K]
coolant_temperature = DTYPE(300.0)                   # [K]
dH1 = DTYPE(-105e3); dH2 = DTYPE(-1327e3)

# =========================
# Precompute geometric factors
# =========================
r_mid = r[1:-1][None, :, None]          # (1, nr-2, 1)
inv_r_mid = DTYPE(1.0) / r_mid          # (1, nr-2, 1)
inv_r_mid_T = inv_r_mid[0]              # (nr-2, 1)

# =========================
# Species setup
# =========================
species_names = ['Ethylene','Oxygen','Ethylene Oxide','Water','Carbon Dioxide','Methane']
n_species = len(species_names)
initial_vals = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=DTYPE)
dirichlet_inlet = cp.asarray([2.0, 6.0, 0.0, 0.0, 0.0, 8.0], dtype=DTYPE)

u = cp.zeros((n_species, nr, nz), dtype=DTYPE)
u += initial_vals[:, None, None]

# Temperature field
T_field = cp.full((nr, nz), DTYPE(inlet_reference_temperature), dtype=DTYPE)

# =========================
# Precomputed energy constants
# =========================
cp_solid = DTYPE(900.0)
rho_cp_eff = DTYPE(void_fraction * effective_gas_density * effective_gas_specific_heat_capacity
                   + catalyst_weight * cp_solid)

# =========================
# Fused reaction (for end-of-run selectivity plots)
# =========================
@cp.fuse()
def reaction_terms_T(concentration_ethylene, concentration_oxygen, T, KE1, KE2, n1, n2, k01, k02, Ea1, Ea2, ideal_gas_constant, catalyst_weight):
    pA = cp.maximum(concentration_ethylene * ideal_gas_constant * T, DTYPE(0.0))
    pB = cp.maximum(concentration_oxygen  * ideal_gas_constant * T, DTYPE(0.0))
    pA_bar = pA / DTYPE(1e5); pB_bar = pB / DTYPE(1e5)
    k1 = k01 * cp.exp(-Ea1 / (ideal_gas_constant * T))
    k2 = k02 * cp.exp(-Ea2 / (ideal_gas_constant * T))
    denom1 = cp.maximum(cp.square(DTYPE(1.0) + KE1 * pA_bar), DTYPE(1e-20))
    denom2 = cp.maximum(cp.square(DTYPE(1.0) + KE2 * pA_bar), DTYPE(1e-20))
    r1 = k1 * pA_bar * (pB_bar ** n1) * catalyst_weight / denom1
    r2 = k2 * pA_bar * (pB_bar ** n2) * catalyst_weight / denom2
    sA = -r1 - r2
    sB = -DTYPE(0.5)*r1 - DTYPE(3.0)*r2
    sC = r1
    sD = DTYPE(2.0)*r2
    sE = DTYPE(2.0)*r2
    return sA, sB, sC, sD, sE, r1, r2

# =========================
# Ergun function
# =========================
def ergun_velocity_profile_vectorized(u_field, pressure_inlet, void_fraction, catalyst_particle_diameter,
                                      mu, T_for_rho, ideal_gas_constant, molecular_weight,
                                      dirichlet_inlet, dz, U_in, iters=1, p_floor=DTYPE(0.5e5)):
    nsp, nr_, nz_ = u_field.shape
    # cross-section avg composition vs z
    c_avg = cp.mean(u_field, axis=1, dtype=DTYPE)        # (nsp, nz)
    c_tot = cp.sum(c_avg, axis=0, dtype=DTYPE) + DTYPE(1e-30)
    y     = c_avg / c_tot[None, :]
    MW_mix = cp.sum(y * molecular_weight[:, None], axis=0, dtype=DTYPE)
    # inlet mixture props
    c_in = dirichlet_inlet
    y_in = c_in / (cp.sum(c_in, dtype=DTYPE) + DTYPE(1e-30))
    MW_in = cp.sum(y_in * molecular_weight, dtype=DTYPE)
    T_in = T_for_rho[0] if hasattr(T_for_rho, "ndim") and T_for_rho.ndim == 1 else T_for_rho
    rho_in = (pressure_inlet * MW_in) / (ideal_gas_constant * T_in)
    # mass flux G fixed by inlet superficial velocity
    G = rho_in * U_in
    # Ergun coefficients
    A1 = DTYPE(150.0) * (DTYPE(1.0) - void_fraction)**2 * mu / (void_fraction**3 * catalyst_particle_diameter**2)
    A2 = DTYPE(1.75)  * (DTYPE(1.0) - void_fraction)     / (void_fraction**3 * catalyst_particle_diameter)
    # initial guess
    rho = (pressure_inlet * MW_mix) / (ideal_gas_constant * T_for_rho)
    rho = cp.maximum(rho, DTYPE(1e-6))
    U   = G / rho
    # simple Picard
    for _ in range(iters):
        dpdz = A1 * U + A2 * rho * U * U
        cum = cp.empty_like(dpdz)
        if nz_ > 0: cum[0] = DTYPE(0.5) * dpdz[0] * dz
        if nz_ > 1:
            mid = DTYPE(0.5) * (dpdz[1:] + dpdz[:-1]) * dz
            cum[1:] = cum[0] + cp.cumsum(mid, dtype=DTYPE)
        p_z = cp.maximum(pressure_inlet - cum, p_floor)
        rho_new = (p_z * MW_mix) / (ideal_gas_constant * T_for_rho)
        rho_new = cp.maximum(rho_new, DTYPE(1e-6))
        rho = DTYPE(0.6)*rho + DTYPE(0.4)*rho_new
        U   = G / rho
    return p_z, U

# =========================
# Macro-step fused kernel (ADR + reactions + energy) with SUB_STEPS subcycling
# =========================
macro_step_src = f'''
extern "C" __global__
void macro_step(
    const int nsp, const int nr, const int nz, const int substeps,
    const {CUDTYPE}* __restrict__ u_in,   {CUDTYPE}* __restrict__ u_out,
    const {CUDTYPE}* __restrict__ T_in,   {CUDTYPE}* __restrict__ T_out,
    const {CUDTYPE}* __restrict__ invr,   // (nr-2)
    const {CUDTYPE}* __restrict__ UL,     // (nz-2)
    const {CUDTYPE}* __restrict__ UR,     // (nz-2)
    const {CUDTYPE}* __restrict__ pc,     // (nz-2)
    const {CUDTYPE} dr, const {CUDTYPE} dz, const {CUDTYPE} dt_sub,
    const {CUDTYPE} a_dr2, const {CUDTYPE} half_a_dr, const {CUDTYPE} a_dz2, const {CUDTYPE} dt_over_dz,
    const {CUDTYPE} beta_dt,
    const {CUDTYPE} D_REF, const {CUDTYPE} T_ref, const {CUDTYPE} p_ref,
    const {CUDTYPE} KE1, const {CUDTYPE} KE2, const {CUDTYPE} n1, const {CUDTYPE} n2,
    const {CUDTYPE} k01, const {CUDTYPE} k02, const {CUDTYPE} Ea1, const {CUDTYPE} Ea2,
    const {CUDTYPE} Rgas, const {CUDTYPE} cat_wt,
    const {CUDTYPE} dH1, const {CUDTYPE} dH2,
    const {CUDTYPE} rho_cp_eff, const {CUDTYPE} T_cool
)
{{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // interior j: 1..nz-2
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // interior i: 1..nr-2
    if (i >= nr-1 || j >= nz-1) return;

    const int W = nz;
    const int H = nr;
    const int stride_r = W;
    const int stride_s = H * W;

    const int j_in = j - 1;      // 0..nz-3
    const int i_in = i - 1;      // 0..nr-3

    const {CUDTYPE} invr_i = invr[i_in];
    const {CUDTYPE} UL_j   = UL[j_in];
    const {CUDTYPE} UR_j   = UR[j_in];
    const {CUDTYPE} pcent  = pc[j_in];

    const {CUDTYPE} zero = ({CUDTYPE})0.0;
    const {CUDTYPE} half = ({CUDTYPE})0.5;
    const {CUDTYPE} two  = ({CUDTYPE})2.0;
    const {CUDTYPE} tiny = ({CUDTYPE})1e-30;

    const {CUDTYPE} *u_read = u_in;
          {CUDTYPE} *u_write= u_out;
    const {CUDTYPE} *T_read = T_in;
          {CUDTYPE} *T_write= T_out;

    for (int sstep = 0; sstep < substeps; ++sstep) {{

        // 1) ADR step for all species
        for (int sp = 0; sp < nsp; ++sp) {{
            int base    = sp*stride_s + i*stride_r + j;
            int base_rp = sp*stride_s + (i+1)*stride_r + j;
            int base_rm = sp*stride_s + (i-1)*stride_r + j;
            int base_zp = sp*stride_s + i*stride_r + (j+1);
            int base_zm = sp*stride_s + i*stride_r + (j-1);

            {CUDTYPE} uc  = u_read[base];
            {CUDTYPE} urp = u_read[base_rp];
            {CUDTYPE} urm = u_read[base_rm];
            {CUDTYPE} uzp = u_read[base_zp];
            {CUDTYPE} uzm = u_read[base_zm];

            // Local D(T,p)
            {CUDTYPE} Tc   = T_read[i*stride_r + j];
            {CUDTYPE} Dloc = D_REF * pow(Tc / T_ref, ({CUDTYPE})1.75) * (p_ref / (pcent + tiny));

            {CUDTYPE} lap_r = (urp - two*uc + urm) / (dr*dr)
                            + (urp - urm) * (half/dr) * invr_i;
            {CUDTYPE} lap_z = (uzp - two*uc + uzm) / (dz*dz);
            {CUDTYPE} diff  = dt_sub * Dloc * (lap_r + lap_z);

            // upwind advection in z
            int base_jm = sp*stride_s + i*stride_r + (j-1);
            int base_jp = sp*stride_s + i*stride_r + (j+1);
            {CUDTYPE} cjm = u_read[base_jm];
            {CUDTYPE} cj  = uc;
            {CUDTYPE} cjp = u_read[base_jp];
            {CUDTYPE} FL  = (UL_j > zero) ? UL_j * cjm : UL_j * cj;
            {CUDTYPE} FR  = (UR_j > zero) ? UR_j * cj  : UR_j * cjp;
            {CUDTYPE} adv = - dt_over_dz * (FR - FL);

            u_write[base] = uc + diff + adv;
        }}

        // 2) Reactions on advected fields
        int baseA = 0*stride_s + i*stride_r + j;
        int baseB = 1*stride_s + i*stride_r + j;
        {CUDTYPE} cA = u_write[baseA];
        {CUDTYPE} cB = u_write[baseB];

        // temperature indices
        int Tidx   = i*stride_r + j;
        int Tidx_rp= (i+1)*stride_r + j;
        int Tidx_rm= (i-1)*stride_r + j;
        int Tidx_zp= i*stride_r + (j+1);
        int Tidx_zm= i*stride_r + (j-1);

        {CUDTYPE} Tc = T_read[Tidx];

        {CUDTYPE} q_term = zero;
        if (cA > tiny && cB > tiny) {{
            {CUDTYPE} pA = cA * Rgas * Tc;
            {CUDTYPE} pB = cB * Rgas * Tc;
            {CUDTYPE} pAb= pA * ({CUDTYPE})1e-5;
            {CUDTYPE} pBb= pB * ({CUDTYPE})1e-5;

            {CUDTYPE} k1 = k01 * exp(-Ea1 / (Rgas * Tc));
            {CUDTYPE} k2 = k02 * exp(-Ea2 / (Rgas * Tc));
            {CUDTYPE} d1 = ({CUDTYPE})1.0 + KE1 * pAb; d1 *= d1;
            {CUDTYPE} d2 = ({CUDTYPE})1.0 + KE2 * pAb; d2 *= d2;

            {CUDTYPE} r1 = k1 * pAb * pow(pBb, n1) * cat_wt / d1;
            {CUDTYPE} r2 = k2 * pAb * pow(pBb, n2) * cat_wt / d2;

            u_write[baseA] += dt_sub * ( -r1 - r2 );
            u_write[baseB] += dt_sub * ( -({CUDTYPE})0.5 * r1 - ({CUDTYPE})3.0 * r2 );
            u_write[2*stride_s + i*stride_r + j] += dt_sub * ( r1 );
            u_write[3*stride_s + i*stride_r + j] += dt_sub * ( ({CUDTYPE})2.0 * r2 );
            u_write[4*stride_s + i*stride_r + j] += dt_sub * ( ({CUDTYPE})2.0 * r2 );

            {CUDTYPE} q_rxn = - (dH1 * r1 + dH2 * r2);
            q_term = dt_sub * (q_rxn / rho_cp_eff);
        }}

        // 3) Temperature (conduction + advection + semi-implicit sink + reaction)
        {CUDTYPE} Trp = T_read[Tidx_rp];
        {CUDTYPE} Trm = T_read[Tidx_rm];
        {CUDTYPE} Tzp = T_read[Tidx_zp];
        {CUDTYPE} Tzm = T_read[Tidx_zm];

        {CUDTYPE} radial = (Trp - two*Tc + Trm) * a_dr2
                         + (Trp - Trm) * half_a_dr * invr_i;
        {CUDTYPE} axial  = (Tzp - two*Tc + Tzm) * a_dz2;

        {CUDTYPE} FLT = (UL_j > zero) ? UL_j * T_read[Tidx_zm] : UL_j * Tc;
        {CUDTYPE} FRT = (UR_j > zero) ? UR_j * Tc              : UR_j * T_read[Tidx_zp];
        {CUDTYPE} advT= - dt_over_dz * (FRT - FLT);

        {CUDTYPE} Tstar = Tc + radial + axial + advT + q_term;
        T_write[Tidx] = (Tstar + beta_dt * T_cool) / ( ({CUDTYPE})1.0 + beta_dt );

        // 4) Clamp non-negative species
        for (int sp = 0; sp < nsp; ++sp) {{
            int b = sp*stride_s + i*stride_r + j;
            {CUDTYPE} val = u_write[b];
            if (val < zero) u_write[b] = zero;
        }}

        // ping-pong
        const {CUDTYPE}* tmpU = u_read;  u_read  = u_write;  u_write  = ({CUDTYPE}*)tmpU;
        const {CUDTYPE}* tmpT = T_read;  T_read  = T_write;  T_write  = ({CUDTYPE}*)tmpT;
    }}
}}
'''
macro_step = cp.RawKernel(macro_step_src, 'macro_step', options=('--use_fast_math',))

# =========================
# Macro-step loop setup
# =========================
SUB_STEPS   = 1        
CHUNK_STEPS = 1          
assert SUB_STEPS % CHUNK_STEPS == 0

n_macro = number_of_steps // SUB_STEPS
rem     = number_of_steps %  SUB_STEPS
if rem:
    raise ValueError("number_of_steps must be divisible by SUB_STEPS")

# substep dt and *substep-scaled* coefficients (CRITICAL)
dt_sub                      = DTYPE(dt / SUB_STEPS)
alpha_dt_over_dr2_sub       = DTYPE(alpha * dt_sub / (dr*dr))
alpha_dt_over_dz2_sub       = DTYPE(alpha * dt_sub / (dz*dz))
half_alpha_dt_over_dr_sub   = DTYPE(0.5) * DTYPE(alpha * dt_sub / dr)
dt_over_dz_sub              = DTYPE(dt_sub / dz)
beta_dt_sub                 = DTYPE((volumetric_heat_transfer_coefficient / rho_cp_eff) * dt_sub)

# ping-pong buffers
u_buf  = cp.empty_like(u)
T_buf  = cp.empty_like(T_field)

# face arrays & 1/r flatten
U_faces  = cp.empty((nz-1,), dtype=DTYPE)
U_L_arr  = cp.empty((nz-2,), dtype=DTYPE)
U_R_arr  = cp.empty((nz-2,), dtype=DTYPE)
inv_r_line = cp.ascontiguousarray(inv_r_mid_T.ravel())  # (nr-2,)

# initialize inlet BC
u[:, :, 0] = dirichlet_inlet[:, None]

# initial Ergun
T_z_avg = cp.mean(T_field, axis=0, dtype=DTYPE)
mu_z = DTYPE(mu_ref) * (T_z_avg / DTYPE(inlet_reference_temperature))**DTYPE(0.7)
p_z, U_profile = ergun_velocity_profile_vectorized(
    u, pressure_inlet, void_fraction, catalyst_particle_diameter,
    mu_z, T_z_avg, ideal_gas_constant, molecular_weight, dirichlet_inlet, dz, U_in,
    iters=1, p_floor=DTYPE(0.5e5)
)

# Kernel launch shape (interior region only)
block = (32, 4, 1)
grid  = ((nz-2 + block[0]-1)//block[0],
         (nr-2 + block[1]-1)//block[1], 1)

# handy constants
beta = volumetric_heat_transfer_coefficient / rho_cp_eff

# -------------------------
# Macro-step main loop
# -------------------------
t0 = pytime.time()
LOG_EVERY = max(1, n_macro // 1)
RECOMP_TOL  = 1e-4  # recompute Ergun when Tz changes “enough”
last_Tz = T_z_avg

# current / next (ping-pong) views
u0, u1 = u, u_buf
T0, T1 = T_field, T_buf

for m in range(n_macro):
    # Recompute Ergun each macro (or use your tolerance logic)
    T_z_avg = cp.mean(T0, axis=0, dtype=DTYPE)
    if cp.max(cp.abs((T_z_avg - last_Tz) / (T_z_avg + 1e-6))) > RECOMP_TOL:
        mu_z = DTYPE(mu_ref) * (T_z_avg / DTYPE(inlet_reference_temperature))**DTYPE(0.7)
        p_z, U_profile = ergun_velocity_profile_vectorized(
            u0, pressure_inlet, void_fraction, catalyst_particle_diameter,
            mu_z, T_z_avg, ideal_gas_constant, molecular_weight, dirichlet_inlet, dz, U_in,
            iters=1, p_floor=DTYPE(0.5e5)
        )
        last_Tz = T_z_avg

    # build face velocities: U_faces = 0.5*(U[:-1]+U[1:])
    cp.add(U_profile[:-1], U_profile[1:], out=U_faces)
    U_faces *= DTYPE(0.5)
    U_L_arr[...] = U_faces[:-1]
    U_R_arr[...] = U_faces[ 1:]
    p_center = p_z[1:-1]  # (nz-2,)

    remaining = SUB_STEPS
    while remaining > 0:
        s_this = min(CHUNK_STEPS, remaining)

        # Call macro_step with *substep* coefficients and dt_sub
        macro_step(
            grid, block,
            (
                # sizes / loop count
                np.int32(n_species), np.int32(nr), np.int32(nz), np.int32(s_this),

                # ping-pong fields (read u0/T0, write u1/T1)
                u0, u1, T0, T1,

                # geometry helpers (1/r), faces, center pressure
                inv_r_line, U_L_arr, U_R_arr, p_center,

                # spacing and time (substep)
                dr, dz, dt_sub,

                # *** substep-scaled transport coefficients ***
                alpha_dt_over_dr2_sub, half_alpha_dt_over_dr_sub,
                alpha_dt_over_dz2_sub, dt_over_dz_sub,
                beta_dt_sub,

                # diffusion & kinetics constants
                D_REF, inlet_reference_temperature, p_ref,
                KE1, KE2, n1, n2, k01, k02, Ea1, Ea2,
                ideal_gas_constant, catalyst_weight,

                # energy source and cooling pieces
                dH1, dH2, rho_cp_eff, coolant_temperature
            )
        )

        # swap for next substep
        u0, u1 = u1, u0
        T0, T1 = T1, T0

        # apply BCs each chunk (CHUNK_STEPS=1 ⇒ every substep)
        u0[:, 0, :]  = u0[:, 1, :]
        u0[:, -1, :] = u0[:, -2, :]
        u0[:, :,  0] = dirichlet_inlet[:, None]
        u0[:, :, -1] = u0[:, :, -2]

        T0[0, :]  = T0[1, :]
        T0[-1, :] = T0[-2, :]
        T0[:,  0] = DTYPE(inlet_reference_temperature)
        T0[:, -1] = T0[:, -2]

        remaining -= s_this

    # (optional) logging per macro
    if (m % LOG_EVERY) == 0:
        p_out = float(p_z[-1])
        T_out_mean = float(cp.mean(T0[:, -1]))
        elapsed = pytime.time() - t0
        frac = (m + 1) / n_macro
        eta  = elapsed * (1 - frac) / max(frac, 1e-9)
        print(f"Macro {m+1}/{n_macro} (sub={SUB_STEPS}) p_out={p_out/1e5:.3f} bar  "
              f"T_out~{T_out_mean:.1f} K  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

# make sure the “current” arrays are the ones you plot later
u       = u0
T_field = T0


t_total = pytime.time() - t0
print(f"Simulation done in {t_total:.1f}s")

# =========================
# CPU transfer & plots
# =========================
u_cpu = cp.asnumpy(u)
r_cpu = cp.asnumpy(r)
z_cpu = cp.asnumpy(z)
U_cpu = cp.asnumpy(U_profile)
p_cpu = cp.asnumpy(p_z)
T_cpu = cp.asnumpy(T_field)
Z, R = np.meshgrid(z_cpu, r_cpu)

# Instantaneous rates at final state for selectivity plots
_, _, _, _, _, r1, r2 = reaction_terms_T(u[0], u[1], T_field,
                                         KE1, KE2, n1, n2, k01, k02, Ea1, Ea2,
                                         ideal_gas_constant, catalyst_weight)
r1_cpu = cp.asnumpy(r1)
r2_cpu = cp.asnumpy(r2)
denom = np.where((r1_cpu + r2_cpu) > 1e-20, r1_cpu + r2_cpu, np.nan)
S_path1 = r1_cpu / denom
S_path2 = r2_cpu / denom

# Species fields
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.ravel()
for i, name in enumerate(species_names):
    im = axes[i].pcolormesh(Z, R, u_cpu[i], shading='auto', cmap='plasma')
    axes[i].set_title(f"Species: {name}")
    axes[i].set_xlabel("z [m]"); axes[i].set_ylabel("r [m]")
    cbar = plt.colorbar(im, ax=axes[i]); cbar.set_label("Concentration [mol/m³]")
plt.suptitle("Final concentration fields (macro-step fused GPU)")
plt.show()

# Temperature field
plt.figure(figsize=(6, 4))
im = plt.pcolormesh(Z, R, T_cpu, shading='auto', cmap='inferno')
plt.colorbar(im, label="Temperature [K]")
plt.xlabel("z [m]"); plt.ylabel("r [m]")
plt.title("Final temperature field")
plt.show()

# Velocity & pressure profiles
plt.figure(figsize=(10,4))
plt.plot(z_cpu, U_cpu); plt.grid(True, alpha=0.3)
plt.xlabel("z [m]"); plt.ylabel("Superficial velocity U(z) [m/s]")
plt.title("Axial velocity from Ergun"); plt.show()

plt.figure(figsize=(10,4))
plt.plot(z_cpu, p_cpu/1e5); plt.grid(True, alpha=0.3)
plt.xlabel("z [m]"); plt.ylabel("Pressure [bar]")
plt.title("Pressure profile (Ergun)"); plt.show()

# Selectivity heatmap (Path 1)
plt.figure(figsize=(12,5))
im1 = plt.pcolormesh(Z, R, S_path1, shading='auto', cmap='viridis')
plt.colorbar(im1, label="Path 1 fraction")
plt.xlabel("z [m]"); plt.ylabel("r [m]")
plt.title("Path selectivity to Ethylene Oxide (instantaneous)")
plt.tight_layout(); plt.show()
