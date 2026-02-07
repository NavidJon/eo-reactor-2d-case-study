# data_generation.py

import itertools
import numpy as np
import pandas as pd
import cupy as cp
import os
import time as pytime
from datetime import datetime

from simulation import run_simulation
from models import Catalyst, Gas, Main_Reaction, Side_Reaction, Reactor, Species
from config import (
    number_axial_cells,
    number_radial_cells,
    log_every,
)


# =============================================================================
# GRIDS
# =============================================================================

def get_test_grid():
    """Quick test: 27 runs"""
    return {
        "T_inlet": [430, 450, 470],
        "T_coolant": [290, 300, 310],
        "v_inlet": [0.4, 0.5, 0.6],
    }
    # 3 × 3 × 3 = 27 runs


def get_full_grid():
    """Complete ML dataset: 26,244 runs"""
    return {
        # Operating conditions
        "T_inlet": [400, 440, 480, 520],           # 4 levels
        "T_coolant": [280, 310, 340],              # 3 levels
        "P_inlet": [5e5, 10e5, 15e5],              # 3 levels
        "v_inlet": [0.3, 0.6, 1.0],                # 3 levels
        
        # Feed composition
        "C_ethylene": [1.0, 2.5, 4.0],             # 3 levels
        "C_oxygen": [3.0, 8.0, 13.0],              # 3 levels
        "C_methane": [5.0, 10.0, 15.0],            # 3 levels
        
        # Catalyst/reactor properties
        "void_fraction": [0.35, 0.40, 0.45],       # 3 levels
        "particle_diameter": [0.02, 0.03, 0.04],   # 3 levels
    }
    # 4 × 3 × 3 × 3 × 3 × 3 × 3 × 3 × 3 = 26,244 runs


# =============================================================================
# PARAMETER MANAGEMENT
# =============================================================================

def store_original():
    """Store all original parameter values for restoration."""
    return {
        # Operating conditions
        "T_inlet": float(Gas.inlet_temperature),
        "T_coolant": float(Reactor.coolant_temperature),
        "P_inlet": float(Gas.inlet_pressure),
        "v_inlet": float(Gas.inlet_superficial_velocity),
        
        # Concentrations
        "inlet_conc": Species.inlet_concentrations.get().copy(),
        "initial_conc": Species.initial_concentration.get().copy(),
        
        # Reactor/catalyst properties
        "void_fraction": float(Reactor.void_fraction),
        "particle_diameter": float(Catalyst.particle_diameter),
        "catalyst_weight": float(Catalyst.weight),
    }


def apply_params(params):
    """Apply parameter dictionary to model classes."""
    
    # Operating conditions
    if "T_inlet" in params:
        Gas.inlet_temperature = float(params["T_inlet"])
    if "T_coolant" in params:
        Reactor.coolant_temperature = float(params["T_coolant"])
    if "P_inlet" in params:
        Gas.inlet_pressure = float(params["P_inlet"])
    if "v_inlet" in params:
        Gas.inlet_superficial_velocity = float(params["v_inlet"])
    
    # Reactor/catalyst properties
    if "void_fraction" in params:
        Reactor.void_fraction = float(params["void_fraction"])
        # IMPORTANT: Update catalyst weight (depends on void fraction)
        Catalyst.weight = Catalyst.density * (1.0 - Reactor.void_fraction)
    
    if "particle_diameter" in params:
        Catalyst.particle_diameter = float(params["particle_diameter"])
    
    # Feed concentrations
    conc = Species.inlet_concentrations.get().copy()
    
    if "C_ethylene" in params:
        conc[0] = float(params["C_ethylene"])
    if "C_oxygen" in params:
        conc[1] = float(params["C_oxygen"])
    if "C_methane" in params:
        conc[5] = float(params["C_methane"])
    
    # Products always start at zero
    conc[2] = 0.0  # Ethylene oxide
    conc[3] = 0.0  # Water
    conc[4] = 0.0  # CO2
    
    Species.inlet_concentrations = cp.asarray(conc)
    Species.initial_concentration = cp.asarray(conc)


def restore_params(original):
    """Restore all original parameter values."""
    
    # Operating conditions
    Gas.inlet_temperature = original["T_inlet"]
    Reactor.coolant_temperature = original["T_coolant"]
    Gas.inlet_pressure = original["P_inlet"]
    Gas.inlet_superficial_velocity = original["v_inlet"]
    
    # Reactor/catalyst properties
    Reactor.void_fraction = original["void_fraction"]
    Catalyst.particle_diameter = original["particle_diameter"]
    Catalyst.weight = original["catalyst_weight"]
    
    # Concentrations
    Species.inlet_concentrations = cp.asarray(original["inlet_conc"])
    Species.initial_concentration = cp.asarray(original["initial_conc"])


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_data(results, run_id):
    """Collect all input parameters and output metrics for one run."""
    
    inlet_conc = Species.inlet_concentrations.get()
    C = results["species_concentration_field"].get()
    T = results["temperature_field"].get()
    P = results["pressure_profile"].get()
    v = results["superficial_velocity_profile"].get()
    r = results["radial_cell_center_position"].get()
    z = results["axial_cell_center_position"].get()
    
    # === INPUT VALUES ===
    T_in = float(Gas.inlet_temperature)
    P_in = float(Gas.inlet_pressure)
    v_in = float(Gas.inlet_superficial_velocity)
    T_cool = float(Reactor.coolant_temperature)
    void_frac = float(Reactor.void_fraction)
    d_particle = float(Catalyst.particle_diameter)
    
    C_eth_in = float(inlet_conc[0])
    C_o2_in = float(inlet_conc[1])
    C_ch4_in = float(inlet_conc[5])
    
    # === OUTPUT VALUES ===
    # Outlet concentrations (radially averaged)
    C_eth_out = float(np.mean(C[0, :, -1]))
    C_o2_out = float(np.mean(C[1, :, -1]))
    C_EO_out = float(np.mean(C[2, :, -1]))
    C_h2o_out = float(np.mean(C[3, :, -1]))
    C_co2_out = float(np.mean(C[4, :, -1]))
    C_ch4_out = float(np.mean(C[5, :, -1]))
    
    # Performance metrics
    eth_consumed = C_eth_in - C_eth_out
    X = eth_consumed / C_eth_in if C_eth_in > 1e-10 else 0
    S = C_EO_out / eth_consumed if eth_consumed > 1e-10 else 0
    Y = X * S
    
    # Temperature metrics
    T_max = float(np.max(T))
    T_mean = float(np.mean(T))
    hotspot_idx = np.unravel_index(np.argmax(T), T.shape)
    
    # Convergence info (if available)
    converged = results.get("converged", None)
    final_step = results.get("final_step", None)
    sim_time = results.get("simulation_time", None)
    
    return {
        # === RUN INFO ===
        "run_id": run_id,
        "converged": converged,
        "final_step": final_step,
        "sim_time_sec": sim_time,
        
        # === INPUTS ===
        # Operating conditions
        "T_inlet": T_in,
        "T_coolant": T_cool,
        "P_inlet": P_in,
        "P_inlet_bar": P_in / 1e5,
        "v_inlet": v_in,
        
        # Feed composition
        "C_ethylene_in": C_eth_in,
        "C_oxygen_in": C_o2_in,
        "C_methane_in": C_ch4_in,
        "C_total_in": C_eth_in + C_o2_in + C_ch4_in,
        "O2_C2H4_ratio": C_o2_in / C_eth_in if C_eth_in > 0 else 0,
        "diluent_fraction": C_ch4_in / (C_eth_in + C_o2_in + C_ch4_in) if (C_eth_in + C_o2_in + C_ch4_in) > 0 else 0,
        
        # Reactor/catalyst properties
        "void_fraction": void_frac,
        "particle_diameter": d_particle,
        
        # === OUTPUTS ===
        # Temperature
        "T_max": T_max,
        "T_mean": T_mean,
        "T_rise": T_max - T_in,
        "T_outlet_mean": float(np.mean(T[:, -1])),
        "T_outlet_center": float(T[0, -1]),
        "T_outlet_wall": float(T[-1, -1]),
        "T_hotspot_r_norm": float(r[hotspot_idx[0]] / r[-1]),
        "T_hotspot_z_norm": float(z[hotspot_idx[1]] / z[-1]),
        
        # Pressure
        "P_outlet": float(P[-1]),
        "P_outlet_bar": float(P[-1] / 1e5),
        "dP": float(P[0] - P[-1]),
        "dP_bar": float((P[0] - P[-1]) / 1e5),
        "dP_percent": float((P[0] - P[-1]) / P[0] * 100) if P[0] > 0 else 0,
        
        # Outlet concentrations
        "C_ethylene_out": C_eth_out,
        "C_oxygen_out": C_o2_out,
        "C_EO_out": C_EO_out,
        "C_water_out": C_h2o_out,
        "C_co2_out": C_co2_out,
        "C_methane_out": C_ch4_out,
        
        # Performance metrics
        "X_ethylene": X,
        "X_oxygen": (C_o2_in - C_o2_out) / C_o2_in if C_o2_in > 1e-10 else 0,
        "S_EO": S,
        "Y_EO": Y,
        
        # Production
        "EO_produced": C_EO_out,
        "CO2_produced": C_co2_out,
        "H2O_produced": C_h2o_out,
    }


def save_arrays(results, run_id, base_dir):
    """Save numpy arrays."""
    dirname = os.path.join(base_dir, f"run_{run_id:05d}")
    os.makedirs(dirname, exist_ok=True)
    
    np.save(f"{dirname}/C.npy", results["species_concentration_field"].get())
    np.save(f"{dirname}/T.npy", results["temperature_field"].get())
    np.save(f"{dirname}/P.npy", results["pressure_profile"].get())
    np.save(f"{dirname}/v.npy", results["superficial_velocity_profile"].get())


def save_profiles(results, run_id, base_dir):
    """Save 1D profiles."""
    C = results["species_concentration_field"].get()
    T = results["temperature_field"].get()
    P = results["pressure_profile"].get()
    z = results["axial_cell_center_position"].get()
    r = results["radial_cell_center_position"].get()
    
    rows = []
    
    # Axial centerline
    for j in range(len(z)):
        rows.append({
            "run_id": run_id, "profile": "axial_center",
            "pos_norm": z[j]/z[-1], "T": T[0,j], "P": P[j],
            "C_eth": C[0,0,j], "C_O2": C[1,0,j], "C_EO": C[2,0,j],
        })
    
    # Radial outlet
    for i in range(len(r)):
        rows.append({
            "run_id": run_id, "profile": "radial_outlet",
            "pos_norm": r[i]/r[-1], "T": T[i,-1], "P": P[-1],
            "C_eth": C[0,i,-1], "C_O2": C[1,i,-1], "C_EO": C[2,i,-1],
        })
    
    filename = os.path.join(base_dir, f"run_{run_id:05d}.csv")
    pd.DataFrame(rows).to_csv(filename, index=False)


# =============================================================================
# MAIN GRID SEARCH
# =============================================================================

def run_grid_search(
    param_grid,
    output_dir="data",
    save_arrays_flag=True,
    save_profiles_flag=True,
    steady_state=True,
    tolerance=1e-6,
    check_every=100,
    min_steps=500,
):
    """
    Run grid search over parameter space.
    
    Args:
        param_grid: Dict of parameter names to value lists
        output_dir: Output directory
        save_arrays_flag: Save .npy files
        save_profiles_flag: Save profile CSVs
        steady_state: Enable early stopping when converged
        tolerance: Convergence tolerance
        check_every: Check convergence every N steps
        min_steps: Minimum steps before checking
    """
    
    # Generate combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    param_list = [dict(zip(keys, combo)) for combo in combinations]
    n_runs = len(param_list)
    
    # Print grid info
    print("\n" + "="*60)
    print("PARAMETER GRID")
    print("="*60)
    for k, v in param_grid.items():
        print(f"  {k:20s}: {len(v)} levels → {v}")
    print("-"*60)
    print(f"  {'TOTAL':20s}: {n_runs} runs")
    
    if steady_state:
        print(f"\n  Steady-state: ENABLED (tol={tolerance}, check every {check_every} steps)")
        est_time = n_runs * 3  # Assume ~3 sec with early stopping
    else:
        est_time = n_runs * 10
    
    print(f"  Estimated time: {est_time/60:.0f} min ({est_time/3600:.1f} hours)")
    print("="*60 + "\n")
    
    # Setup directories
    summary_file = os.path.join(output_dir, "summary.csv")
    arrays_dir = os.path.join(output_dir, "arrays")
    profiles_dir = os.path.join(output_dir, "profiles")
    
    os.makedirs(output_dir, exist_ok=True)
    if save_arrays_flag:
        os.makedirs(arrays_dir, exist_ok=True)
    if save_profiles_flag:
        os.makedirs(profiles_dir, exist_ok=True)
    
    # Store original
    original = store_original()
    
    # Run loop
    successful, failed = [], []
    start_time = pytime.time()
    
    for i in range(n_runs):
        params = param_list[i]
        
        # Progress
        elapsed = pytime.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            eta_sec = avg_time * (n_runs - i)
            eta_str = f"ETA: {eta_sec/60:.0f}m ({eta_sec/3600:.1f}h)"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"{'─'*60}")
        print(f"RUN {i}/{n_runs-1} | {eta_str}")
        print(f"  {params}")
        
        try:
            apply_params(params)
            
            results = run_simulation(
                log_every=log_every,
                steady_state=steady_state,
                tolerance=tolerance,
                check_every=check_every,
                min_steps=min_steps,
            )
            
            # Collect and save
            data = collect_data(results, i)
            
            df = pd.DataFrame([data])
            write_header = not os.path.exists(summary_file)
            df.to_csv(summary_file, mode='a', header=write_header, index=False)
            
            if save_arrays_flag:
                save_arrays(results, i, arrays_dir)
            if save_profiles_flag:
                save_profiles(results, i, profiles_dir)
            
            # Print results
            conv_str = "✓" if data.get("converged") else "○"
            print(f"  {conv_str} T_max={data['T_max']:.1f}K, "
                  f"X={data['X_ethylene']*100:.1f}%, "
                  f"S={data['S_EO']*100:.1f}%, "
                  f"Y={data['Y_EO']*100:.1f}%")
            
            successful.append(i)
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append({"run_id": i, "params": str(params), "error": str(e)})
            
            # Log error to file
            with open(os.path.join(output_dir, "errors.log"), "a") as f:
                f.write(f"{datetime.now()} | Run {i} | {params} | {e}\n")
        
        finally:
            restore_params(original)
    
    # Final summary
    total_time = pytime.time() - start_time
    print(f"\n{'='*60}")
    print("COMPLETE")
    print("="*60)
    print(f"  Successful: {len(successful)}/{n_runs}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    if successful:
        print(f"  Avg time/run: {total_time/len(successful):.1f} sec")
    print(f"  Output: {summary_file}")
    print("="*60)
    
    # Save failed runs
    if failed:
        failed_file = os.path.join(output_dir, "failed_runs.csv")
        pd.DataFrame(failed).to_csv(failed_file, index=False)
        print(f"  Failed runs saved to: {failed_file}")
    
    return successful, failed