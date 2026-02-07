# __main__.py

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time as pytime

from config import (
    steady_state_enabled,
    steady_state_tolerance,
    steady_state_check_every,
    steady_state_min_steps,
    log_every,
)

# -----------------------------
# Toggle time check prints
# -----------------------------
PROFILE = False

class GPUTimer:
    def __init__(self): 
        self.t = {}

    def start_timing(self, section_name):
        if not PROFILE: 
            return
        start_event = cp.cuda.Event()
        stop_event = cp.cuda.Event()
        start_event.record()
        self.t[section_name] = [start_event, stop_event]

    def stop_timing(self, section_name, accumulated_time):
        if not PROFILE: 
            return
        start_event, stop_event = self.t[section_name]
        stop_event.record()
        stop_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, stop_event)
        accumulated_time[section_name] = accumulated_time.get(section_name, 0.0) + elapsed_time


class NoOperationTimer:
    def start_timing(self, *a, **k):
        pass
    def stop_timing(self, *a, **k):
        pass


timer = GPUTimer() if PROFILE else NoOperationTimer()
elapsed_time_accumulator = {}


from plots import plot_results
from simulation import run_simulation


# =============================================================================
# CONFIGURATION
# =============================================================================

MODE = "single"      # Options: "single", "test", "full"
OUTPUT_DIR = "data"
SAVE_ARRAYS = True
SAVE_PROFILES = True

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_single_simulation():
    """Run a single simulation and plot results."""
    
    print("="*60)
    print("SINGLE SIMULATION")
    print("="*60)

    results = run_simulation(
        log_every=log_every,
        steady_state=steady_state_enabled,
        tolerance=steady_state_tolerance,
        check_every=steady_state_check_every,
        min_steps=steady_state_min_steps,
    )
        
    # Print summary
    T = results["temperature_field"].get()
    C = results["species_concentration_field"].get()
    P = results["pressure_profile"].get()
    
    C_eth_in = C[0, :, 0].mean()
    C_eth_out = C[0, :, -1].mean()
    C_EO_out = C[2, :, -1].mean()
    
    X = (C_eth_in - C_eth_out) / C_eth_in if C_eth_in > 0 else 0
    S = C_EO_out / (C_eth_in - C_eth_out) if (C_eth_in - C_eth_out) > 0 else 0
    Y = X * S
    
    print(f"\nRESULTS:")
    print(f"  Converged:   {results.get('converged', 'N/A')}")
    print(f"  Final step:  {results.get('final_step', 'N/A')}")
    print(f"  T_max:       {T.max():.1f} K")
    print(f"  Conversion:  {X*100:.1f}%")
    print(f"  Selectivity: {S*100:.1f}%")
    print(f"  Yield:       {Y*100:.1f}%")
    
    plot_results(
        results["species_concentration_field"],
        results["radial_cell_center_position"],
        results["axial_cell_center_position"],
        results["superficial_velocity_profile"],
        results["pressure_profile"],
        results["temperature_field"],
    )


def run_data_generation(grid_type):
    """Run ML data generation."""
    
    from data_generation import run_grid_search, get_test_grid, get_full_grid
    
    if grid_type == "test":
        grid = get_test_grid()
        print("="*60)
        print("TEST RUN (27 simulations)")
        print("="*60)
    else:
        grid = get_full_grid()
        print("="*60)
        print("FULL RUN (26,244 simulations)")
        print("="*60)
    
    run_grid_search(
        param_grid=grid,
        output_dir=OUTPUT_DIR,
        save_arrays_flag=SAVE_ARRAYS,
        save_profiles_flag=SAVE_PROFILES,
        steady_state=steady_state_enabled,
        tolerance=steady_state_tolerance,
        check_every=steady_state_check_every,
        min_steps=steady_state_min_steps,
    )


def main():
    if MODE == "single":
        run_single_simulation()
    elif MODE == "test":
        run_data_generation("test")
    elif MODE == "full":
        run_data_generation("full")
    else:
        print(f"Unknown MODE: {MODE}")
        print("Options: 'single', 'test', 'full'")


if __name__ == "__main__":
    main()