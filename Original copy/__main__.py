import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time as pytime

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
        elapsed_time = cp.cuda.get_elapsed_time(start_event, stop_event)  # milliseconds
        accumulated_time[section_name] = accumulated_time.get(section_name, 0.0) + elapsed_time

class NoOperationTimer:
    def start_timing(self, *a, **k):
        pass
    def stop_timing(self, *a, **k):
        pass

timer = GPUTimer() if PROFILE else NoOperationTimer()
elapsed_time_accumulator = {}

from plots import plot_results
from Original.simulation import run_simulation

def main():
    results = run_simulation()
    plot_results(results)

if __name__ == "__main__":
    main()