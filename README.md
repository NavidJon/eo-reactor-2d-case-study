# EO Reactor 2D Case Study

This repository contains the implementation of a two-dimensional multiphysics model of a single tubular ethylene oxide (EO) reactor. The code is used as a case study to generate high-fidelity simulation data for subsequent analysis and surrogate modelling in the associated dissertation.

The reactor is modelled as a packed-bed tubular reactor with coupled mass and energy transport and heterogeneous reaction kinetics for EO formation and total oxidation.

## Model overview

- Reactor type: Single tubular, packed-bed EO reactor
- Dimensions: Cylindrical tube of order 1 cm radius and 10 m length
- Dimensions resolved: 2D axisymmetric grid in radial and axial coordinates \((r,z)\)
- Species:
  - Ethylene (C₂H₄)  
  - Oxygen (O₂)  
  - Ethylene oxide (C₂H₄O)  
  - Water (H₂O)  
  - Carbon dioxide (CO₂)  
  - Methane (CH₄, inert)
- Reactions:
  - Selective epoxidation: C₂H₄ + ½ O₂ → C₂H₄O  
  - Total oxidation: C₂H₄ + 3 O₂ → 2 CO₂ + 2 H₂O  
  with Langmuir–Hinshelwood type kinetics for a Cs-promoted Ag/α-Al₂O₃ catalyst.
- Transport:
  - Axial convection with a velocity profile obtained from the Ergun equation
  - Radial and axial diffusion with a temperature- and pressure-dependent effective diffusivity \(D(T,p)\)
  - Transient convection–diffusion–reaction balances for all species
- Energy balance:
  - Effective gas–solid energy balance with radial/axial conduction
  - Axial convection of sensible heat
  - Heat release from both reactions
  - Volumetric heat removal term representing coolant heat exchange

## Numerical implementation

- Discretisation:
  - Finite-volume–style grid on cell centres in \(r\) and \(z\)
  - Second-order central differences for radial and axial diffusion
  - First-order upwind scheme for axial convection
- Time integration:
  - Explicit time stepping over a fixed time horizon
  - Species and temperature updates fused into a single GPU kernel
  - Ping–pong buffer strategy to avoid race conditions
- Hydrodynamics:
  - 1D Ergun model solved along the tube to obtain pressure \(p(z)\) and superficial velocity \(U(z)\)
  - Velocity and pressure profiles updated when the axial temperature profile changes significantly

The code is implemented in Python using CuPy/CUDA for GPU acceleration and NumPy/Matplotlib for post-processing and visualisation.
