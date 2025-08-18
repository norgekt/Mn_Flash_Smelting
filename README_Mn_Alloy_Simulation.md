# Conceptual Mn Alloy Process Simulation

A comprehensive Python simulation for a two-stage hydrogen-based flash smelting process for ferromanganese alloy production with reduced CO₂ emissions and enhanced productivity.

## Process Overview

This simulation models a novel two-stage reduction process:

### Stage 1: Pre-reduction with H₂
- **Goal**: Reduce Mn ore (MnO₂, Mn₂O₃) to MnO using H₂ in a flash smelting shift reactor
- **Key Reactions**:
  - `3MnO₂ + H₂ → Mn₃O₄ + H₂O`
  - `Mn₃O₄ + H₂ → 3MnO + H₂O`
  - `3Mn₂O₃ + H₂ → 2Mn₃O₄ + H₂O`
- **Advantages**: Lower CO₂ emissions due to H₂ usage; no coke required

### Stage 2: Slag-Metal Reaction
- **Goal**: React pre-reduced MnO-containing slag with molten alloy bath to form Mn-rich alloy
- **Key Reaction**: `3MnO + 2Al → 3Mn + Al₂O₃`
- **Fluxes**: CaO, MgO addition to optimize slag properties

## Features

- **Kinetic Modeling**: Temperature-dependent reaction kinetics using Arrhenius equations
- **Mass Balance**: Comprehensive mass and elemental balance calculations
- **DNN Integration**: Optional deep neural network for Stage 2 predictions (based on existing `DNN_Model_Al_Red_MnO.py`)
- **Process Optimization**: Parameter sensitivity analysis for temperature and pressure effects
- **Visualization**: Comprehensive plotting of species evolution, reduction degrees, and process metrics
- **Environmental Impact**: Assessment of CO₂ reduction compared to traditional carbon-based processes

## Installation

1. Clone the repository or copy the simulation files
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Simulation

```python
from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters

# Define process parameters
params = ProcessParameters(
    temperature_stage1=1000.0,      # K (727°C)
    temperature_stage2=1873.0,      # K (1600°C)
    h2_partial_pressure=0.8,        # atm
    particle_size=50e-6,            # 50 μm
    al_content=0.15                 # 15% Al in metal bath
)

# Create and run simulator
simulator = ManganesReductionSimulator(params)

# Define initial ore composition
initial_ore = {
    'MnO2': 100.0,   # kg
    'Mn2O3': 30.0,   # kg
    'Mn3O4': 10.0,   # kg
    'MnO': 5.0,      # kg
}

# Run Stage 1
stage1_results = simulator.run_stage1_simulation(
    initial_composition=initial_ore,
    time_range=(0, 7200, 200)  # 2 hours simulation
)

# Run Stage 2
stage2_results = simulator.run_stage2_simulation(
    stage1_output={'MnO': stage1_results['MnO'].iloc[-1]},
    initial_metal_composition={'Al': 20.0, 'Fe': 80.0}
)

# Generate report and plots
print(simulator.generate_report())
simulator.plot_results(save_plots=True)
```

### Running the Example

```bash
python run_mn_alloy_simulation.py
```

This will run a complete simulation with:
- Typical manganese ore composition
- Process optimization study
- Comprehensive results and visualizations

## File Structure

```
├── conceptual_mn_alloy_process_simulation.py    # Main simulation code
├── run_mn_alloy_simulation.py                   # Example usage script
├── requirements.txt                             # Python dependencies
├── README_Mn_Alloy_Simulation.md               # This documentation
├── dnn_model.pkl                                # Optional: Pre-trained DNN model
└── MS_RE/
    └── DNN_Model_Al_Red_MnO.py                 # Reference Stage 2 model
```

## Process Parameters

### Stage 1 Parameters
- `temperature_stage1`: H₂ reduction temperature (K)
- `h2_partial_pressure`: H₂ partial pressure (atm)
- `h2o_partial_pressure`: H₂O partial pressure (atm)
- `particle_size`: Ore particle size (m)

### Stage 2 Parameters
- `temperature_stage2`: Al reduction temperature (K)
- `al_content`: Al fraction in metal bath
- `cao_content`: CaO flux addition fraction
- `mgo_content`: MgO flux addition fraction

### Kinetic Parameters
- `k1_mno2_h2`, `k2_mn3o4_h2`, `k3_mn2o3_h2`: Rate constants
- `ea1`, `ea2`, `ea3`: Activation energies (J/mol)

## Output Files

The simulation generates several output files:

### Results Files
- `stage1_simulation_results.csv`: Detailed Stage 1 time-series data
- `stage2_simulation_results.csv`: Stage 2 mass balance results
- `simulation_report.txt`: Comprehensive process report
- `process_optimization_results.csv`: Parameter optimization study results

### Visualization Files
- `simulation_plots/process_simulation_results.png`: Main process plots
- `process_optimization_study.png`: Optimization study plots

## Key Classes and Methods

### `ProcessParameters`
Dataclass containing all process parameters with default values.

### `ManganesReductionSimulator`
Main simulation class with methods:
- `run_stage1_simulation()`: Execute H₂ reduction kinetics
- `run_stage2_simulation()`: Execute Al reduction (DNN or simplified)
- `calculate_overall_efficiency()`: Compute process metrics
- `plot_results()`: Generate comprehensive visualizations
- `generate_report()`: Create detailed process report

## Scientific Background

### Thermodynamics
The simulation incorporates temperature-dependent equilibrium constants and considers:
- Gibbs free energy changes for each reaction
- Activity coefficients for non-ideal behavior
- Gas-solid reaction kinetics

### Kinetics
Reaction rates follow Arrhenius behavior:
```
k(T) = k₀ × exp(-Ea / RT)
```

Where:
- `k₀`: Pre-exponential factor
- `Ea`: Activation energy
- `R`: Gas constant
- `T`: Temperature

### Mass Balance
Comprehensive elemental mass balance ensures:
- Conservation of Mn, O, H, Al
- Stoichiometric consistency
- Error tracking and reporting

## Environmental Benefits

Compared to traditional carbon-based processes:
- **CO₂ Reduction**: H₂ produces H₂O instead of CO₂
- **Energy Efficiency**: Two-stage optimization reduces energy consumption
- **Resource Utilization**: High metal recovery efficiency
- **Waste Minimization**: Optimized slag composition

## Validation and Calibration

The model can be calibrated using:
- Experimental kinetic data from your existing datasets
- Thermodynamic databases (FactSage, HSC Chemistry)
- Industrial process data
- DNN models trained on experimental results

## Troubleshooting

### Common Issues

1. **DNN Model Not Found**
   - The simulation will automatically fall back to simplified kinetics
   - Ensure `dnn_model.pkl` is in the working directory for advanced Stage 2 modeling

2. **Convergence Issues**
   - Reduce time step size in `time_range` parameter
   - Adjust kinetic parameters for better numerical stability

3. **Unrealistic Results**
   - Check initial composition values
   - Verify temperature and pressure ranges are realistic
   - Ensure mass balance is preserved

### Performance Optimization
- Use fewer time points for faster computation
- Run parameter studies in parallel using multiprocessing
- Cache results for repeated simulations

## Extensions and Modifications

### Adding New Reactions
1. Extend the `stage1_kinetic_model()` method
2. Add new species to the state vector
3. Include corresponding rate constants and activation energies

### Custom DNN Models
1. Train models using your experimental data
2. Save in the compatible pickle format
3. Ensure input/output dimensions match the interface

### Process Optimization
1. Use the optimization framework for parameter studies
2. Implement genetic algorithms or other optimization methods
3. Include economic objective functions

## Contributing

When extending the simulation:
1. Follow existing code structure and documentation standards
2. Include unit tests for new functionality
3. Update this README with new features
4. Validate results against experimental data when possible

## References

1. Thermodynamic data from FactSage databases
2. Kinetic parameters from literature and experimental studies
3. DNN architecture based on materials science applications
4. Process design principles from metallurgical engineering

## Contact

For questions, issues, or contributions related to this simulation, please refer to the project documentation or create an issue in the repository.

---

*This simulation is designed for research and educational purposes. For industrial applications, additional validation and calibration with experimental data is recommended.* 