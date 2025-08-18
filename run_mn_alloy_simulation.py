"""
Example script to run the Conceptual Mn Alloy Process Simulation
"""

from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_example_simulation():
    """Run an example simulation with typical manganese ore composition"""
    
    print("=== EXAMPLE MN ALLOY PROCESS SIMULATION ===\n")
    
    # Define process parameters using user's validated conditions
    params = ProcessParameters(
        temperature_stage1=1173.0,      # K (900°C) - User's validated range 900-1000°C
        temperature_stage2=1873.0,      # K (1600°C) - Al reduction temperature
        h2_partial_pressure=0.8,        # atm
        particle_size=80e-6,            # 80 micrometers (< 0.1 mm as specified)
        residence_time=2.0,             # 2 seconds as specified
        al_content=0.15                 # 15% Al in metal bath
    )
    
    # Create simulator
    simulator = ManganesReductionSimulator(params)
    
    # Load DNN model if available (optional)
    model_loaded = simulator.load_stage2_dnn_model('dnn_model.pkl')
    if not model_loaded:
        print("Using simplified kinetic model for Stage 2\n")
    
    # Define initial manganese ore composition (typical high-grade ore)
    initial_ore_composition = {
        'MnO2': 100.0,   # kg - Primary manganese oxide
        'Mn2O3': 30.0,   # kg - Secondary oxide
        'Mn3O4': 10.0,   # kg - Intermediate oxide
        'MnO': 5.0,      # kg - Target oxide
        'H2': 50.0,      # kg - Excess hydrogen
        'H2O': 0.0       # kg - Initial water content
    }
    
    print("Initial Ore Composition:")
    for species, mass in initial_ore_composition.items():
        print(f"  {species}: {mass:.1f} kg")
    print()
    
    # Run Stage 1: H2 Reduction
    print("Running Stage 1: H2 Reduction of Manganese Oxides...")
    stage1_results = simulator.run_stage1_simulation(
        initial_composition=initial_ore_composition, 
        time_range=(0, 5, 100)  # 5 seconds (covering 2s residence time) with 100 time points
    )
    
    # Show Stage 1 summary
    print(f"\nStage 1 Results Summary:")
    print(f"  Simulation Time: {stage1_results['Time'].iloc[-1]:.1f} seconds")
    
    # Show key results at 2-second residence time (user's specification)
    rd_2s_idx = np.argmin(np.abs(stage1_results['Time'] - 2.0))
    print(f"\n  At 2-second residence time (user's target):")
    print(f"    Reduction Degree: {stage1_results['Reduction_Degree'].iloc[rd_2s_idx]:.3f}")
    print(f"    MnO Produced: {stage1_results['MnO'].iloc[rd_2s_idx]:.1f} kg")
    print(f"    H2 Consumed: {stage1_results['H2'].iloc[0] - stage1_results['H2'].iloc[rd_2s_idx]:.1f} kg")
    print(f"    H2O Generated: {stage1_results['H2O'].iloc[rd_2s_idx]:.1f} kg")
    
    print(f"\n  Final results ({stage1_results['Time'].iloc[-1]:.1f}s):")
    print(f"    Reduction Degree: {stage1_results['Reduction_Degree'].iloc[-1]:.3f}")
    print(f"    MnO Produced: {stage1_results['MnO'].iloc[-1]:.1f} kg")
    
    # Extract final composition for Stage 2
    final_stage1_output = {
        'MnO': stage1_results['MnO'].iloc[-1],
        'Mn3O4': stage1_results['Mn3O4'].iloc[-1],
        'MnO2': stage1_results['MnO2'].iloc[-1],
        'Mn2O3': stage1_results['Mn2O3'].iloc[-1]
    }
    
    # Define metal bath composition
    metal_bath_composition = {
        'Al': 20.0,      # kg - Aluminum for reduction
        'Fe': 80.0,      # kg - Iron base
        'Si': 8.0,       # kg - Silicon content
        'Mn': 5.0,       # kg - Existing manganese
        'C': 2.0         # kg - Carbon content
    }
    
    print(f"\nMetal Bath Composition:")
    for species, mass in metal_bath_composition.items():
        print(f"  {species}: {mass:.1f} kg")
    
    # Run Stage 2: Al Reduction
    print(f"\nRunning Stage 2: Al Reduction of MnO...")
    stage2_results = simulator.run_stage2_simulation(
        stage1_output=final_stage1_output,
        initial_metal_composition=metal_bath_composition
    )
    
    # Show Stage 2 summary
    print(f"\nStage 2 Results Summary:")
    print(f"  Mn Produced: {stage2_results['Mn_produced']:.1f} kg")
    print(f"  Al Consumed: {stage2_results['Al_consumed']:.1f} kg")
    print(f"  Al2O3 Generated: {stage2_results['Al2O3_produced']:.1f} kg")
    print(f"  Process Efficiency: {stage2_results['efficiency']:.1%}")
    
    # Calculate overall metrics
    efficiency_metrics = simulator.calculate_overall_efficiency()
    
    print(f"\n=== OVERALL PROCESS PERFORMANCE ===")
    print(f"Overall Efficiency: {efficiency_metrics['overall_efficiency']:.1%}")
    print(f"Total Mn Production: {efficiency_metrics['mn_production_kg']:.1f} kg")
    print(f"H2 Consumption: {efficiency_metrics['h2_consumption_kg']:.1f} kg")
    print(f"Al Consumption: {efficiency_metrics['al_consumption_kg']:.1f} kg")
    
    # Generate comprehensive report
    report = simulator.generate_report()
    
    # Save report to file
    with open('simulation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nDetailed report saved to 'simulation_report.txt'")
    
    # Generate plots
    simulator.plot_results(save_plots=True)
    
    # Save results to CSV files
    stage1_results.to_csv('stage1_simulation_results.csv', index=False, encoding='utf-8')
    
    stage2_df = pd.DataFrame([stage2_results])
    stage2_df.to_csv('stage2_simulation_results.csv', index=False, encoding='utf-8')
    
    print(f"Results saved to CSV files:")
    print(f"  - stage1_simulation_results.csv")
    print(f"  - stage2_simulation_results.csv")
    
    return simulator, stage1_results, stage2_results

def plot_process_comparison():
    """Create comparison plots for different process conditions"""
    
    print("\n=== PROCESS OPTIMIZATION STUDY ===")
    
    # Test different temperatures for Stage 1
    temperatures = [950, 1000, 1050, 1100]  # K
    h2_pressures = [0.6, 0.8, 1.0]  # atm
    
    results_matrix = []
    
    for temp in temperatures:
        for pressure in h2_pressures:
            params = ProcessParameters(
                temperature_stage1=temp,
                h2_partial_pressure=pressure
            )
            
            simulator = ManganesReductionSimulator(params)
            
            # Simplified ore composition for faster computation
            ore_comp = {
                'MnO2': 50.0, 'Mn2O3': 15.0, 'Mn3O4': 5.0, 'MnO': 2.0
            }
            
            # Run simulation  
            stage1_results = simulator.run_stage1_simulation(
                initial_composition=ore_comp,
                time_range=(0, 5, 50)  # 5 seconds for flash smelting, fewer points
            )
            
            final_reduction = stage1_results['Reduction_Degree'].iloc[-1]
            h2_consumed = stage1_results['H2'].iloc[0] - stage1_results['H2'].iloc[-1]
            
            results_matrix.append({
                'Temperature': temp,
                'H2_Pressure': pressure,
                'Reduction_Degree': final_reduction,
                'H2_Consumption': h2_consumed
            })
    
    # Create optimization plots
    results_df = pd.DataFrame(results_matrix)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot reduction degree vs temperature for different pressures
    for pressure in h2_pressures:
        subset = results_df[results_df['H2_Pressure'] == pressure]
        axes[0].plot(subset['Temperature'], subset['Reduction_Degree'], 
                    'o-', label=f'P_H2 = {pressure} atm')
    
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Final Reduction Degree')
    axes[0].set_title('Effect of Temperature and H2 Pressure on Reduction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot H2 consumption
    for pressure in h2_pressures:
        subset = results_df[results_df['H2_Pressure'] == pressure]
        axes[1].plot(subset['Temperature'], subset['H2_Consumption'], 
                    's-', label=f'P_H2 = {pressure} atm')
    
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('H2 Consumption (kg)')
    axes[1].set_title('H2 Consumption vs Process Conditions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_optimization_study.png', dpi=300)
    plt.show()
    
    # Save optimization results
    results_df.to_csv('process_optimization_results.csv', index=False, encoding='utf-8')
    print("Optimization study completed. Results saved to 'process_optimization_results.csv'")
    
    return results_df

if __name__ == "__main__":
    # Run example simulation
    simulator, stage1_results, stage2_results = run_example_simulation()
    
    # Run optimization study
    optimization_results = plot_process_comparison()
    
    print("\n=== SIMULATION COMPLETED SUCCESSFULLY ===")
    print("Check the generated files and plots for detailed results!") 