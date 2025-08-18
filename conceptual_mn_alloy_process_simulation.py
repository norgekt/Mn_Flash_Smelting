"""
Conceptual Mn Alloy Process Simulation
Two-stage hydrogen-based flash smelting process for ferromanganese alloy production

Stage 1: Pre-reduction with H₂ (MnO₂, Mn₂O₃ → MnO)
Stage 2: Slag-Metal Reaction (MnO + Al → Mn + Al₂O₃)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProcessParameters:
    """Process parameters for the two-stage reduction process"""
    # Stage 1 Parameters (H2 Reduction) - Based on user's validated model
    temperature_stage1: float = 1173.0  # K (900°C) - User specified 900-1000°C range
    h2_partial_pressure: float = 1.05   # atm
    h2o_partial_pressure: float = 0.2   # atm
    particle_size: float = 100e-6       # m (100 micrometers)
    residence_time: float = 2.0         # s - User specified ~2 seconds
    
    # Stage 2 Parameters (Al Reduction)
    temperature_stage2: float = 1823.0  # K (1550°C)
    al_content: float = 0.15            # Al fraction in metal bath
    cao_content: float = 0.25           # CaO flux addition
    mgo_content: float = 0.10           # MgO flux addition
    
    # User's Validated Kinetic Parameters for Stage 1
    # Rate constant: k = 91535*exp(-109721/RT)
    k0_reduction: float = 91535.0       # Pre-exponential factor (s^-1)
    ea_reduction: float = 109721.0      # Activation energy (J/mol)
    
    # Legacy kinetic parameters (kept for compatibility but not used in Stage 1)
    k1_mno2_h2: float = 1.2e-3         # Rate constant for MnO2 + H2 → Mn3O4
    k2_mn3o4_h2: float = 2.5e-3        # Rate constant for Mn3O4 + H2 → MnO
    k3_mn2o3_h2: float = 1.8e-3        # Rate constant for Mn2O3 + H2 → Mn3O4
    
    # Legacy activation energies (J/mol)
    ea1: float = 85000.0
    ea2: float = 95000.0
    ea3: float = 80000.0

class ManganesReductionSimulator:
    """
    Simulator for the two-stage manganese reduction process
    """
    
    def __init__(self, params: ProcessParameters):
        self.params = params
        self.R = 8.314  # Gas constant J/(mol·K)
        
        # Initialize data storage
        self.stage1_results = None
        self.stage2_results = None
        self.dnn_model = None
        self.scalers = None
        
    def calculate_rate_constants(self, temperature: float) -> Dict[str, float]:
        """Calculate temperature-dependent rate constants using Arrhenius equation"""
        k1 = self.params.k1_mno2_h2 * np.exp(-self.params.ea1 / (self.R * temperature))
        k2 = self.params.k2_mn3o4_h2 * np.exp(-self.params.ea2 / (self.R * temperature))
        k3 = self.params.k3_mn2o3_h2 * np.exp(-self.params.ea3 / (self.R * temperature))
        
        return {'k1': k1, 'k2': k2, 'k3': k3}
    
    def stage1_kinetic_model(self, y: List[float], t: float) -> List[float]:
        """
        Kinetic model for Stage 1 H2 reduction based on user's validated model
        
        Uses the reduction degree equation: RD = 1 - (1-kt)^3 (chemical reaction control)
        Rate constant: k = 91535*exp(-109721/RT)
        
        Assumption: Complete reduction of Mn oxides to MnO and Fe oxides to Fe
        Particle size < 0.1 mm, residence time ~2 seconds at 900-1000°C
        
        Variables: [MnO2, Mn2O3, Mn3O4, MnO, H2, H2O]
        """
        MnO2, Mn2O3, Mn3O4, MnO, H2, H2O = y
        
        # User's validated rate constant equation
        # k = k0*exp(-Ea/RT) where k0 = 91535 s^-1, Ea = 109721 J/mol
        k_reduction = self.params.k0_reduction * np.exp(-self.params.ea_reduction / (self.R * self.params.temperature_stage1))
        
        # Calculate reduction degree: RD = 1 - (1-kt)^3
        # For numerical stability, limit kt to prevent (1-kt) < 0
        kt = min(k_reduction * t, 0.999)
        reduction_degree = 1 - (1 - kt)**3
        
        # Calculate total initial Mn oxides (in moles)
        initial_mn_oxides = self.initial_mn_total if hasattr(self, 'initial_mn_total') else (MnO2 + Mn2O3 + Mn3O4 + MnO)
        
        # Target MnO production based on reduction degree
        target_mno = initial_mn_oxides * reduction_degree
        current_total_mn = MnO2 + Mn2O3 + Mn3O4 + MnO
        
        # Calculate reaction rates to achieve target composition
        # Prioritize reduction: MnO2 → Mn3O4 → MnO, Mn2O3 → Mn3O4 → MnO
        
        # Rate of MnO formation (driving force)
        dmno_dt = max(0, (target_mno - MnO) * k_reduction)
        
        # Stoichiometric rates for intermediate species
        # Simplified: assume direct conversion pathways
        if MnO2 > 0.001:  # If significant MnO2 remains
            dmno2_dt = -min(MnO2 * k_reduction, dmno_dt * 0.5)
        else:
            dmno2_dt = 0
            
        if Mn2O3 > 0.001:  # If significant Mn2O3 remains
            dmn2o3_dt = -min(Mn2O3 * k_reduction, dmno_dt * 0.3)
        else:
            dmn2o3_dt = 0
            
        if Mn3O4 > 0.001:  # If significant Mn3O4 remains
            dmn3o4_dt = -min(Mn3O4 * k_reduction, dmno_dt * 0.2) + abs(dmno2_dt) * 0.33 + abs(dmn2o3_dt) * 0.67
        else:
            dmn3o4_dt = 0
        
        # H2 consumption and H2O production
        total_reduction_rate = abs(dmno2_dt) + abs(dmn2o3_dt) + abs(dmn3o4_dt)
        dh2_dt = -total_reduction_rate  # H2 consumed
        dh2o_dt = total_reduction_rate  # H2O produced
        
        return [dmno2_dt, dmn2o3_dt, dmn3o4_dt, dmno_dt, dh2_dt, dh2o_dt]
    
    def run_stage1_simulation(self, initial_composition: Dict[str, float], 
                             time_range: Tuple[float, float, int]) -> pd.DataFrame:
        """
        Run Stage 1 H2 reduction simulation using validated kinetic model
        
        Based on user's model:
        - RD = 1 - (1-kt)^3 (chemical reaction control)
        - k = 91535*exp(-109721/RT)
        - Particle size < 0.1 mm, residence time ~2s at 900-1000°C
        
        Args:
            initial_composition: Initial masses of species (kg)
            time_range: (start_time, end_time, num_points) in seconds
            
        Returns:
            DataFrame with time series results
        """
        print("Running Stage 1 H2 Reduction Simulation...")
        print(f"Using validated kinetic model: RD = 1 - (1-kt)^3")
        print(f"Rate constant: k = 91535*exp(-109721/RT)")
        print(f"Temperature: {self.params.temperature_stage1:.0f}K ({self.params.temperature_stage1-273:.0f}°C)")
        
        # Convert initial composition to molar basis
        molecular_weights = {
            'MnO2': 86.94, 'Mn2O3': 157.87, 'Mn3O4': 228.81, 
            'MnO': 70.94, 'H2': 2.016, 'H2O': 18.015
        }
        
        y0 = [
            initial_composition.get('MnO2', 0) / molecular_weights['MnO2'],
            initial_composition.get('Mn2O3', 0) / molecular_weights['Mn2O3'],
            initial_composition.get('Mn3O4', 0) / molecular_weights['Mn3O4'],
            initial_composition.get('MnO', 0) / molecular_weights['MnO'],
            initial_composition.get('H2', 10) / molecular_weights['H2'],  # Excess H2
            initial_composition.get('H2O', 0) / molecular_weights['H2O']
        ]
        
        # Store initial total Mn for reduction degree calculation
        self.initial_mn_total = y0[0] + 2*y0[1] + 3*y0[2] + y0[3]  # Total Mn moles
        
        # Calculate expected rate constant at this temperature
        k_expected = self.params.k0_reduction * np.exp(-self.params.ea_reduction / (self.R * self.params.temperature_stage1))
        print(f"Rate constant k = {k_expected:.2e} s^-1")
        
        # For typical 2-second residence time, calculate expected RD
        if time_range[1] <= 10:  # Short simulation typical of flash smelting
            expected_rd_2s = 1 - (1 - k_expected * 2)**3 if k_expected * 2 < 1 else 1.0
            print(f"Expected RD at 2s residence time: {expected_rd_2s:.3f}")
        
        # Time vector
        t = np.linspace(time_range[0], time_range[1], time_range[2])
        
        # Solve ODE system
        solution = odeint(self.stage1_kinetic_model, y0, t)
        
        # Convert back to mass basis and create DataFrame
        species_names = ['MnO2', 'Mn2O3', 'Mn3O4', 'MnO', 'H2', 'H2O']
        results_df = pd.DataFrame(solution, columns=species_names)
        
        # Convert moles back to kg
        for i, species in enumerate(species_names):
            results_df[species] *= molecular_weights[species]
        
        results_df['Time'] = t
        results_df['Temperature'] = self.params.temperature_stage1
        
        # Calculate reduction degree using user's equation
        k_values = self.params.k0_reduction * np.exp(-self.params.ea_reduction / (self.R * self.params.temperature_stage1))
        kt_values = np.minimum(k_values * t, 0.999)  # Prevent numerical issues
        results_df['Reduction_Degree'] = 1 - (1 - kt_values)**3
        
        # Also calculate based on actual MnO production for comparison
        current_mno_moles = solution[:, 3]  # MnO moles
        results_df['RD_Actual'] = current_mno_moles / self.initial_mn_total if self.initial_mn_total > 0 else 0
        
        self.stage1_results = results_df
        print(f"Stage 1 simulation completed.")
        print(f"Final reduction degree (model): {results_df['Reduction_Degree'].iloc[-1]:.3f}")
        print(f"Final reduction degree (actual): {results_df['RD_Actual'].iloc[-1]:.3f}")
        
        return results_df
    
    def load_stage2_dnn_model(self, model_path: str = 'dnn_model.pkl'):
        """Load the DNN model for Stage 2 Al reduction"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.dnn_model = model_data['model']
            self.scalers = {
                'scaler_X': model_data['scaler_X'],
                'scaler_y': model_data['scaler_y'],
                'feature_names': model_data['feature_names'],
                'target_names': model_data['target_names']
            }
            print("DNN model loaded successfully for Stage 2")
            return True
        except FileNotFoundError:
            print(f"DNN model file {model_path} not found. Stage 2 will use simplified kinetic model.")
            return False
        except Exception as e:
            print(f"Error loading DNN model: {e}")
            return False
    
    def stage2_simplified_kinetics(self, stage1_output: Dict[str, float]) -> Dict[str, float]:
        """
        Simplified kinetic model for Stage 2 Al reduction when DNN is not available
        
        Main reaction: 3MnO + 2Al → 3Mn + Al2O3
        """
        print("Running Stage 2 Simplified Al Reduction...")
        
        # Extract MnO from Stage 1 output
        mno_available = stage1_output.get('MnO', 0)  # kg
        
        # Stoichiometry: 3MnO + 2Al → 3Mn + Al2O3
        mw_mno = 70.94
        mw_al = 26.98
        mw_mn = 54.94
        mw_al2o3 = 101.96
        
        mno_moles = mno_available / mw_mno
        
        # Assume 90% efficiency for Al reduction
        efficiency = 0.90
        mn_produced_moles = mno_moles * efficiency
        al_consumed_moles = (2/3) * mn_produced_moles
        al2o3_produced_moles = (1/3) * mn_produced_moles
        
        # Convert to masses
        results = {
            'Mn_produced': mn_produced_moles * mw_mn,
            'Al_consumed': al_consumed_moles * mw_al,
            'Al2O3_produced': al2o3_produced_moles * mw_al2o3,
            'MnO_remaining': (mno_moles - mn_produced_moles) * mw_mno,
            'efficiency': efficiency,
            'temperature': self.params.temperature_stage2
        }
        
        return results
    
    def run_stage2_simulation(self, stage1_output: Dict[str, float], 
                             initial_metal_composition: Dict[str, float] = None) -> Dict[str, float]:
        """
        Run Stage 2 Al reduction simulation
        
        Args:
            stage1_output: Output from Stage 1 (MnO content)
            initial_metal_composition: Initial metal bath composition
            
        Returns:
            Dictionary with Stage 2 results
        """
        print("Running Stage 2 Al Reduction Simulation...")
        
        if self.dnn_model is None:
            # Use simplified kinetics
            results = self.stage2_simplified_kinetics(stage1_output)
        else:
            # Use DNN model (requires proper input preparation)
            results = self.stage2_dnn_prediction(stage1_output, initial_metal_composition)
        
        self.stage2_results = results
        return results
    
    def stage2_dnn_prediction(self, stage1_output: Dict[str, float], 
                            initial_metal_composition: Dict[str, float]) -> Dict[str, float]:
        """Use DNN model for Stage 2 prediction"""
        # This would require proper input preparation based on the DNN model structure
        # For now, fall back to simplified kinetics
        print("DNN prediction not fully implemented, using simplified kinetics...")
        return self.stage2_simplified_kinetics(stage1_output)
    
    def calculate_overall_efficiency(self) -> Dict[str, float]:
        """Calculate overall process efficiency metrics"""
        if self.stage1_results is None or self.stage2_results is None:
            print("Both stages must be completed before calculating overall efficiency")
            return {}
        
        # Calculate metrics
        stage1_efficiency = self.stage1_results['Reduction_Degree'].iloc[-1]
        stage2_efficiency = self.stage2_results.get('efficiency', 0)
        overall_efficiency = stage1_efficiency * stage2_efficiency
        
        # Energy considerations (simplified)
        h2_consumption = self.stage1_results['H2'].iloc[0] - self.stage1_results['H2'].iloc[-1]
        al_consumption = self.stage2_results.get('Al_consumed', 0)
        
        metrics = {
            'stage1_reduction_degree': stage1_efficiency,
            'stage2_efficiency': stage2_efficiency,
            'overall_efficiency': overall_efficiency,
            'h2_consumption_kg': h2_consumption,
            'al_consumption_kg': al_consumption,
            'mn_production_kg': self.stage2_results.get('Mn_produced', 0)
        }
        
        return metrics
    
    def plot_results(self, save_plots: bool = True):
        """Generate comprehensive plots of simulation results"""
        if self.stage1_results is None:
            print("No results to plot. Run simulations first.")
            return
        
        # Create plots directory
        if save_plots and not os.path.exists('simulation_plots'):
            os.makedirs('simulation_plots')
        
        # Stage 1 Results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Species evolution
        axes[0, 0].plot(self.stage1_results['Time'], self.stage1_results['MnO2'], 'r-', label='MnO2')
        axes[0, 0].plot(self.stage1_results['Time'], self.stage1_results['Mn2O3'], 'b-', label='Mn2O3')
        axes[0, 0].plot(self.stage1_results['Time'], self.stage1_results['Mn3O4'], 'g-', label='Mn3O4')
        axes[0, 0].plot(self.stage1_results['Time'], self.stage1_results['MnO'], 'k-', label='MnO')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Mass (kg)')
        axes[0, 0].set_title('Stage 1: Mn Species Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # H2 consumption
        axes[0, 1].plot(self.stage1_results['Time'], self.stage1_results['H2'], 'c-', label='H2')
        axes[0, 1].plot(self.stage1_results['Time'], self.stage1_results['H2O'], 'm-', label='H2O')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Mass (kg)')
        axes[0, 1].set_title('Stage 1: H2 Consumption and H2O Production')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reduction degree
        axes[1, 0].plot(self.stage1_results['Time'], self.stage1_results['Reduction_Degree'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Reduction Degree')
        axes[1, 0].set_title('Stage 1: Reduction Degree Progress')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Stage 2 Results (bar chart)
        if self.stage2_results:
            species = ['Mn Produced', 'Al Consumed', 'Al2O3 Produced', 'MnO Remaining']
            masses = [
                self.stage2_results.get('Mn_produced', 0),
                self.stage2_results.get('Al_consumed', 0),
                self.stage2_results.get('Al2O3_produced', 0),
                self.stage2_results.get('MnO_remaining', 0)
            ]
            axes[1, 1].bar(species, masses, color=['red', 'blue', 'green', 'orange'])
            axes[1, 1].set_ylabel('Mass (kg)')
            axes[1, 1].set_title('Stage 2: Material Balance')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('simulation_plots/process_simulation_results.png', dpi=300)
            print("Plots saved to simulation_plots/process_simulation_results.png")
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive simulation report"""
        if self.stage1_results is None or self.stage2_results is None:
            return "Simulation incomplete. Run both stages first."
        
        efficiency_metrics = self.calculate_overall_efficiency()
        
        report = f"""
=== CONCEPTUAL MN ALLOY PROCESS SIMULATION REPORT ===

PROCESS PARAMETERS:
- Stage 1 Temperature: {self.params.temperature_stage1:.0f} K ({self.params.temperature_stage1-273:.0f}°C)
- Stage 2 Temperature: {self.params.temperature_stage2:.0f} K ({self.params.temperature_stage2-273:.0f}°C)
- H2 Partial Pressure: {self.params.h2_partial_pressure:.2f} atm
- Particle Size: {self.params.particle_size*1e6:.0f} micrometers

STAGE 1 RESULTS (H2 Reduction):
- Final Reduction Degree: {efficiency_metrics['stage1_reduction_degree']:.3f}
- H2 Consumption: {efficiency_metrics['h2_consumption_kg']:.2f} kg
- Final MnO Production: {self.stage1_results['MnO'].iloc[-1]:.2f} kg

STAGE 2 RESULTS (Al Reduction):
- Process Efficiency: {efficiency_metrics['stage2_efficiency']:.3f}
- Mn Production: {efficiency_metrics['mn_production_kg']:.2f} kg
- Al Consumption: {efficiency_metrics['al_consumption_kg']:.2f} kg

OVERALL PERFORMANCE:
- Overall Process Efficiency: {efficiency_metrics['overall_efficiency']:.3f}
- Total Mn Production: {efficiency_metrics['mn_production_kg']:.2f} kg

ENVIRONMENTAL BENEFITS:
- CO2 Reduction: Using H2 instead of carbon reduces CO2 emissions
- Energy Efficiency: Two-stage process optimizes energy usage
- Resource Utilization: High Al utilization in Stage 2

=== END REPORT ===
        """
        
        return report

def main():
    """Main simulation function"""
    print("=== CONCEPTUAL MN ALLOY PROCESS SIMULATION ===\n")
    
    # Initialize process parameters
    params = ProcessParameters()
    
    # Create simulator
    simulator = ManganesReductionSimulator(params)
    
    # Try to load DNN model for Stage 2
    simulator.load_stage2_dnn_model()
    
    # Define initial ore composition (example)
    initial_ore = {
        'MnO2': 50.0,    # kg
        'Mn2O3': 20.0,   # kg
        'Mn3O4': 5.0,    # kg
        'MnO': 2.0,      # kg
    }
    
    # Run Stage 1 simulation
    stage1_results = simulator.run_stage1_simulation(
        initial_composition=initial_ore,
        time_range=(0, 3600, 100)  # 1 hour simulation with 100 time points
    )
    
    # Extract final Stage 1 composition for Stage 2
    final_stage1_composition = {
        'MnO': stage1_results['MnO'].iloc[-1],
        'Mn3O4': stage1_results['Mn3O4'].iloc[-1],
        'MnO2': stage1_results['MnO2'].iloc[-1],
        'Mn2O3': stage1_results['Mn2O3'].iloc[-1]
    }
    
    # Define initial metal bath composition
    initial_metal = {
        'Al': 15.0,      # kg
        'Fe': 50.0,      # kg
        'Si': 5.0,       # kg
        'Mn': 2.0        # kg (existing)
    }
    
    # Run Stage 2 simulation
    stage2_results = simulator.run_stage2_simulation(
        stage1_output=final_stage1_composition,
        initial_metal_composition=initial_metal
    )
    
    # Generate and display results
    print("\n" + simulator.generate_report())
    
    # Create plots
    simulator.plot_results(save_plots=True)
    
    # Save detailed results
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')
    
    stage1_results.to_csv('simulation_results/stage1_results.csv', index=False)
    
    stage2_df = pd.DataFrame([stage2_results])
    stage2_df.to_csv('simulation_results/stage2_results.csv', index=False)
    
    print("\nDetailed results saved to 'simulation_results/' directory")
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main() 