"""
Simple test script to verify the Mn alloy process simulation components
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.integrate import odeint
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neural_network import MLPRegressor
        import pickle
        from dataclasses import dataclass
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False

def test_process_parameters():
    """Test ProcessParameters dataclass"""
    print("\nTesting ProcessParameters...")
    try:
        from conceptual_mn_alloy_process_simulation import ProcessParameters
        params = ProcessParameters()
        print(f"✓ ProcessParameters created with default temperature_stage1: {params.temperature_stage1}K")
        
        # Test custom parameters
        custom_params = ProcessParameters(temperature_stage1=1050.0, h2_partial_pressure=0.9)
        print(f"✓ Custom ProcessParameters: T1={custom_params.temperature_stage1}K, P_H2={custom_params.h2_partial_pressure}atm")
        return True
    except Exception as e:
        print(f"✗ ProcessParameters test failed: {e}")
        traceback.print_exc()
        return False

def test_simulator_creation():
    """Test ManganesReductionSimulator creation"""
    print("\nTesting ManganesReductionSimulator creation...")
    try:
        from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters
        params = ProcessParameters()
        simulator = ManganesReductionSimulator(params)
        print("✓ ManganesReductionSimulator created successfully")
        
        # Test rate constants calculation
        rate_constants = simulator.calculate_rate_constants(1000.0)
        print(f"✓ Rate constants calculated: k1={rate_constants['k1']:.2e}, k2={rate_constants['k2']:.2e}")
        return True
    except Exception as e:
        print(f"✗ Simulator creation test failed: {e}")
        traceback.print_exc()
        return False

def test_stage1_simulation():
    """Test Stage 1 simulation with minimal data"""
    print("\nTesting Stage 1 simulation...")
    try:
        from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters
        
        params = ProcessParameters()
        simulator = ManganesReductionSimulator(params)
        
        # Simple test composition
        test_ore = {
            'MnO2': 10.0,   # kg
            'Mn2O3': 5.0,   # kg
            'Mn3O4': 2.0,   # kg
            'MnO': 1.0,     # kg
        }
        
        # Short simulation for testing
        results = simulator.run_stage1_simulation(
            initial_composition=test_ore,
            time_range=(0, 600, 10)  # 10 minutes, 10 points
        )
        
        print(f"✓ Stage 1 simulation completed")
        print(f"  - Final reduction degree: {results['Reduction_Degree'].iloc[-1]:.3f}")
        print(f"  - Final MnO content: {results['MnO'].iloc[-1]:.2f} kg")
        return True, results
    except Exception as e:
        print(f"✗ Stage 1 simulation test failed: {e}")
        traceback.print_exc()
        return False, None

def test_stage2_simulation():
    """Test Stage 2 simulation"""
    print("\nTesting Stage 2 simulation...")
    try:
        from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters
        
        params = ProcessParameters()
        simulator = ManganesReductionSimulator(params)
        
        # Test Stage 2 with mock Stage 1 output
        stage1_output = {
            'MnO': 15.0,     # kg
            'Mn3O4': 2.0,    # kg
            'MnO2': 0.5,     # kg
            'Mn2O3': 0.2     # kg
        }
        
        metal_composition = {
            'Al': 10.0,      # kg
            'Fe': 40.0,      # kg
            'Si': 3.0,       # kg
            'Mn': 2.0        # kg
        }
        
        results = simulator.run_stage2_simulation(
            stage1_output=stage1_output,
            initial_metal_composition=metal_composition
        )
        
        print(f"✓ Stage 2 simulation completed")
        print(f"  - Mn produced: {results['Mn_produced']:.2f} kg")
        print(f"  - Al consumed: {results['Al_consumed']:.2f} kg")
        print(f"  - Efficiency: {results['efficiency']:.1%}")
        return True, results
    except Exception as e:
        print(f"✗ Stage 2 simulation test failed: {e}")
        traceback.print_exc()
        return False, None

def test_full_process():
    """Test complete process simulation"""
    print("\nTesting complete process simulation...")
    try:
        from conceptual_mn_alloy_process_simulation import ManganesReductionSimulator, ProcessParameters
        
        params = ProcessParameters()
        simulator = ManganesReductionSimulator(params)
        
        # Run Stage 1
        test_ore = {
            'MnO2': 20.0, 'Mn2O3': 8.0, 'Mn3O4': 3.0, 'MnO': 1.0
        }
        
        stage1_results = simulator.run_stage1_simulation(
            initial_composition=test_ore,
            time_range=(0, 1800, 20)  # 30 minutes, 20 points
        )
        
        # Extract final composition for Stage 2
        final_composition = {
            'MnO': stage1_results['MnO'].iloc[-1],
            'Mn3O4': stage1_results['Mn3O4'].iloc[-1],
            'MnO2': stage1_results['MnO2'].iloc[-1],
            'Mn2O3': stage1_results['Mn2O3'].iloc[-1]
        }
        
        # Run Stage 2
        stage2_results = simulator.run_stage2_simulation(
            stage1_output=final_composition,
            initial_metal_composition={'Al': 15.0, 'Fe': 60.0, 'Si': 5.0}
        )
        
        # Calculate overall efficiency
        efficiency_metrics = simulator.calculate_overall_efficiency()
        
        print(f"✓ Complete process simulation successful")
        print(f"  - Overall efficiency: {efficiency_metrics['overall_efficiency']:.1%}")
        print(f"  - Total Mn production: {efficiency_metrics['mn_production_kg']:.2f} kg")
        
        # Test report generation
        report = simulator.generate_report()
        print(f"✓ Report generated successfully (length: {len(report)} chars)")
        
        return True
    except Exception as e:
        print(f"✗ Complete process test failed: {e}")
        traceback.print_exc()
        return False

def test_parameter_study():
    """Test the parameter study helper."""
    print("\nTesting parameter study...")
    try:
        from conceptual_mn_alloy_process_simulation import ProcessParameters
        from parameter_study import parameter_study, plot_recovery_vs_temperature, plot_recovery_vs_al_content
        import matplotlib

        # Use non-interactive backend for tests
        matplotlib.use("Agg")

        params_list = [
            ProcessParameters(temperature_stage1=1100.0, al_content=0.10),
            ProcessParameters(temperature_stage1=1200.0, al_content=0.20),
        ]

        results_df = parameter_study(params_list)
        print(f"✓ Parameter study produced DataFrame with {len(results_df)} rows")

        # Basic plotting calls to ensure functions execute
        plot_recovery_vs_temperature(results_df)
        plot_recovery_vs_al_content(results_df)

        return True
    except Exception as e:
        print(f"✗ Parameter study test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== CONCEPTUAL MN ALLOY PROCESS SIMULATION TESTS ===\n")
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("ProcessParameters", test_process_parameters()))
    test_results.append(("Simulator Creation", test_simulator_creation()))
    
    stage1_success, _ = test_stage1_simulation()
    test_results.append(("Stage 1 Simulation", stage1_success))
    
    stage2_success, _ = test_stage2_simulation()
    test_results.append(("Stage 2 Simulation", stage2_success))

    test_results.append(("Complete Process", test_full_process()))
    test_results.append(("Parameter Study", test_parameter_study()))
    
    # Summary
    print(f"\n=== TEST SUMMARY ===")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The simulation is ready to use.")
        print("\nNext steps:")
        print("1. Run the example: python run_mn_alloy_simulation.py")
        print("2. Modify parameters in ProcessParameters for your specific case")
        print("3. Use your own ore composition data")
        print("4. Add the DNN model file (dnn_model.pkl) for advanced Stage 2 modeling")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 