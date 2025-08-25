"""Parameter study utilities for Mn flash smelting simulations."""

from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from conceptual_mn_alloy_process_simulation import (
    ManganesReductionSimulator,
    ProcessParameters,
)


def parameter_study(params_list: List[ProcessParameters]) -> pd.DataFrame:
    """Run simulations for multiple sets of process parameters.

    Parameters
    ----------
    params_list:
        List of :class:`ProcessParameters` instances representing different
        operating conditions.

    Returns
    -------
    pd.DataFrame
        DataFrame summarising key outputs for each parameter set. Columns
        include stage 1 temperature, Al content, Mn recovery and material
        consumption metrics.
    """

    records = []

    # Default compositions used for all simulations
    initial_ore_composition = {
        "MnO2": 50.0,
        "Mn2O3": 15.0,
        "Mn3O4": 5.0,
        "MnO": 2.0,
        "H2": 50.0,
        "H2O": 0.0,
    }

    metal_bath_composition = {
        "Al": 20.0,
        "Fe": 80.0,
        "Si": 8.0,
        "Mn": 5.0,
        "C": 2.0,
    }

    for params in params_list:
        simulator = ManganesReductionSimulator(params)

        stage1 = simulator.run_stage1_simulation(
            initial_composition=initial_ore_composition,
            time_range=(0, 5, 50),
        )

        stage1_output = {
            "MnO": stage1["MnO"].iloc[-1],
            "Mn3O4": stage1["Mn3O4"].iloc[-1],
            "MnO2": stage1["MnO2"].iloc[-1],
            "Mn2O3": stage1["Mn2O3"].iloc[-1],
        }

        stage2 = simulator.run_stage2_simulation(
            stage1_output=stage1_output,
            initial_metal_composition=metal_bath_composition,
        )

        metrics = simulator.calculate_overall_efficiency()

        records.append(
            {
                "temperature_stage1": params.temperature_stage1,
                "al_content": params.al_content,
                "mn_recovery_kg": stage2.get("Mn_produced", 0.0),
                "slag_Al2O3_kg": stage2.get("Al2O3_produced", 0.0),
                "slag_MnO_remaining_kg": stage2.get("MnO_remaining", 0.0),
                "h2_consumption_kg": metrics.get("h2_consumption_kg", 0.0),
                "al_consumption_kg": metrics.get("al_consumption_kg", 0.0),
                "overall_efficiency": metrics.get("overall_efficiency", 0.0),
            }
        )

    return pd.DataFrame(records)


def plot_recovery_vs_temperature(results: pd.DataFrame) -> None:
    """Plot Mn recovery as a function of Stage 1 temperature."""

    if results.empty:
        print("No results to plot")
        return

    plt.figure()
    plt.plot(results["temperature_stage1"], results["mn_recovery_kg"], "o-")
    plt.xlabel("Stage 1 Temperature (K)")
    plt.ylabel("Mn Recovery (kg)")
    plt.title("Mn Recovery vs. Stage 1 Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_recovery_vs_al_content(results: pd.DataFrame) -> None:
    """Plot Mn recovery as a function of Al content."""

    if results.empty:
        print("No results to plot")
        return

    plt.figure()
    plt.plot(results["al_content"], results["mn_recovery_kg"], "s-")
    plt.xlabel("Al Content in Bath (fraction)")
    plt.ylabel("Mn Recovery (kg)")
    plt.title("Mn Recovery vs. Al Content")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

