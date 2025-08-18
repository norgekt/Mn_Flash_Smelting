import numpy as np
import matplotlib.pyplot as plt

# Data: T(°C) and t(min) for X=0.95
T_C = np.array([350, 375, 400, 425, 450, 500])
t_min = np.array([210, 80, 35, 15, 9.5, 3.5])
T_K = T_C + 273.15

# Calculate k values using kt = 1-(1-X)^(1/3) for X=0.95
X = 0.95
kt = 1 - (1 - X)**(1/3)
k_values = kt / t_min

# Arrhenius analysis: ln(k) = ln(k0) - E/RT
R = 8.314  # J/(mol·K)
inv_T = 1/T_K
ln_k = np.log(k_values)
coeffs = np.polyfit(inv_T, ln_k, 1)
E = -coeffs[0] * R  # Activation energy
k0 = np.exp(coeffs[1])  # Pre-exponential factor

# Results
print(f"kt = {kt:.6f}")
print(f"Activation Energy: {E:.0f} J/mol = {E/1000:.0f} kJ/mol")
print(f"Pre-exponential factor k0: {k0:.3e} min⁻¹")
print(f"Correlation: {np.corrcoef(inv_T, ln_k)[0,1]:.4f}") 