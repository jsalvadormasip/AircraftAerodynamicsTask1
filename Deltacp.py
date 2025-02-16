import pandas as pd
import numpy as np
from scipy import interpolate

# Load Cp data from XFOIL (assuming the file contains columns: 'x_upper', 'Cp_upper', 'x_lower', 'Cp_lower')
# data = pd.read_csv('cp_data.txt', sep='\t')  # Adjust the separator as needed
#load code values:
csv_file = 'aircraftaerodynamicsassingment1.csv'
data = pd.read_csv(csv_file)
x_values = data['x'].values
delta_cp_values = data['Cp'].values
# Extract the x and Cp values for both upper and lower surfaces
x_upper = np.array([1, 0.99592, 0.98923, 0.98171, 0.97316, 0.96332, 0.95183, 0.93819, 0.92179, 0.90185, 
 0.87763, 0.84878, 0.81567, 0.77938, 0.74114, 0.7019, 0.66224, 0.62253, 0.58299, 
 0.54379, 0.50507, 0.46701, 0.42979, 0.39368, 0.35887, 0.3253, 0.29299, 0.2621, 
 0.23282, 0.20535, 0.17991, 0.15666, 0.13567, 0.11692, 0.1003, 0.08565, 0.07277, 
 0.06144, 0.05149, 0.04274, 0.03504, 0.02829, 0.02239, 0.01726, 0.01285, 0.00913, 
 0.00608, 0.00367, 0.00188, 0.0007])

cp_upper = np.array([0.58706, 0.43976, 0.359, 0.29924, 0.24881, 0.20327, 0.15821, 0.11393, 0.06755, 
 0.01891, -0.0329, -0.08766, -0.14384, -0.20029, -0.25596, -0.31047, -0.3642, 
 -0.41717, -0.4701, -0.52301, -0.57627, -0.63003, -0.68411, -0.74656, -0.80048, 
 -0.84331, -0.87838, -0.90433, -0.92116, -0.92754, -0.92284, -0.90687, -0.87987, 
 -0.84254, -0.79468, -0.73773, -0.67103, -0.59525, -0.51013, -0.41421, -0.3082, 
 -0.18982, -0.06055, 0.08134, 0.23334, 0.39224, 0.55201, 0.70345, 0.83463, 0.93304])
x_lower = np.array([1, 0.99579, 0.98888, 0.98111, 0.97225, 0.96203, 0.95004, 0.93572, 0.91833, 0.89695, 
 0.87062, 0.83886, 0.80219, 0.76202, 0.71987, 0.67686, 0.63363, 0.59056, 0.54787, 
 0.50572, 0.4642, 0.42335, 0.38314, 0.34385, 0.30606, 0.27028, 0.23694, 0.20641, 
 0.17893, 0.15456, 0.13317, 0.11452, 0.09826, 0.08408, 0.07165, 0.06072, 0.05106, 
 0.0425, 0.0349, 0.02817, 0.02223, 0.01703, 0.01254, 0.00876, 0.00567, 0.00328, 
 0.00157, 0.0005, 0.00003, 0.0001])

cp_lower = np.array([0.58706, 0.45359, 0.38424, 0.33549, 0.29675, 0.26206, 0.23002, 0.19848, 0.16691, 
 0.13438, 0.10072, 0.06569, 0.03027, -0.00515, -0.04009, -0.07495, -0.10994, 
 -0.14536, -0.18143, -0.218, -0.25454, -0.29007, -0.31775, -0.35421, -0.3944, 
 -0.43525, -0.47485, -0.51108, -0.54261, -0.56755, -0.58555, -0.595, -0.59581, 
 -0.58749, -0.56855, -0.53804, -0.49443, -0.43571, -0.3586, -0.26171, -0.1415, 
 0.00259, 0.16985, 0.35363, 0.54231, 0.7184, 0.86085, 0.95592, 0.99757, 0.98889])

# Interpolate Cp values for the lower surface to match the x-coordinates of the upper surface
interp_func_lower = interpolate.interp1d(x_lower, cp_lower, kind='linear', fill_value="extrapolate")

# Interpolate Cp values for the upper surface to match the x-coordinates of the lower surface
interp_func_upper = interpolate.interp1d(x_upper, cp_upper, kind='linear', fill_value="extrapolate")

# Choose the x-coordinates where you want to compute Delta Cp (can use x_upper or x_lower or a new set)
x_common = np.linspace(min(x_upper.min(), x_lower.min()), max(x_upper.max(), x_lower.max()), 100)  # Common x values

# Compute interpolated Cp values for both surfaces
cp_upper_interp = interp_func_upper(x_common)
cp_lower_interp = interp_func_lower(x_common)

# Calculate Delta Cp (Cp_upper - Cp_lower)
delta_cp = cp_upper_interp - cp_lower_interp

# Output results
for x, delta in zip(x_common, delta_cp):
    print(f"x = {x:.4f}, Delta Cp = {delta:.4f}")

# Plot Delta Cp vs x
import matplotlib.pyplot as plt
plt.plot(x_common, -delta_cp, label = 'XFOIL')
plt.plot(x_values, delta_cp_values, label = 'Code')
plt.xlabel('x')
plt.ylabel('Delta Cp')
plt.title('Delta Cp vs x for NACA 2421')
plt.grid()
plt.legend()
plt.show()

