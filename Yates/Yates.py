import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error

# Load the CSV file
data = pd.read_csv('C:/Users/amgh628/Downloads/Yates_etal/Yates_Mahdi/Tiwai/Tiwai_XGB_filled.csv', parse_dates=['t'])
data.set_index('t', inplace=True)

# Convert data to daily resolution by interpolating missing values
data = data.resample('D').interpolate(method='linear', limit_direction='forward')

# Separate the data into training (up to 2021) and prediction (2022 onwards) periods
train_data = data[:'2021-12-31']
predict_data = data['2022-01-01':]

# Calculate wave energy as E = H^2
train_data.loc[:, 'wave_energy'] = train_data['Hs'] ** 2
predict_data.loc[:, 'wave_energy'] = predict_data['Hs'] ** 2

# Extract shoreline position and wave energy for training
S_obs = train_data['shore'].values  # Observed shoreline position
E_obs = train_data['wave_energy'].values  # Wave energy

# Detrend the observed shoreline positions
time_index = np.arange(len(S_obs))
trend_coeffs = np.polyfit(time_index, S_obs, 1)
trend = np.polyval(trend_coeffs, time_index)
S_obs_detrended = S_obs - trend

class EquilibriumShorelineModel:
    def __init__(self, a, b, C_plus, C_minus):
        self.a = a
        self.b = b
        self.C_plus = C_plus
        self.C_minus = C_minus

    def equilibrium_wave_energy(self, S):
        return self.a * S + self.b

    def shoreline_change_rate(self, S, E):
        E_eq = self.equilibrium_wave_energy(S)
        delta_E = E - E_eq
        delta_E = np.clip(delta_E, -1e6, 1e6)
        if delta_E < 0:
            return self.C_plus * np.sqrt(E) * abs(delta_E)  # Accretion
        else:
            return -self.C_minus * np.sqrt(E) * abs(delta_E)  # Erosion

    def simulate(self, initial_S, E_series, dt=1):
        S_series = [initial_S]
        for E in E_series:
            dS_dt = self.shoreline_change_rate(S_series[-1], E)
            S_series.append(np.nan_to_num(S_series[-1] + dS_dt * dt))
        return np.array(S_series[:-1])

# Define the custom loss function with weights and regularization
def custom_loss_function(params, initial_S_detrended, E_obs, trend, S_obs, w_rmse=1.0, w_cc=1.0, w_std=1.0, reg_lambda=0.01):
    """
    Calculate the custom loss function with weights and regularization.
    
    Parameters:
    - params: tuple of (a, b, C_plus, C_minus) model parameters
    - initial_S_detrended: initial detrended shoreline position
    - E_obs: observed wave energy series
    - trend: linear trend array to add back to simulated series
    - S_obs: observed shoreline positions on original scale
    - w_rmse, w_cc, w_std: weights for RMSE, correlation, and std components
    - reg_lambda: regularization strength
    
    Returns:
    - loss: value of the custom loss function
    """
    a, b, C_plus, C_minus = params
    model = EquilibriumShorelineModel(a, b, C_plus, C_minus)
    S_sim_detrended = model.simulate(initial_S_detrended, E_obs)
    S_sim = S_sim_detrended + trend
    
    # Calculate metrics
    cc = np.corrcoef(S_obs, S_sim)[0, 1]
    rmse = np.sqrt(mean_squared_error(S_obs, S_sim))
    std_pred = np.std(S_sim)
    std_obs = np.std(S_obs)
    
    # Normalized metrics
    norm_rmse = rmse / std_obs if std_obs != 0 else np.inf
    norm_std = std_pred / std_obs if std_obs != 0 else np.inf
    
    # Weighted loss with regularization
    loss = np.sqrt((w_rmse * norm_rmse)**2 + (w_cc * (1 - cc))**2 + (w_std * (1 - norm_std))**2)
    reg_term = reg_lambda * (a**2 + b**2 + C_plus**2 + C_minus**2)  # L2 regularization
    return loss + reg_term

# Define parameter bounds
param_bounds = [
    (-0.7, -0.65),  # a: Negative slope
    (1.5, 3.5),     # b: Intercept
    (0.5, 1.5),     # C_plus: Accretion coefficient
    (0, 1),         # C_minus: Erosion coefficient
]

# Initial guess (refined based on typical values)
initial_params = [-0.675, 2.5, 1.0, 0.5]

# Set the initial detrended shoreline position
initial_S_detrended = S_obs_detrended[0]

# Hybrid optimization: Global search with differential_evolution, then local refinement with L-BFGS-B
def optimize_parameters():
    # Step 1: Global optimization
    result_global = differential_evolution(
        custom_loss_function,
        bounds=param_bounds,
        args=(initial_S_detrended, E_obs, trend, S_obs, 1.0, 1.0, 1.0, 0.01),
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        disp=True
    )
    
    # Step 2: Local refinement
    result_local = minimize(
        custom_loss_function,
        result_global.x,
        args=(initial_S_detrended, E_obs, trend, S_obs, 1.0, 1.0, 1.0, 0.01),
        bounds=param_bounds,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 10000, 'ftol': 1e-8}
    )
    
    return result_local

# Run optimization
result = optimize_parameters()

# Check optimization success
if result.success:
    print("Optimization successful!")
else:
    print(f"Optimization failed: {result.message}")

# Extract optimized parameters
a_opt, b_opt, C_plus_opt, C_minus_opt = result.x
print(f"Optimized parameters: a={a_opt:.4f}, b={b_opt:.2f}, C_plus={C_plus_opt:.4f}, C_minus={C_minus_opt:.4f}")

# Create the model with optimized parameters
model = EquilibriumShorelineModel(a_opt, b_opt, C_plus_opt, C_minus_opt)

# Simulate shoreline positions for the training period
S_sim_train_detrended = model.simulate(S_obs_detrended[0], E_obs)
S_sim_train = S_sim_train_detrended + trend

# Extract wave energy for the prediction period
E_predict = predict_data['wave_energy'].values

# Predict shoreline positions for 2022 onwards
initial_S_detrended = S_obs_detrended[-1]
S_predict_detrended = model.simulate(initial_S_detrended, E_predict)
time_index_predict = np.arange(len(S_obs), len(S_obs) + len(S_predict_detrended))
trend_predict = np.polyval(trend_coeffs, time_index_predict)
S_predict = S_predict_detrended + trend_predict

# Add predictions to the prediction dataframe
predict_data['predicted_shoreline'] = S_predict
train_data['modeled_shoreline'] = S_sim_train

# Combine the training and prediction data
combined_data = pd.concat([train_data, predict_data])
# Uncomment to save: combined_data.to_csv('C:/Users/amgh628/Downloads/Yates_etal/Yates_Mahdi/OTAMA/OTAMA_pred_SG_linear_new.csv')

# Calculate metrics for training period (1999-2021)
cc_train = np.corrcoef(S_obs, S_sim_train)[0, 1]
rmse_train = np.sqrt(mean_squared_error(S_obs, S_sim_train))
std_train_pred = np.std(S_sim_train)
std_train_obs = np.std(S_obs)
norm_std_train = std_train_pred / std_train_obs
norm_rmse_train = rmse_train / std_train_obs
loss_train = np.sqrt((0 - norm_rmse_train) ** 2 + (1 - cc_train) ** 2 + (1 - norm_std_train) ** 2)

# Calculate metrics for testing period (2022 onwards)
y_test = predict_data['shore'].values  # Observed shoreline in prediction period
y_pred = S_predict  # Predicted shoreline in prediction period
cc_test = np.corrcoef(y_test, y_pred)[0, 1]
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
std_test_pred = np.std(y_pred)
std_test_obs = np.std(y_test)
norm_std_test = std_test_pred / std_test_obs
norm_rmse_test = rmse_test / std_test_obs
loss_test = np.sqrt((0 - norm_rmse_test) ** 2 + (1 - cc_test) ** 2 + (1 - norm_std_test) ** 2)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(train_data.index, S_obs, label='Observed Shoreline (1999-2021)', color='blue')
plt.plot(train_data.index, S_sim_train, label='Modeled Shoreline (1999-2021)', color='red', linestyle='--')
plt.plot(predict_data.index, S_predict, label='Predicted Shoreline (2022 onwards)', color='red', linestyle='--')
plt.plot(predict_data.index, y_test, label='Observed Shoreline (2022 onwards)', color='green', linestyle=':')

# Add metrics to the plot
plt.text(
    0.05, 0.95,
    f'Training (1999-2021):\nCC = {cc_train:.3f}\nNorm RMSE = {norm_rmse_train:.3f}\nNorm STD = {norm_std_train:.3f}\nLoss = {loss_train:.3f}\n\n'
    f'Testing (2022 onwards):\nCC = {cc_test:.3f}\nNorm RMSE = {norm_rmse_test:.3f}\nNorm STD = {norm_std_test:.3f}\nLoss = {loss_test:.3f}',
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

plt.xlabel('Date')
plt.ylabel('Shoreline Position')
plt.title('Shoreline Position: Observed, Modeled (1999-2021), and Predicted (2022 onwards)')
plt.legend()
plt.show()