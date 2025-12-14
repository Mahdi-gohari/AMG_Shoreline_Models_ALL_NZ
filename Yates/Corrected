import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.api.types import is_datetime64_any_dtype
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """Load and preprocess CSV data."""
    try:
        data = pd.read_csv(file_path, parse_dates=['t'], dayfirst=True)
        if not is_datetime64_any_dtype(data['t']):
            logging.warning("Date parsing may have failed; 't' column is not datetime.")
        data.set_index('t', inplace=True)
        data = data.resample('D').interpolate(method='linear', limit_direction='forward')
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_and_prepare_data(data, split_date='2020-12-31'):
    """Split data into training and prediction periods, calculate wave energy, and detrend training data."""
    train_data = data[:split_date].copy()
    predict_data = data[split_date:].copy()
    
    train_data['wave_energy'] = train_data['Hs'] ** 2
    predict_data['wave_energy'] = predict_data['Hs'] ** 2
    
    S_obs = train_data['shore'].values
    time_index = np.arange(len(S_obs))
    trend_coeffs = np.polyfit(time_index, S_obs, 1)
    trend = np.polyval(trend_coeffs, time_index)
    S_obs_detrended = S_obs - trend
    
    return train_data, predict_data, S_obs, S_obs_detrended, trend, trend_coeffs

class EquilibriumShorelineModel:
    def __init__(self, a, b, C_plus, C_minus):
        """Initialize shoreline model with parameters."""
        if C_plus < 0 or C_minus < 0:
            raise ValueError("C_plus and C_minus must be non-negative.")
        self.a = a
        self.b = b
        self.C_plus = C_plus
        self.C_minus = C_minus

    def equilibrium_wave_energy(self, S):
        """Calculate equilibrium wave energy."""
        return self.a * S + self.b

    def shoreline_change_rate(self, S, E):
        """Calculate shoreline change rate based on wave energy."""
        E_eq = self.equilibrium_wave_energy(S)
        delta_E = np.clip(E - E_eq, -1e6, 1e6)
        return (self.C_plus if delta_E < 0 else -self.C_minus) * np.sqrt(np.abs(E)) * np.abs(delta_E)

    def simulate(self, initial_S, E_series, dt=1):
        """Simulate shoreline positions for given wave energy series."""
        S_series = np.zeros(len(E_series) + 1)
        S_series[0] = initial_S
        for i, E in enumerate(E_series):
            dS_dt = self.shoreline_change_rate(S_series[i], E)
            S_series[i + 1] = S_series[i] + dS_dt * dt
        return np.nan_to_num(S_series[:-1])

def mse_loss_function(params, initial_S_detrended, E_obs, trend, S_obs):
    """Calculate mean squared error loss between observed and simulated shoreline positions."""
    try:
        a, b, C_plus, C_minus = params
        model = EquilibriumShorelineModel(a, b, C_plus, C_minus)
        S_sim_detrended = model.simulate(initial_S_detrended, E_obs)
        S_sim = S_sim_detrended + trend
        return mean_squared_error(S_obs, S_sim)
    except Exception as e:
        logging.warning(f"Error in loss function: {e}")
        return np.inf

def optimize_parameters(initial_params, param_bounds, initial_S_detrended, E_obs, trend, S_obs):
    """Optimize model parameters using global and local optimization."""
    try:
        result_global = differential_evolution(
            mse_loss_function,
            bounds=param_bounds,
            args=(initial_S_detrended, E_obs, trend, S_obs),
            maxiter=500,
            popsize=10,
            tol=1e-4,
            disp=False
        )
        
        result_local = minimize(
            mse_loss_function,
            result_global.x,
            args=(initial_S_detrended, E_obs, trend, S_obs),
            bounds=param_bounds,
            method='L-BFGS-B',
            options={'disp': False, 'maxiter': 5000, 'ftol': 1e-6}
        )
        return result_local
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

def calculate_metrics(S_obs, S_sim):
    """Calculate evaluation metrics."""
    cc = np.corrcoef(S_obs, S_sim)[0, 1]
    rmse = np.sqrt(mean_squared_error(S_obs, S_sim))
    std_obs = np.std(S_obs)
    std_pred = np.std(S_sim)
    norm_rmse = rmse / std_obs if std_obs != 0 else np.inf
    norm_std = std_pred / std_obs if std_obs != 0 else np.inf
    return {'cc': cc, 'rmse': rmse, 'norm_rmse': norm_rmse, 'norm_std': norm_std}

def plot_results(train_data, predict_data, S_obs, S_sim_train, S_predict, y_test, metrics_train, metrics_test):
    """Plot observed, modeled, and predicted shoreline positions with metrics."""
    if not (isinstance(train_data.index, pd.DatetimeIndex) and isinstance(predict_data.index, pd.DatetimeIndex)):
        logging.warning("Data indices are not DatetimeIndex; plot may not format dates correctly.")
    
    plt.figure(figsize=(14, 6))
    plt.plot(train_data.index, S_obs, label='Observed Shoreline (1999-2021)', color='blue')
    plt.plot(train_data.index, S_sim_train, label='Modeled Shoreline (1999-2021, with trend)', color='red', linestyle='--')
    plt.plot(predict_data.index, S_predict, label='Predicted Shoreline (2022 onwards, new trend)', color='red', linestyle='--')
    plt.plot(predict_data.index, y_test, label='Observed Shoreline (2022 onwards)', color='green', linestyle=':')
    
    plt.text(
        0.05, 0.95,
        f'Training (1999-2021):\nCC = {metrics_train["cc"]:.3f}\nNorm RMSE = {metrics_train["norm_rmse"]:.3f}\nNorm STD = {metrics_train["norm_std"]:.3f}\n\n'
        f'Testing (2022 onwards):\nCC = {metrics_test["cc"]:.3f}\nNorm RMSE = {metrics_test["norm_rmse"]:.3f}\nNorm STD = {metrics_test["norm_std"]:.3f}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    
    plt.xlabel('Date')
    plt.ylabel('Shoreline Position')
    plt.title('Shoreline Position: Observed, Modeled, and Predicted (New Forecast Trend)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.show()

def main():
    file_path = 'C:/Users/amgh628/Downloads/XGBoost/Wave4/Yates/Muriwai/Muriwai_smoothed.csv'
    
    data = load_and_preprocess_data(file_path)
    train_data, predict_data, S_obs, S_obs_detrended, trend, trend_coeffs = split_and_prepare_data(data)
    
    param_bounds = [(-2, 1), (0, 6), (0, 5), (0, 5)]
    initial_params = [-1.5, 3, 2.5, 2.5]
    
    result = optimize_parameters(initial_params, param_bounds, S_obs_detrended[0], train_data['wave_energy'].values, trend, S_obs)
    
    if result.success:
        logging.info("Optimization successful!")
    else:
        logging.warning(f"Optimization failed: {result.message}")
    
    a_opt, b_opt, C_plus_opt, C_minus_opt = result.x
    logging.info(f"Optimized parameters: a={a_opt:.4f}, b={b_opt:.2f}, C_plus={C_plus_opt:.4f}, C_minus={C_minus_opt:.4f}")
    
    model = EquilibriumShorelineModel(a_opt, b_opt, C_plus_opt, C_minus_opt)
    S_sim_train_detrended = model.simulate(S_obs_detrended[0], train_data['wave_energy'].values)
    S_sim_train = S_sim_train_detrended + trend
    
    E_predict = predict_data['wave_energy'].values
    S_predict_detrended = model.simulate(S_obs_detrended[-1], E_predict)
    
    # Estimate new trend for forecast period
    time_index_predict = np.arange(len(predict_data))
    if len(predict_data['shore'].values) >= 10:
        trend_coeffs_predict = np.polyfit(time_index_predict, predict_data['shore'].values, 1)
        trend_predict = np.polyval(trend_coeffs_predict, time_index_predict)
    else:
        trend_predict = np.zeros(len(predict_data))
        logging.warning("Insufficient data for forecast trend; using detrended predictions.")
    S_predict = S_predict_detrended + trend_predict
    
    train_data['modeled_shoreline'] = S_sim_train
    predict_data['predicted_shoreline'] = S_predict
    combined_data = pd.concat([train_data, predict_data])
    
    output_path = 'C:/Users/amgh628/Downloads/XGBoost/Wave4/Yates/Muriwai/Muriwai_pred2.csv'
    try:
        combined_data.to_csv(output_path)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
    
    metrics_train = calculate_metrics(S_obs, S_sim_train)
    metrics_test = calculate_metrics(predict_data['shore'].values, S_predict)
    
    plot_results(train_data, predict_data, S_obs, S_sim_train, S_predict, predict_data['shore'].values, metrics_train, metrics_test)

if __name__ == "__main__":
    main()
