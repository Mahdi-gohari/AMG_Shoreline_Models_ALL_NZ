import pandas as pd
import numpy as np
import xarray as xr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import optuna
from optuna.samplers import TPESampler

# Suppress Optuna's trial-by-trial output, showing only warnings and errors
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load the NetCDF dataset
ds = xr.open_dataset("C:/Users/amgh628/Downloads/XGBoost/Wave4/After_pyr/Okari/Okari.nc")

# Load the 'shore' data from the original CSV
shore_df = pd.read_csv("C:/Users/amgh628/Downloads/XGBoost/Wave4/XGboost/Okari1/Okari_shore_wave.csv",
                       dayfirst=True, parse_dates=['t'], index_col='t')

# --- NEW FUNCTIONS FOR AUTOMATED FEATURE GENERATION AND PROCESSING ---
def transform_direction(theta):
    """Transforms wave direction (dpm) into a value between 0 and 1."""
    theta = np.array(theta) % 360
    psi = np.where(theta <= 180, 1 - theta / 180, (theta - 180) / 180)
    return psi

def process_and_extract_features(ds, shore_df):
    """
    Processes all grid points, variables, and merges with shore data.
    """
    print("Processing and extracting all features from the NetCDF file...")
    
    if 'time_bnds' in ds.keys():
        ds = ds.drop_vars('time_bnds')
    
    wave_data_daily = ds.resample(time='1D').mean()
    wave_data_daily['dpm'] = xr.apply_ufunc(transform_direction, wave_data_daily['dpm'], vectorize=True)
    wave_data_daily['Eng'] = (wave_data_daily['hs'] ** 2) * wave_data_daily['ptp1'] * 0.49
    
    features_to_extract = ["hs", "ptp1", "dpm", "xwnd", "ywnd", "Eng", "tps", "wspd"]
    latitudes = wave_data_daily['latitude'].values
    longitudes = wave_data_daily['longitude'].values
    
    df_dict = {}
    for feature in features_to_extract:
        if feature in wave_data_daily.data_vars:
            for lat in latitudes:
                for lon in longitudes:
                    try:
                        data = wave_data_daily[feature].sel(latitude=lat, longitude=lon, method='nearest').to_dataframe()
                        col_name = f"{feature}_latlon_{lat:.2f}_{lon:.2f}"
                        df_dict[col_name] = data[feature]
                    except KeyError:
                        print(f"Warning: Could not extract '{feature}' at ({lat}, {lon}). Skipping.")
        else:
            print(f"Warning: Feature '{feature}' not found in the dataset.")
            
    features_df = pd.DataFrame(df_dict)
    features_df.index = pd.to_datetime(features_df.index)
    
    merged_df = features_df.join(shore_df['shore'], how='inner')
    
    return merged_df

def select_top_features_with_xgboost(X_train, y_train, exact_importance_threshold=0.01, max_features=7):
    print("\nStarting automated feature selection process...")
    temp_model = XGBRegressor(n_estimators=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': temp_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    selected_features_by_importance = importance_df[importance_df['Importance'] > exact_importance_threshold]['Feature'].tolist()
    
    print(f"\nPhase 1: Features with importance > {exact_importance_threshold}:")
    print(selected_features_by_importance)

    print("\nPhase 2: Removing highly correlated features...")
    if not selected_features_by_importance:
        print("No features selected from Phase 1. Skipping correlation check.")
        return []

    corr_matrix = X_train[selected_features_by_importance].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    selected_features_after_correlation = [f for f in selected_features_by_importance if f not in to_drop]

    print(f"Removed {len(to_drop)} highly correlated features.")
    print(f"Selected {len(selected_features_after_correlation)} features after correlation check:\n{selected_features_after_correlation}")
    
    # Phase 3: Limit to max_features (7)
    print("\nPhase 3: Limiting to a maximum of 7 features...")
    if len(selected_features_after_correlation) > max_features:
        selected_features_final = selected_features_after_correlation[:max_features]
        print(f"Reduced to top {max_features} features based on importance ranking.")
    else:
        selected_features_final = selected_features_after_correlation
    
    print(f"Final selected {len(selected_features_final)} features:\n{selected_features_final}")
    
    return selected_features_final

# --- Main Script Start ---
all_features_df = process_and_extract_features(ds, shore_df)

def create_new_features(df):
    df = df.sort_index()
    df['month'] = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['month'])
    
    netcdf_features = [col for col in df.columns if col != 'shore' and not col.startswith('sin_') and not col.startswith('cos_')]
    
    print(f"\nAdding rolling averages for {len(netcdf_features)} features...")
    for feature in netcdf_features:
        df[f'rolling_avg_30d_{feature}'] = df[feature].rolling(window=30, min_periods=1).mean()
        df[f'ewm_avg_30d_{feature}'] = df[feature].ewm(span=30, min_periods=1).mean()
    
    return df

# Split data before creating new features to avoid leakage
train_df = all_features_df['1999':'2020']
test_df = all_features_df['2021':'2025']

# Apply feature creation separately to train and test sets
train_df = create_new_features(train_df)
test_df = create_new_features(test_df)

main_features = [col for col in train_df.columns if col != 'shore']
train_df.dropna(subset=main_features, inplace=True)
test_df.dropna(subset=main_features, inplace=True)

X_train_all = train_df[main_features]
y_train_all = train_df['shore']
X_test_all = test_df[main_features]
y_test_all = test_df['shore'] if 'shore' in test_df.columns else pd.Series(np.nan, index=X_test_all.index)

selected_features = select_top_features_with_xgboost(X_train_all, y_train_all, exact_importance_threshold=0.01, max_features=7)

if not selected_features:
    print("Warning: No features were selected. Cannot proceed with model training.")
else:
    X_train = X_train_all[selected_features]
    y_train = y_train_all
    X_test = X_test_all[selected_features]
    y_test = y_test_all

    # --- NEW ROLLING WINDOW CROSS-VALIDATION CLASS ---
    class FixedRollingWindowSplit:
        def __init__(self, train_size_samples, test_size_samples, step_size_samples):
            self.train_size = train_size_samples
            self.test_size = test_size_samples
            self.step_size = step_size_samples
            self.n_splits = None

        def split(self, X):
            n_samples = len(X)
            if self.train_size + self.test_size > n_samples:
                raise ValueError("Training and test size combined is larger than the dataset.")

            self.n_splits = (n_samples - self.train_size - self.test_size) // self.step_size + 1
            if self.n_splits < 1:
                raise ValueError("The specified sizes do not allow for a single fold to be created.")

            for i in range(self.n_splits):
                start = i * self.step_size
                
                if start + self.train_size + self.test_size > n_samples:
                    break
                
                train_indices = np.arange(start, start + self.train_size)
                test_indices = np.arange(start + self.train_size, start + self.train_size + self.test_size)
                
                yield (train_indices, test_indices)

        def get_n_splits(self):
            return self.n_splits
    # --- END OF NEW CLASS ---

    # --- UPDATED CODE: OPTUNA IMPLEMENTATION WITH CROSS-VALIDATION ---
    def objective_cv(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.7, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 500, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 500, log=True),
        }
        
        cv_scores = []
        rws_folding = FixedRollingWindowSplit(
            train_size_samples=10 * 365,
            test_size_samples=3 * 365,
            step_size_samples=3 * 365
        )

        for train_index, val_index in rws_folding.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            model = XGBRegressor(**params, random_state=42, early_stopping_rounds=30)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)

    print("Starting Optuna hyperparameter optimization with cross-validation...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective_cv, n_trials=100)
    
    best_params = study.best_params
    print("\nBest Hyperparameters found:", best_params)
    print(f"Best cross-validation RMSE: {study.best_value:.4f}")

    # --- Final Training with the Best Parameters ---
    print("\n--- Training final model with best hyperparameters ---")
    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train, verbose=False)

    # --- Evaluation and Visualization ---
    def calculate_metrics(y_true, y_pred):
        y_true = y_true.dropna()
        y_pred = pd.Series(y_pred, index=y_true.index)
        if len(y_true) == 0:
            return {'cc': np.nan, 'norm_rmse': np.nan, 'norm_std': np.nan}
        cc, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        norm_rmse = rmse / np.std(y_true) if np.std(y_true) != 0 else np.nan
        norm_std = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
        return {'cc': cc, 'norm_rmse': norm_rmse, 'norm_std': norm_std}

    y_train_pred = final_model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print(f"\nTraining RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, Training Correlation: {train_metrics['cc']:.4f}")

    y_test_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}, Testing Correlation: {test_metrics['cc']:.4f}")

    print("\n--- Feature Importance Analysis ---")
    feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    print(sorted_importances)

    plt.figure(figsize=(10, 6))
    sorted_importances.plot(kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    training_predictions_df = pd.DataFrame({
        'Date': X_train.index,
        'Actual_Shoreline': y_train.values,
        'Predicted_Shoreline': y_train_pred,
        'Period': 'Training'
    })
    test_predictions_df = pd.DataFrame({
        'Date': X_test.index,
        'Actual_Shoreline': y_test.values,
        'Predicted_Shoreline': y_test_pred,
        'Period': 'Future'
    })
    combined_df = pd.concat([training_predictions_df, test_predictions_df], ignore_index=True)
    combined_df.to_csv('C:/Users/amgh628/Downloads/XGBoost/Wave4/XGboost/Okari1/Okari_pediction_final_global_updated.csv', index=False)
    print("Combined predictions saved to 'Okari_prediction_final_global.csv'.")

    plt.figure(figsize=(14, 6))
    plt.plot(X_train.index, y_train, label='Observed Shoreline (Training, 1999–2020)', color='blue')
    plt.plot(X_train.index, y_train_pred, label='Modeled Shoreline (Training)', color='red', linestyle='--')
    plt.plot(X_test.index, y_test, label='Observed Shoreline (Future, 2021–2025)', color='green', linestyle=':')
    plt.plot(X_test.index, y_test_pred, label='Predicted Shoreline (Future)', color='red', linestyle='--')

    plt.text(
        0.05, 0.95,
        f'Training (1999–2020):\nCC = {train_metrics["cc"]:.3f}\nNorm RMSE = {train_metrics["norm_rmse"]:.3f}\nNorm STD = {train_metrics["norm_std"]:.3f}\n\n'
        f'Future (2021–2025):\nCC = {test_metrics["cc"]:.3f}\nNorm RMSE = {test_metrics["norm_rmse"]:.3f}\nNorm STD = {test_metrics["norm_std"]:.3f}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.xlabel('Date')
    plt.ylabel('Shoreline Position')
    plt.title('Shoreline Position: Observed, Modeled, and Predicted (1999–2025)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
