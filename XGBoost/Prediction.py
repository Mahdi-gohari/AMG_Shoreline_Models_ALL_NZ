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

# Suppress Optuna's trial-by-trial output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def transform_direction(theta):
    """Transforms wave direction (dpm) into a value between 0 and 1."""
    theta = np.array(theta) % 360
    psi = np.where(theta <= 180, 1 - theta / 180, (theta - 180) / 180)
    return psi

def process_and_extract_features(ds, shore_df):
    """Processes all grid points, variables, and merges with shore data."""
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
    
    # Check for empty features_df
    if features_df.empty or features_df.shape[1] == 0:
        raise ValueError("No valid features extracted from NetCDF file.")
    
    # Drop columns with >50% NaNs
    nan_threshold = 0.5 * len(features_df)
    features_df = features_df.dropna(axis=1, thresh=nan_threshold)
    
    # Debug: Print feature count and NaN prevalence
    print(f"Extracted {features_df.shape[1]} features from NetCDF")
    print(f"NaN proportion per feature:\n{features_df.isna().mean()}")
    print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    
    # Check if any columns remain
    if features_df.empty or features_df.shape[1] == 0:
        raise ValueError("All feature columns dropped due to excessive NaNs.")
    
    # Merge with shore_df
    merged_df = features_df.join(shore_df['smoothed_shoreline'], how='inner')
    
    # Check for empty merged_df
    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty after joining with shore_df. Check date alignment or NaN values.")
    
    # Fill NaNs for feature columns only (exclude 'smoothed_shoreline')
    feature_cols = [col for col in merged_df.columns if col != 'smoothed_shoreline']
    merged_df[feature_cols] = merged_df[feature_cols].ffill().bfill()
    
    return merged_df

def select_top_features_with_xgboost(X_train, y_train, exact_importance_threshold=0.005, max_features=7):
    """Selects top features using XGBoost feature importance and correlation analysis."""
    print("\nStarting automated feature selection process...")
    
    # Check for empty dataset
    if X_train.empty or y_train.empty:
        print("Error: Empty training dataset provided to feature selection.")
        return []
    
    temp_model = XGBRegressor(n_estimators=1000, random_state=42)
    temp_model.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': temp_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Print all feature importances for debugging
    print("\nAll feature importances:")
    print(importance_df)
    
    selected_features_by_importance = importance_df[importance_df['Importance'] > exact_importance_threshold]['Feature'].tolist()
    
    print(f"\nPhase 1: Features with importance > {exact_importance_threshold}:")
    print(selected_features_by_importance)

    print("\nPhase 2: Removing highly correlated features...")
    if not selected_features_by_importance:
        print("No features selected from Phase 1. Skipping correlation check.")
        return []

    corr_matrix = X_train[selected_features_by_importance].corr().abs()
    # Print correlation matrix for debugging
    print("\nCorrelation matrix of selected features:")
    print(corr_matrix)
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    selected_features_after_correlation = [f for f in selected_features_by_importance if f not in to_drop]

    print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    print(f"Selected {len(selected_features_after_correlation)} features after correlation check:\n{selected_features_after_correlation}")
    
    print("\nPhase 3: Limiting to a maximum of {} features...".format(max_features))
    if len(selected_features_after_correlation) > max_features:
        # Prioritize raw and seasonal features
        raw_features = [f for f in selected_features_after_correlation if not ('rolling_avg_30d_' in f or 'ewm_avg_30d_' in f or 'rolling_std_30d_' in f or 'rolling_max_30d_' in f)]
        rolling_features = [f for f in selected_features_after_correlation if ('rolling_avg_30d_' in f or 'ewm_avg_30d_' in f or 'rolling_std_30d_' in f or 'rolling_max_30d_' in f)]
        selected_features_final = (raw_features + rolling_features)[:max_features]
        print(f"Prioritized raw features, reduced to top {max_features} features.")
    else:
        selected_features_final = selected_features_after_correlation
    
    print(f"Final selected {len(selected_features_final)} features:\n{selected_features_final}")
    
    return selected_features_final

def create_new_features(df, include_rolling_features=True):
    """Creates additional features including seasonal and optional rolling features."""
    if df.empty:
        raise ValueError("Input DataFrame is empty in create_new_features.")
    
    df = df.sort_index()
    df['month'] = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['month'])
    
    if include_rolling_features:
        netcdf_features = [col for col in df.columns if col != 'smoothed_shoreline' and not col.startswith('sin_') and not col.startswith('cos_')]
        print(f"\nAdding rolling features for {len(netcdf_features)} features...")
        new_columns = {}
        for feature in netcdf_features:
            new_columns[f'rolling_avg_30d_{feature}'] = df[feature].rolling(window=30, min_periods=1).mean()
            new_columns[f'ewm_avg_30d_{feature}'] = df[feature].ewm(span=30, min_periods=1).mean()
            new_columns[f'rolling_std_30d_{feature}'] = df[feature].rolling(window=30, min_periods=1).std()
            new_columns[f'rolling_max_30d_{feature}'] = df[feature].rolling(window=30, min_periods=1).max()
        
        # Concatenate new columns to the DataFrame at once
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    # Fill NaNs in new columns
    feature_cols = [col for col in df.columns if col != 'smoothed_shoreline']
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    return df

class FixedRollingWindowSplit:
    """Custom class for rolling window cross-validation."""
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

def objective_cv(trial, X_train, y_train):
    """Optuna objective function for hyperparameter optimization with cross-validation."""
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
        step_size_samples=1 * 365
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

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics (correlation, normalized RMSE, normalized STD)."""
    y_true = y_true.dropna()
    y_pred = pd.Series(y_pred, index=y_true.index)
    if len(y_true) == 0:
        return {'cc': np.nan, 'norm_rmse': np.nan, 'norm_std': np.nan}
    cc, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    norm_rmse = rmse / np.std(y_true) if np.std(y_true) != 0 else np.nan
    norm_std = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
    return {'cc': cc, 'norm_rmse': norm_rmse, 'norm_std': norm_std}

def train_and_evaluate(nc_file, csv_file, output_csv, output_plot, max_features=7, selected_features=None, include_rolling_features=True):
    """Main function to process data, train model, and generate plots."""
    # Load datasets
    ds = xr.open_dataset(nc_file)
    shore_df = pd.read_csv(csv_file, dayfirst=True, parse_dates=['t', 'dates'], index_col='dates')

    # Process and extract features
    all_features_df = process_and_extract_features(ds, shore_df)

    # Split data
    train_df = all_features_df['1999':'2020']
    test_df = all_features_df['2021':'2025']

    # Check for empty splits
    if train_df.empty:
        raise ValueError("Training DataFrame (1999–2020) is empty. Check date range or NaN handling.")
    if test_df.empty:
        print("Warning: Testing DataFrame (2021–2025) is empty. Predictions may be limited.")

    # Apply feature creation
    train_df = create_new_features(train_df, include_rolling_features=include_rolling_features)
    test_df = create_new_features(test_df, include_rolling_features=include_rolling_features)

    # Prepare training and testing data
    main_features = [col for col in train_df.columns if col != 'smoothed_shoreline']
    # Drop rows with NaN in 'smoothed_shoreline' only, preserve feature NaNs
    train_df = train_df.dropna(subset=['smoothed_shoreline'])
    test_df = test_df.dropna(subset=['smoothed_shoreline']) if 'smoothed_shoreline' in test_df.columns else test_df

    # Fill NaNs in feature columns
    train_df[main_features] = train_df[main_features].ffill().bfill()
    test_df[main_features] = test_df[main_features].ffill().bfill() if not test_df.empty else test_df

    # Debug: Check dataset sizes and NaNs
    print(f"X_train_all shape: {train_df[main_features].shape}")
    print(f"y_train_all shape: {train_df['smoothed_shoreline'].shape}")
    print(f"NaN count in X_train_all:\n{train_df[main_features].isna().sum()}")
    print(f"NaN count in y_train_all: {train_df['smoothed_shoreline'].isna().sum()}")

    # Check for empty datasets after processing
    if train_df.empty:
        raise ValueError("Training DataFrame is empty after processing.")
    if test_df.empty:
        print("Warning: Testing DataFrame is empty after processing. Skipping testing evaluation.")

    X_train_all = train_df[main_features]
    y_train_all = train_df['smoothed_shoreline']
    X_test_all = test_df[main_features]
    y_test_all = test_df['smoothed_shoreline'] if 'smoothed_shoreline' in test_df.columns else pd.Series(np.nan, index=X_test_all.index)

    # Feature selection
    if selected_features is None:
        print("No pre-selected features provided. Running automated feature selection...")
        selected_features = select_top_features_with_xgboost(X_train_all, y_train_all, exact_importance_threshold=0.005, max_features=max_features)
    else:
        print(f"Using pre-selected features: {selected_features}")
        # Validate that selected_features exist in X_train_all
        invalid_features = [f for f in selected_features if f not in X_train_all.columns]
        if invalid_features:
            raise ValueError(f"Invalid features provided: {invalid_features}. Available features: {X_train_all.columns.tolist()}")

    if not selected_features:
        print("Warning: No features were selected. Cannot proceed with model training.")
        return

    X_train = X_train_all[selected_features]
    y_train = y_train_all
    X_test = X_test_all[selected_features]
    y_test = y_test_all

    # Check for empty training data
    if X_train.empty or y_train.empty:
        raise ValueError("Training dataset is empty after feature selection.")

    # Hyperparameter optimization
    print("Starting Optuna hyperparameter optimization with cross-validation...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective_cv(trial, X_train, y_train), n_trials=100)
    
    best_params = study.best_params
    print("\nBest Hyperparameters found:", best_params)
    print(f"Best cross-validation RMSE: {study.best_value:.4f}")

    # Train final model
    print("\n--- Training final model with best hyperparameters ---")
    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train, verbose=False)

    # Evaluate model
    y_train_pred = final_model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    print(f"\nTraining RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}, Training Correlation: {train_metrics['cc']:.4f}")

    y_test_pred = final_model.predict(X_test) if not X_test.empty else np.array([])
    test_metrics = calculate_metrics(y_test, y_test_pred) if not X_test.empty else {'cc': np.nan, 'norm_rmse': np.nan, 'norm_std': np.nan}
    print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}, Testing Correlation: {test_metrics['cc']:.4f}" if not X_test.empty else "Testing skipped: Empty test dataset.")

    # Feature importance plot
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

    # Save predictions
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
    }) if not X_test.empty else pd.DataFrame()
    combined_df = pd.concat([training_predictions_df, test_predictions_df], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)

    # Create and save metrics to a separate CSV
    metrics_df = pd.DataFrame({
        'Period': ['Training', 'Future'],
        'CC_train': [train_metrics['cc'], np.nan],
        'Norm_RMSE_train': [train_metrics['norm_rmse'], np.nan],
        'Norm_STD_train': [train_metrics['norm_std'], np.nan],
        'Loss_train': [np.sqrt((0 - train_metrics['norm_rmse']) ** 2 + (1 - train_metrics['cc']) ** 2 + (1 - train_metrics['norm_std']) ** 2) if not np.isnan(train_metrics['norm_rmse']) else np.nan, np.nan],
        'CC_test': [np.nan, test_metrics['cc']],
        'Norm_RMSE_test': [np.nan, test_metrics['norm_rmse']],
        'Norm_STD_test': [np.nan, test_metrics['norm_std']],
        'Loss_test': [np.nan, np.sqrt((0 - test_metrics['norm_rmse']) ** 2 + (1 - test_metrics['cc']) ** 2 + (1 - test_metrics['norm_std']) ** 2) if not np.isnan(test_metrics['norm_rmse']) else np.nan]
    })
    metrics_csv = output_csv.replace('.csv', '_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Predictions saved to '{output_csv}'.")
    print(f"Metrics saved to '{metrics_csv}'.")

    # Plotting
    data = pd.read_csv(output_csv)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=False, format='%Y-%m-%d')

    original_data = pd.read_csv(csv_file, parse_dates=['t', 'dates'], dayfirst=True)

    filtered_data = data[data['Date'].isin(original_data['t'])].copy()
    filtered_data = filtered_data.dropna(subset=['Actual_Shoreline', 'Predicted_Shoreline'], how='any')

    train_data = filtered_data[filtered_data['Date'] <= '2020-12-31']
    test_data = filtered_data[filtered_data['Date'] >= '2021-01-01']

    obs_train = train_data['Actual_Shoreline'].values
    yates_train = train_data['Predicted_Shoreline'].values
    obs_test = test_data['Actual_Shoreline'].values
    yates_test = test_data['Predicted_Shoreline'].values

    if len(obs_train) > 0 and len(yates_train) > 0:
        cc_train = np.corrcoef(obs_train, yates_train)[0, 1]
        rmse_train = np.sqrt(mean_squared_error(obs_train, yates_train))
        std_train_obs = np.std(obs_train)
        std_train_yates = np.std(yates_train)
        norm_rmse_train = rmse_train / std_train_obs if std_train_obs != 0 else np.nan
        norm_std_train = std_train_yates / std_train_obs if std_train_obs != 0 else np.nan
        loss_train = np.sqrt((0 - norm_rmse_train) ** 2 + (1 - cc_train) ** 2 + (1 - norm_std_train) ** 2)
    else:
        cc_train = norm_rmse_train = norm_std_train = loss_train = np.nan
        print("Warning: No valid training data for metrics calculation.")

    if len(obs_test) > 0 and len(yates_test) > 0:
        cc_test = np.corrcoef(obs_test, yates_test)[0, 1]
        rmse_test = np.sqrt(mean_squared_error(obs_test, yates_test))
        std_test_obs = np.std(obs_test)
        std_test_yates = np.std(yates_test)
        norm_rmse_test = rmse_test / std_test_obs if std_test_obs != 0 else np.nan
        norm_std_test = std_test_yates / std_test_obs if std_test_obs != 0 else np.nan
        loss_test = np.sqrt((0 - norm_rmse_test) ** 2 + (1 - cc_test) ** 2 + (1 - norm_std_test) ** 2)
    else:
        cc_test = norm_rmse_test = norm_std_test = loss_test = np.nan
        print("Warning: No valid testing data for metrics calculation.")

    plt.figure(figsize=(14, 6))
    plt.plot(filtered_data['Date'], filtered_data['Actual_Shoreline'],
             label='Observation', color='blue', linewidth=1)
    plt.plot(filtered_data['Date'], filtered_data['Predicted_Shoreline'],
             label='XGBoost', color='red', linewidth=1.5, alpha=0.7)
    plt.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2025-12-31'),
                color='gray', alpha=0.3, label='Prediction Period (2021–2025)')
    plt.text(
        0.01, 0.95,
        f'Training (1999–2020):\nCC = {cc_train:.3f}\nNorm RMSE = {norm_rmse_train:.3f}\nNorm STD = {norm_std_train:.3f}\nLoss = {loss_train:.3f}\n\n'
        f'Future (2021–2025):\nCC = {cc_test:.3f}\nNorm RMSE = {norm_rmse_test:.3f}\nNorm STD = {norm_std_test:.3f}\nLoss = {loss_test:.3f}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='black')
    )
    plt.xlabel('Date')
    plt.ylabel('Shoreline Position')
    plt.title('Shoreline Position: Observed vs. Predicted (XGBoost)')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.show()
