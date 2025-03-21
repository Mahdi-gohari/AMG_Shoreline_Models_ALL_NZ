import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
import shap



# Load the dataset
df = pd.read_csv('C:/Users/amgh628/Downloads/XGBoost/Muriwai/New_XGB_Garcia/Muriwai_filled_train.csv', dayfirst=True, parse_dates=['t'], index_col='t')


def weighted_moving_average(series, window):
    weights = np.arange(1, window + 1)  # Increasing weights (1, 2, ..., window)
    return series.rolling(window=window, min_periods=1).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)

"""# Function to create features
def create_features(df):
    # Adding Exponential Moving Averages
    for col in ['Eng', 'Dir', 'Tp', 'Hs']:
        df[f'{col}_ema_7'] = df[col].rolling(window=7, min_periods=1).mean()
        df[f'{col}_ema_30'] = df[col].rolling(window=30, min_periods=1).mean()

    # Seasonal and trend features
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    df['trend'] = (df.index - df.index.min()).days

    return df

"""
# Feature engineering function
def create_features(df):
    for col in ['Eng', 'Dir', 'Tp', 'Hs']:
        df[f'{col}_wma_7'] = df[col].rolling(window=7, min_periods=1).mean()
        df[f'{col}_wma_30'] = df[col].rolling(window=30, min_periods=1).mean()
    df['month'] = df.index.month
    df['trend'] = (df.index - df.index.min()).days
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    for col in ['Eng', 'Hs', 'Dir', 'Tp']:
        df[f'{col}_lag1'] = df[col].shift(1)
    #df['shore_lag1'] = df['shore'].shift(1)
    return df



# Apply feature engineering
df = create_features(df)
#df.dropna(inplace=True)

# Dropping rows with NaNs in the main features
#df.dropna(subset=['Eng', 'Dir', 'Tp', 'Hs'], inplace=True)

# Splitting the dataset
train_df = df['1999':'2021']
test_df = df['2022':'2024']

# Define all available features
features = [
    'Hs', 'Tp', 'Eng', 'Dir',
    'Eng_wma_7', 'Dir_wma_7', 'Tp_wma_7', 'Hs_wma_7',
    'Eng_wma_30', 'Dir_wma_30', 'Tp_wma_30', 'Hs_wma_30', 
    'month', 'trend', 'month_sin', 'month_cos',
    'Eng_lag1', 'Hs_lag1', 'Dir_lag1', 'Tp_lag1', 
]


# Ensure all features are available in DataFrame
missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

X_train = train_df[features]
y_train = train_df['shore']
X_test = test_df[features]


"""# Compute permutation importance
perm_importance = permutation_importance(xgb_model, X_train, y_train, scoring='neg_mean_squared_error', n_repeats=10, random_state=42)

# Convert results into a dictionary
feature_importance_dict = dict(zip(features, perm_importance.importances_mean))

# Identify top 12 most important features
top_6_features = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:12]
print("Top 12 important features based on Permutation Importance:", top_6_features)"""


# Perform feature importance with an initial XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importances
feature_importance = xgb_model.feature_importances_
feature_importance_dict = dict(zip(features, feature_importance))

# Identify the top 6 most important features
top_features = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:10]

print("Top important features:", top_features)


# Train an initial XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

"""# SHAP analysis
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_train)

# Compute mean absolute SHAP values for feature importance
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_feature_importance = dict(zip(features, shap_importance))

# Identify the top most important features
top_features = sorted(shap_feature_importance, key=shap_feature_importance.get, reverse=True)[:7]
print("Top important features:", top_features)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_train)"""



# Use the top important features for model training and testing
X_train = X_train[top_features]
X_test = X_test[top_features]

# Split train_df further into training and validation sets for early stopping
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)


# Custom loss function
def custom_loss(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_norm = rmse / np.std(y_true)
    corr, _ = pearsonr(y_true, y_pred)
    std_pred_norm = np.std(y_pred) / np.std(y_true)
    loss = np.sqrt((0 - rmse_norm)**2 + (1 - corr)**2 + (1 - std_pred_norm)**2)
    return loss



# XGBoost Regressor with hyperparameter tuning

param_grid = {
    'n_estimators': [500],
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [1, 2, 3],              # Reduced from [2, 3, 5]
    'min_child_weight': [10, 15, 20],    # Increased from [6, 8, 10]
    'subsample': [0.2, 0.3],             # Reduced from [0.3, 0.4]
    'colsample_bytree': [0.2, 0.3],      # Reduced from [0.3, 0.4]
    'reg_lambda': [10, 200, 500],      # Increased from [10, 100, 200]
    'reg_alpha': [10, 200, 500]        # Increased from [10.0, 100, 200]
}

tscv = TimeSeriesSplit(n_splits=5)
model = XGBRegressor(random_state=42, early_stopping_rounds=50)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring= 'neg_mean_squared_error',
    n_jobs=-1
)

# Fit the model with early stopping
grid_search.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Function to calculate RMSE and correlation
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlation, _ = pearsonr(y_true, y_pred)
    return rmse, correlation

# Evaluate on Training Data
y_train_pred = best_model.predict(X_train_split)
train_loss = custom_loss(y_train_split, y_train_pred)
print(f"Training Loss: {train_loss:.4f}")

# Evaluate on Validation Data
y_val_pred = best_model.predict(X_val)
val_loss = custom_loss(y_val, y_val_pred)
print(f"Validation Loss: {val_loss:.4f}")

# Make predictions
y_pred = best_model.predict(X_test)

# Save predictions to a CSV
prediction_df = pd.DataFrame({
    'Date': X_test.index,
    'Predicted_Shoreline': y_pred
})
prediction_df.to_csv('C:/Users/amgh628/Downloads/XGBoost/Muriwai/New_XGB_Garcia/Muriwai_prediction.csv', index=False)

correlation_matrix = X_train.corr()
print(correlation_matrix)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(X_test.index, y_pred, label='Predicted Shoreline Position', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Shoreline Position')
plt.title('Predicted Shoreline Position (2022-2024)')
plt.legend()
plt.tight_layout()
plt.show()