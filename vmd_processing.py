import pandas as pd
import numpy as np
from vmdpy import VMD
from scipy.stats import pearsonr

def apply_vmd(signal, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """Apply VMD to a time series."""
    try:
        modes, modes_freqs, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        return modes, modes_freqs
    except Exception as e:
        print(f"VMD failed: {e}")
        return None, None

def compute_imf_similarity(modes1, modes2):
    """Calculate average correlation between corresponding IMFs."""
    similarities = []
    for k in range(len(modes1)):
        corr, _ = pearsonr(modes1[k], modes2[k])
        similarities.append(corr if not np.isnan(corr) else 0)
    return np.mean(similarities)

def impute_missing_values(df, df_filled, modes_all, target_transect, transects):
    """Impute missing values in target transect using IMFs from similar transect."""
    missing_indices = df[target_transect].isna()
    if not missing_indices.any():
        return df[target_transect]
    similarities = {}
    target_modes = modes_all[target_transect]
    for transect in transects:
        if transect != target_transect:
            sim_score = compute_imf_similarity(target_modes, modes_all[transect])
            similarities[transect] = sim_score
    similar_transect = max(similarities, key=similarities.get)
    similar_modes = modes_all[similar_transect]
    imputed_signal = df[target_transect].copy()
    
    # Ensure idx_num is within bounds
    max_idx = similar_modes.shape[1] - 1
    for idx in missing_indices[missing_indices].index:
        idx_num = df.index.get_loc(idx)
        if idx_num > max_idx:
            imputed_signal.loc[idx] = np.sum(similar_modes[:, max_idx])
        else:
            imputed_signal.loc[idx] = np.sum(similar_modes[:, idx_num])
    return imputed_signal

def process_shoreline_data(df, df_filled, K=5, alpha=2000, corr_remove_pct=0.1):
    """Process shoreline data with VMD: impute missing values and return DataFrame."""
    # Select transect columns (exclude dates and satname)
    transects = [col for col in df.columns if col not in ['dates', 'satname']]
    original_num = len(transects)

    # Remove first three and last three transects
    transects_sorted = sorted(transects)
    transects_to_keep = transects_sorted[1:-1]
    df = df[['dates'] + transects_to_keep]
    df_filled = df_filled[['dates'] + transects_to_keep]
    transects = transects_to_keep

    # Compute correlation matrix and remove top corr_remove_pct lowest CC transects
    corr_matrix = df_filled[transects].corr()
    avg_corrs = {}
    for t in transects:
        avg_corrs[t] = (corr_matrix[t].sum() - 1) / (len(transects) - 1)
    sorted_transects = sorted(avg_corrs, key=avg_corrs.get)
    num_to_remove = round(corr_remove_pct * len(transects))
    low_cc = sorted_transects[:num_to_remove]
    df = df.drop(columns=low_cc)
    df_filled = df_filled.drop(columns=low_cc)
    transects = [t for t in transects if t not in low_cc]

    removed = original_num - len(transects)
    print(f"Originally {original_num} transects existed, {removed} have been removed.")

    # Apply VMD to one transect to get expected mode length
    sample_transect = transects[0]
    sample_signal = df_filled[sample_transect].values
    sample_modes, _ = apply_vmd(sample_signal, K, alpha)
    if sample_modes is not None:
        modes_length = sample_modes.shape[1]
        if df.shape[0] > modes_length:
            df = df.iloc[:modes_length].copy()
            df_filled = df_filled.iloc[:modes_length].copy()
    
    # Apply VMD to all transects
    modes_all = {}
    freqs_all = {}
    for transect in transects:
        signal = df_filled[transect].values
        modes, modes_freqs = apply_vmd(signal, K, alpha)
        if modes is not None:
            modes_all[transect] = modes
            freqs_all[transect] = modes_freqs
        else:
            print(f"VMD failed for transect {transect}, skipping.")

    # Impute missing values for each transect
    df_imputed = df.copy()
    for transect in transects:
        df_imputed[transect] = impute_missing_values(df, df_filled, modes_all, transect, transects)

    # Compute average across transects
    transect_cols = [col for col in df_imputed.columns if col not in ['dates', 'satname']]
    shoreline = df_imputed[transect_cols].mean(axis=1)
    avg_df = pd.DataFrame({'dates': df_imputed['dates'], 'shoreline': shoreline})

    # Apply requested interpolation format
    avg_df['dates'] = pd.to_datetime(avg_df['dates'], dayfirst=True, errors='coerce')
    avg_df = avg_df.set_index('dates').resample('D').mean().interpolate(method='linear').reset_index()
    avg_df['dates'] = avg_df['dates'].apply(lambda x: pd.Timestamp(x).date())
    
    print("VMD processing completed, returning averaged and interpolated shoreline data.")
    return avg_df, modes_all