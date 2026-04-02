"""
Hypothesis Verification Script
===============================
Tests 3 hypotheses about why Run 1 (global scaling, LOOKBACK=60, LSTM)
outperformed Run 3 (per-company scaling, LOOKBACK=20, Bidirectional LSTM).

Each experiment changes ONE variable at a time while keeping others fixed.
Results are printed side-by-side for comparison.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.callbacks import EarlyStopping

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

FEATURE_COLS = ['Open', 'High', 'Low', 'Volume']
TARGET_COL = ['Close']


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_stock_data.csv'), parse_dates=['Date'])
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test_stock_data.csv'), parse_dates=['Date'])
    # Drop unnamed index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    return train_df, test_df


# ============================================================
# SCALING FUNCTIONS
# ============================================================
def scale_global(train_df, test_df, feature_cols, target_col):
    """Single scaler for all companies."""
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    train_features = feature_scaler.fit_transform(train_df[feature_cols])
    test_features  = feature_scaler.transform(test_df[feature_cols])
    train_target   = target_scaler.fit_transform(train_df[target_col])
    test_target    = target_scaler.transform(test_df[target_col])

    return train_features, test_features, train_target, test_target, target_scaler


def scale_per_company(train_df, test_df, feature_cols, target_col):
    """Per-company scaler for both features and target."""
    feature_scalers = {}
    target_scalers  = {}

    train_features = np.zeros((len(train_df), len(feature_cols)))
    test_features  = np.zeros((len(test_df),  len(feature_cols)))
    train_target   = np.zeros((len(train_df), 1))
    test_target    = np.zeros((len(test_df),  1))

    for ticker in train_df['Ticker'].unique():
        f_scaler = RobustScaler()
        t_scaler = RobustScaler()
        tr_mask = train_df['Ticker'] == ticker
        te_mask = test_df['Ticker'] == ticker

        train_features[tr_mask] = f_scaler.fit_transform(train_df.loc[tr_mask, feature_cols])
        test_features[te_mask]  = f_scaler.transform(test_df.loc[te_mask, feature_cols])
        train_target[tr_mask]   = t_scaler.fit_transform(train_df.loc[tr_mask, target_col])
        test_target[te_mask]    = t_scaler.transform(test_df.loc[te_mask, target_col])

        feature_scalers[ticker] = f_scaler
        target_scalers[ticker]  = t_scaler

    return train_features, test_features, train_target, test_target, target_scalers


# ============================================================
# SEQUENCE GENERATION
# ============================================================
def create_sequences_per_company(df, features, target, lookback):
    X, y = [], []
    for _, group_idx in df.groupby('Ticker').groups.items():
        idx = group_idx.values
        for i in range(lookback, len(idx)):
            X.append(features[idx[i - lookback:i]])
            y.append(target[idx[i], 0])
    return np.array(X), np.array(y)


def prepare_test_sequences(train_df, test_df, train_features, test_features,
                           train_target, test_target, lookback):
    test_dfs_padded, test_features_list, test_target_list = [], [], []
    for ticker, group_idx in train_df.groupby('Ticker').groups.items():
        idx = group_idx.values
        test_mask = test_df['Ticker'] == ticker
        n_test = test_mask.sum()
        if n_test == 0:
            continue
        test_features_list.append(train_features[idx[-lookback:]])
        test_features_list.append(test_features[test_df.index[test_mask].values])
        test_target_list.append(train_target[idx[-lookback:]])
        test_target_list.append(test_target[test_df.index[test_mask].values])
        test_dfs_padded.append(pd.concat([
            train_df.iloc[idx[-lookback:]],
            test_df.loc[test_mask]
        ], ignore_index=True))

    test_df_padded = pd.concat(test_dfs_padded, ignore_index=True)
    test_features_padded = np.vstack(test_features_list)
    test_target_padded = np.vstack(test_target_list)
    return test_df_padded, test_features_padded, test_target_padded


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_lstm(lookback, n_features):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_bidirectional(lookback, n_features):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=False)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ============================================================
# EVALUATION
# ============================================================
def inverse_transform_global(pred_scaled, actual_scaled, target_scaler):
    pred   = target_scaler.inverse_transform(pred_scaled)
    actual = target_scaler.inverse_transform(actual_scaled.reshape(-1, 1))
    return pred, actual


def inverse_transform_per_company(pred_scaled, actual_scaled, target_scalers,
                                   test_company_labels):
    pred   = np.zeros_like(pred_scaled)
    actual = np.zeros((len(actual_scaled), 1))
    for ticker in np.unique(test_company_labels):
        mask = test_company_labels == ticker
        pred[mask]   = target_scalers[ticker].inverse_transform(pred_scaled[mask])
        actual[mask] = target_scalers[ticker].inverse_transform(
            actual_scaled[mask].reshape(-1, 1))
    return pred, actual


def compute_metrics(y_actual, y_pred):
    mse  = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def get_company_labels(test_df_padded, lookback):
    labels = []
    for ticker, group_idx in test_df_padded.groupby('Ticker').groups.items():
        idx = group_idx.values
        for i in range(lookback, len(idx)):
            labels.append(ticker)
    return np.array(labels)


# ============================================================
# SINGLE EXPERIMENT
# ============================================================
def run_experiment(name, train_df, test_df, lookback, scaling_method, model_type):
    print(f'\n{"="*60}')
    print(f'  Experiment: {name}')
    print(f'  LOOKBACK={lookback}  Scaling={scaling_method}  Model={model_type}')
    print(f'{"="*60}')

    # 1. Scale
    if scaling_method == 'global':
        train_feat, test_feat, train_tgt, test_tgt, scaler_or_dict = \
            scale_global(train_df, test_df, FEATURE_COLS, TARGET_COL)
    else:
        train_feat, test_feat, train_tgt, test_tgt, scaler_or_dict = \
            scale_per_company(train_df, test_df, FEATURE_COLS, TARGET_COL)

    # 2. Generate sequences
    X_train, y_train = create_sequences_per_company(train_df, train_feat, train_tgt, lookback)
    test_df_padded, test_feat_padded, test_tgt_padded = prepare_test_sequences(
        train_df, test_df, train_feat, test_feat, train_tgt, test_tgt, lookback)
    X_test, y_test = create_sequences_per_company(
        test_df_padded, test_feat_padded, test_tgt_padded, lookback)

    print(f'  X_train: {X_train.shape}  X_test: {X_test.shape}')

    # 3. Build model
    if model_type == 'lstm':
        model = build_lstm(lookback, X_train.shape[2])
    else:
        model = build_bidirectional(lookback, X_train.shape[2])

    # 4. Train
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                               restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f'  Best epoch: {best_epoch}/{len(history.history["loss"])}')

    # 5. Predict
    pred_scaled = model.predict(X_test, verbose=0)

    # 6. Inverse transform
    if scaling_method == 'global':
        prediction, y_actual = inverse_transform_global(pred_scaled, y_test, scaler_or_dict)
    else:
        company_labels = get_company_labels(test_df_padded, lookback)
        prediction, y_actual = inverse_transform_per_company(
            pred_scaled, y_test, scaler_or_dict, company_labels)

    # 7. Metrics
    metrics = compute_metrics(y_actual, prediction)
    print(f'  MAPE: {metrics["MAPE"]:.2f}%   MAE: {metrics["MAE"]:.4f}   '
          f'RMSE: {metrics["RMSE"]:.4f}')
    print()

    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print('Loading data...')
    train_df, test_df = load_data()
    print(f'Train: {train_df.shape}  Test: {test_df.shape}')
    print(f'Tickers: {train_df["Ticker"].nunique()}')

    results = {}

    # --------------------------------------------------------
    # HYPOTHESIS 1: LOOKBACK matters
    #   A: LOOKBACK=60, global, lstm        (Run 1 baseline)
    #   B: LOOKBACK=20, global, lstm        (change only LOOKBACK)
    # --------------------------------------------------------
    results['A_Run1_Baseline'] = run_experiment(
        'A: Run 1 Baseline (LOOKBACK=60, Global, LSTM)',
        train_df, test_df,
        lookback=60, scaling_method='global', model_type='lstm'
    )

    results['B_LOOKBACK20'] = run_experiment(
        'B: LOOKBACK=20 (only change: 60→20)',
        train_df, test_df,
        lookback=20, scaling_method='global', model_type='lstm'
    )

    # --------------------------------------------------------
    # HYPOTHESIS 2: Scaling method matters
    #   A: LOOKBACK=60, global, lstm        (same as above)
    #   C: LOOKBACK=60, per_company, lstm   (change only scaling)
    # --------------------------------------------------------
    results['C_PerCompany'] = run_experiment(
        'C: Per-Company Scaling (only change: global→per_company)',
        train_df, test_df,
        lookback=60, scaling_method='per_company', model_type='lstm'
    )

    # --------------------------------------------------------
    # HYPOTHESIS 3: Architecture matters
    #   A: LOOKBACK=60, global, lstm        (same as above)
    #   D: LOOKBACK=60, global, bidirectional (change only model)
    # --------------------------------------------------------
    results['D_Bidirectional'] = run_experiment(
        'D: Bidirectional LSTM (only change: lstm→bidirectional)',
        train_df, test_df,
        lookback=60, scaling_method='global', model_type='bidirectional'
    )

    # --------------------------------------------------------
    # Run 3 replica for reference
    #   E: LOOKBACK=20, per_company, bidirectional (Run 3)
    # --------------------------------------------------------
    results['E_Run3'] = run_experiment(
        'E: Run 3 Replica (LOOKBACK=20, Per-Company, Bidirectional)',
        train_df, test_df,
        lookback=20, scaling_method='per_company', model_type='bidirectional'
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print('\n' + '=' * 70)
    print('  SUMMARY — Which hypothesis is correct?')
    print('=' * 70)
    print(f'{"Experiment":<50} {"MAPE":>8} {"MAE":>10} {"RMSE":>10}')
    print('-' * 70)

    labels = {
        'A_Run1_Baseline':   'A: Baseline (60, global, lstm)',
        'B_LOOKBACK20':      'B: LOOKBACK 60→20',
        'C_PerCompany':      'C: Scaling global→per_company',
        'D_Bidirectional':   'D: Model lstm→bidirectional',
        'E_Run3':            'E: Run 3 (20, per_company, bidi)',
    }

    for key, label in labels.items():
        m = results[key]
        print(f'{label:<50} {m["MAPE"]:>7.2f}% {m["MAE"]:>10.4f} {m["RMSE"]:>10.4f}')

    print('-' * 70)
    print()

    # Verdict
    baseline_mape = results['A_Run1_Baseline']['MAPE']

    print('Verdict:')
    for key, label in [('B_LOOKBACK20', 'LOOKBACK (60→20)'),
                        ('C_PerCompany', 'Scaling (global→per_company)'),
                        ('D_Bidirectional', 'Architecture (lstm→bidirectional)')]:
        diff = results[key]['MAPE'] - baseline_mape
        worse = 'WORSE' if diff > 0 else 'BETTER'
        print(f'  {label}: {diff:+.2f}% MAPE change → {worse}')

    print()
    biggest_culprit = max(
        ['B_LOOKBACK20', 'C_PerCompany', 'D_Bidirectional'],
        key=lambda k: results[k]['MAPE']
    )
    print(f'Biggest culprit: {labels[biggest_culprit]} '
          f'({results[biggest_culprit]["MAPE"]:.2f}% MAPE)')


if __name__ == '__main__':
    main()
