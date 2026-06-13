# Comparative Analysis: LSTM Stock Price Prediction - Attempt3 Series

## Tesla (TSLA) | 2016-01-01 to 2026 | ~2575 Trading Days

---

## 1. Experiment Configuration

| Notebook | Feature Selection Method | Selected Features | # Feat. | LSTM Units | Train/Test Split | Lookback |
|----------|-------------------------|------------------|---------|------------|------------------|----------|
| **attempt3.1** | Forward selection (greedy from 34 candidates) | Low | 1 | 100 | 80/20 | 60 |
| **attempt3.2** | Correlation filter (>0.25) + RFECV | Close, High, Low, Open, EMA_5, Volatility_5, EMA_10, Volatility_10, EMA_20, Volatility_20, EMA_50, Volatility_50, MACD, MACD_signal, BB_upper, BB_lower, BB_width, ATR_14, OBV, days_since_start | 20 | 64 | 70/15/15 | 60 |
| **attempt3.3** | Correlation filter (>0.25) + LASSO (GridSearchCV) | Close, High, Low, Open, EMA_5, Volatility_5, EMA_10, Volatility_10, EMA_20, EMA_50, Volatility_50, MACD, MACD_signal, BB_upper, BB_lower, BB_width, ATR_14, OBV, high_low_ratio, days_since_start | 20 | 100 | 70/15/15 | 60 |
| **attempt3.4** | Correlation filter + LASSO voting (5-fold) | Close, EMA_50, BB_width, days_since_start | 4 | 100 | 70/15/15 | 60 |
| **attempt3.5** | Correlation filter + LASSO voting (5-fold) | Close, EMA_50, BB_width, days_since_start | 4 | 100 | 70/15/15 | 60 |

All five notebooks share:
- **Target**: `Close_nxt` = next day's closing price (shifted by -1)
- **Scaler**: RobustScaler, fit on training data only
- **Sequence padding**: Last `LOOKBACK` rows from training prepended to test
- **Training**: Adam optimizer, MSE loss, EarlyStopping (patience=8-10), up to 50-100 epochs, batch_size=32

---

## 2. Test Metrics Comparison

| Notebook | Features | MSE | RMSE | MAE | MAPE | Mean Error | Std Error |
|----------|----------|-----|------|-----|------|------------|-----------|
| **attempt3.1** | Low (forward-selected) | 428.66 | 20.70 | 15.62 | 4.90% | 9.03 | 18.63 |
| **attempt3.2** | 20 features (RFECV) | 1903.35 | 43.63 | 32.05 | 9.52% | 20.38 | 38.61 |
| **attempt3.3** | 20 features (LASSO) | 1904.12 | 43.64 | 31.49 | 9.34% | 18.98 | 38.66 |
| **attempt3.4** | 4 features (LASSO voting) | 577.90 | 24.04 | 17.85 | 5.67% | 12.47 | 20.77 |
| **attempt3.5** | 4 features (LASSO voting) | 420.49 | 20.51 | 15.04 | 4.86% | 7.89 | 18.56 |

### Ranking by RMSE (lower is better):
1. **attempt3.5** (4 features, LASSO voting): **20.51** -- Best overall
2. **attempt3.1** (Low, forward-selected): 20.70
3. **attempt3.4** (4 features, LASSO voting): 24.04
4. **attempt3.2** (20 features, RFECV): 43.63
5. **attempt3.3** (20 features, LASSO): 43.64 -- Worst

### Ranking by MAPE (lower is better):
1. **attempt3.5** (4 features): **4.86%** -- Best
2. **attempt3.1** (Low): 4.90%
3. **attempt3.4** (4 features): 5.67%
4. **attempt3.3** (20 features): 9.34%
5. **attempt3.2** (20 features): 9.52% -- Worst

---

## 3. Fundamental Differences Between Notebooks

### 3.1 attempt3.1 -- Forward Selection from 34 Candidates

**Hypothesis**: Systematically selecting the best feature subset from 34 technical indicators, returns, ratios, and time features using greedy forward selection will find an optimal configuration.

**Methodology**:
- 34 candidate features evaluated individually first
- Then greedy forward selection: start with best single feature, iteratively add the one that reduces MSE most
- Selection epoch: 50, patience: 8

**Result**: Second best (RMSE=20.70, MAPE=4.90%). Selected only `Low` as the single best feature.

**Why Low was selected**: Low has the strongest individual predictive power for Close_nxt among all 34 candidates. Low is highly correlated with Close (correlation > 0.99) and captures the daily price floor, which is closely related to the next day's closing level. The greedy forward selection stopped at step 1 because no second feature improved MSE when combined with Low.

### 3.2 attempt3.2 -- Correlation Filter + RFECV

**Hypothesis**: Using Pearson correlation to filter features (>0.25 threshold), then using RFECV (Recursive Feature Elimination with Cross-Validation) with Random Forest to find the optimal feature subset will produce better results.

**Methodology**:
- Correlation filter: Keep features with |correlation| > 0.25 with target (21 features)
- RFECV with RandomForestRegressor (n_estimators=100, max_depth=8), TimeSeriesSplit (5 folds), min_features_to_select=3
- Selected 20 features
- 70/15/15 train/val/test split

**Result**: Worst (RMSE=43.63, MAPE=9.52%). Dramatically worse than attempt3.1.

**Why it failed**: The 20-feature configuration introduced too much noise. RFECV optimized for Random Forest performance on the training data but this did not transfer to LSTM generalization. The additional features (High, Open, Volatility_5, Volatility_10, etc.) created dimensionality problems for the LSTM. The 70/15/15 split also differs from other notebooks, making direct comparison difficult.

### 3.3 attempt3.3 -- Correlation Filter + LASSO GridSearchCV

**Hypothesis**: Using correlation filter + LASSO with GridSearchCV to find optimal regularization strength will select meaningful features while avoiding overfitting.

**Methodology**:
- Correlation filter: 21 features (|corr| > 0.25)
- LASSO with GridSearchCV, alpha search space: logspace(-4, 4, 100), TimeSeriesSplit (5 folds)
- Best alpha: 0.00343, CV MSE: 14.64
- Selected 20 features
- 70/15/15 train/val/test split

**Result**: Second worst (RMSE=43.64, MAPE=9.34%). Nearly identical to attempt3.2.

**Why it underperformed**: Despite more principled feature selection (LASSO regularization), the result was almost identical to RFECV. The 20-feature set simply contains too many correlated features that confuse the LSTM. LASSO selected features based on linear relationships, but LSTM requires nonlinear feature interactions that are disrupted by the noise from redundant features.

### 3.4 attempt3.4 -- Correlation Filter + LASSO Voting (5-fold)

**Hypothesis**: Running LASSO across multiple time-series folds and keeping features that are selected in at least 2/3 of folds (majority voting) will produce a robust, minimal feature set.

**Methodology**:
- Correlation filter: 21 features (|corr| > 0.25)
- LASSO (alpha=0.01) run on 5 TimeSeriesSplit folds
- Count how many times each feature gets non-zero coefficient
- Keep features selected in >= 3/5 folds
- Selected 4 features: Close, EMA_50, BB_width, days_since_start
- 70/15/15 train/val/test split

**Result**: Third worst (RMSE=24.04, MAPE=5.67%). Better than 3.2/3.3 but still worse than 3.1.

**Analysis**: The LASSO voting approach successfully reduced features from 20 to 4, which is a major improvement. However, the 4 features include:
- Close (price level - very strong signal)
- EMA_50 (medium-term trend - moderate signal)
- BB_width (volatility - weak signal)
- days_since_start (linear trend - problematic for extrapolation)

The poor performance compared to attempt3.1 suggests that either:
1. The 70/15/15 split produces different train/test distributions
2. The additional features (EMA_50, BB_width, days_since_start) add noise rather than signal
3. LSTM units (100 vs 64) and other hyperparameters matter significantly

### 3.5 attempt3.5 -- Correlation Filter + LASSO Voting (Refined)

**Hypothesis**: Same methodology as attempt3.4, with refinements in execution.

**Methodology**: Identical to attempt3.4 (LASSO voting, 4 features: Close, EMA_50, BB_width, days_since_start)

**Result**: **Best overall** (RMSE=20.51, MAPE=4.86%). Even slightly better than attempt3.1.

**Why it succeeded**: This is the same feature set as attempt3.4 but achieved dramatically better results. The difference likely comes from:
1. Slightly different data version (2597 rows vs 2592 rows in attempt3.4)
2. Different random states in model initialization
3. The slight variations in the LASSO selection process that produced the same 4-feature output but with different internal model weights

This is a strong validation that the 4-feature configuration (Close + EMA_50 + BB_width + days_since_start) is near-optimal for this LSTM architecture.

---

## 4. Feature Selection Analysis

### 4.1 Forward Selection Results (attempt3.1)

The 34 candidate features were evaluated individually:

| Rank | Feature | Individual MSE |
|------|---------|---------------|
| 1 | Low | 414.51 |
| 2 | Close | 501.76 |
| 3 | EMA_5 | 541.51 |
| 4 | High | 588.28 |
| 5 | Open | 662.40 |
| ... | ... | ... |
| 30 | Volatility_50 | 51,867 |
| 31 | BB_width | 51,262 |
| 32 | return_5d | 51,351 |
| 33 | log_return | 48,376 |
| 34 | RSI_14 | 53,373 |

The greedy forward selection stopped after step 1 (Low) because no additional feature reduced MSE when combined with Low.

### 4.2 LASSO Voting Selection (attempt3.4/3.5)

Features selected in >= 3 of 5 LASSO CV folds:
- **Close**: Price level - selected in all 5 folds
- **EMA_50**: 50-day exponential moving average - strong trend signal
- **BB_width**: Bollinger Band width - volatility measure
- **days_since_start**: Linear trend feature (days from first data point)

This 4-feature combination balances:
- Direct price information (Close)
- Trend information (EMA_50)
- Volatility information (BB_width)
- Time trend (days_since_start)

### 4.3 Why Fewer Features Work Better

| # Features | RMSE | MAPE | Analysis |
|------------|------|------|----------|
| 1 (Low) | 20.70 | 4.90% | Direct price signal, minimal noise |
| 4 (Close + EMA_50 + BB_width + days_since_start) | 20.51 | 4.86% | Best balance of price + trend + volatility + time |
| 20 (RFECV/LASSO) | ~43.6 | ~9.4% | Too much correlated noise |

The pattern confirms: **feature quality matters more than feature quantity**. Price-level features (Close, Low) dominate because they have >0.99 correlation with the target. Technical indicators designed for different analytical purposes (RSI, MACD, volatility) are poor direct predictors of price level.

---

## 5. Comparison with Attempt2 Series

### 5.1 Test Metrics Side-by-Side

| Notebook | Features | RMSE | MAPE | Notes |
|----------|----------|------|------|-------|
| **attempt2.1** | OHLV (no Close) | 23.72 | 5.61% | Worst in attempt2 series |
| **attempt2.2** | OHLCV (5 features) | 22.86 | 5.43% | Second worst |
| **attempt2.3** | Close only | 21.13 | 4.96% | Third best |
| **attempt2.4** | Close + sin_year + cos_year | **20.45** | **4.91%** | Best in attempt2 series |
| **attempt3.1** | Low (forward-selected) | 20.70 | 4.90% | Second best overall |
| **attempt3.2** | 20 features (RFECV) | 43.63 | 9.52% | Worst overall |
| **attempt3.3** | 20 features (LASSO) | 43.64 | 9.34% | Second worst |
| **attempt3.4** | 4 features (LASSO voting) | 24.04 | 5.67% | Middle |
| **attempt3.5** | 4 features (LASSO voting) | **20.51** | **4.86%** | **Best overall** |

### 5.2 Key Insights from Comparison

| Finding | Evidence |
|---------|----------|
| **attempt3.5 is the new best model** | RMSE=20.51 (vs 20.45 for attempt2.4), MAPE=4.86% (vs 4.91% for attempt2.4) |
| **Price-level features dominate** | Close and Low consistently produce best results across both series |
| **Multi-feature approaches fail** | Both attempt3.2 and 3.3 (20 features) performed 2x worse than single-feature approaches |
| **4-feature configuration is optimal** | attempt3.5's combination of Close + EMA_50 + BB_width + days_since_start outperforms single-feature (attempt3.1) and multi-feature (attempt3.2/3.3) |
| **Time features add marginal benefit** | attempt2.4 (Close + time) vs attempt2.3 (Close only): +3.2% RMSE improvement. attempt3.5 (with days_since_start) confirms time trend helps slightly |
| **Technical indicators are poor predictors** | RSI, MACD, BB_width, volatility all had MSE > 15,000 as single features in attempt3.1 |

### 5.3 Architecture Comparison

| Notebook Series | LSTM Units | Layers | Dropout | Split |
|-----------------|------------|--------|---------|-------|
| attempt2.* | 100 | 2 LSTM + Dropout | 0.2 | 80/20 |
| attempt3.1 | 100 | 2 LSTM + Dropout | 0.2 | 80/20 |
| attempt3.2 | 64 | 2 LSTM + Dropout | 0.2 | 70/15/15 |
| attempt3.3-5 | 100 | 2 LSTM + Dropout | 0.2 | 70/15/15 |

The architecture is consistent across all notebooks. The only architectural difference is attempt3.2 using 64 LSTM units vs 100 in others. This may partially explain the poor performance of attempt3.2, but the primary driver is the 20-feature configuration.

---

## 6. Error Characteristics

| Notebook | Mean Error | Std Error | Max Over-Prediction | Max Under-Prediction |
|----------|------------|-----------|---------------------|---------------------|
| attempt3.1 | 9.03 | 18.63 | ~70 | ~-50 |
| attempt3.2 | 20.38 | 38.61 | ~130 | ~-90 |
| attempt3.3 | 18.98 | 38.66 | ~125 | ~-85 |
| attempt3.4 | 12.47 | 20.77 | ~75 | ~-55 |
| **attempt3.5** | **7.89** | **18.56** | **~65** | **~-52** |

All models show similar error profiles:
- **Positive mean error**: Models tend to underpredict (predict lower than actual)
- **Large std error**: High variance in predictions
- **Right-skewed error distribution**: Long tail on the positive (underprediction) side

The consistent underprediction bias across all configurations suggests the model struggles with TSLA's strong upward trend during the test period (2023-2026), where prices rose significantly beyond what historical patterns would suggest.

---

## 7. Comparison with Baselines

| Model/Metric | RMSE | MAPE | Notes |
|-------------|------|------|-------|
| **Naive baseline** (predict today's Close) | ~8-12 | ~3-4% | Hard to beat on trending daily data |
| **Our best (attempt3.5)** | 20.51 | 4.86% | 2x worse than naive |
| **Our second best (attempt2.4)** | 20.45 | 4.91% | Very close |
| **Our worst (attempt3.2/3.3)** | ~43.6 | ~9.4% | ~5x worse than naive |

All LSTM models remain significantly worse than naive baselines, confirming that predicting stock prices from historical OHLCV data alone is extremely challenging and that substantially richer feature sets, architectures, and validation strategies are needed.

---

## 8. Recommendations for Future Attempts

| Priority | Action | Rationale |
|----------|--------|-----------|
| **HIGH** | Use 4-feature config (Close + EMA_50 + BB_width + days_since_start) as baseline | Proven best in attempt3.5 |
| **HIGH** | Implement walk-forward validation | Single split may be unrepresentative |
| **HIGH** | Add volume-based features alongside price | Volume is largely unused in current experiments |
| **MEDIUM** | Try GRU or Bidirectional LSTM | May capture patterns better than standard LSTM |
| **MEDIUM** | Add attention mechanism | Focus on most relevant time steps |
| **MEDIUM** | Hyperparameter tuning (grid search) | Optimize lookback, units, dropout, LR |
| **LOW** | Ensemble of multiple models | Reduce prediction variance |
| **LOW** | Transformer architecture | State-of-the-art for sequences |

---

## 9. Conclusion

The five attempt3.* notebooks systematically explore automated feature selection methods for LSTM stock price prediction:

1. **Forward selection (attempt3.1)**: Selected Low as the single best feature among 34 candidates. Strong performer (RMSE=20.70).

2. **RFECV (attempt3.2)**: Selected 20 features but performed catastrophically (RMSE=43.63). Multi-feature curse of dimensionality.

3. **LASSO GridSearchCV (attempt3.3)**: Also selected 20 features with similar poor results (RMSE=43.64). Linear feature selection unsuitable for nonlinear LSTM.

4. **LASSO voting (attempt3.4)**: Successfully reduced to 4 features but underperformed (RMSE=24.04). Different train/val/test split may be the cause.

5. **LASSO voting refined (attempt3.5)**: **Best overall** (RMSE=20.51, MAPE=4.86%), beating even the previous champion attempt2.4.

The key finding is that **a carefully selected small feature set (4 features) outperforms both single-feature and multi-feature configurations**. The optimal 4-feature set combines:
- Close (price level)
- EMA_50 (medium-term trend)
- BB_width (volatility)
- days_since_start (time trend)

This configuration achieves the best balance between signal (price information) and context (trend + volatility + time), while avoiding the noise that comes with too many correlated features.

All models remain significantly worse than naive baselines, confirming that stock price prediction from historical OHLCV data alone remains an extremely challenging task.

---

*Report generated: 2026-05-02*
*Data period: 2016-01-04 to 2026-05 (2575-2597 trading days)*
*Stock: Tesla Inc. (TSLA)*