# Comparative Analysis: Attempt3 Series LSTM Stock Price Prediction

## Tesla (TSLA) | 2016-01-01 to 2026 | ~2575 Trading Days

---

## 1. Experiment Overview

The attempt3 series explores different feature selection methodologies to determine optimal input features for LSTM stock price prediction. All notebooks share the same base architecture:
- **Target**: `Close_nxt` = next day's closing price
- **LSTM Architecture**: 2 LSTM layers (100 units each) with Dropout(0.2)
- **Lookback**: 60 days
- **Train/Test Split**: 80/20 time-based (before scaling)
- **Scaler**: RobustScaler (fit on training data only)

---

## 2. Feature Selection Methods Comparison

| Notebook | Method | Description |
|----------|--------|-------------|
| **attempt3.1** | Greedy Forward Selection | Evaluated all 34 features individually, then greedily added features that reduced MSE |
| **attempt3.2** | RFECV | Recursive Feature Elimination with Cross-Validation using Random Forest estimator |
| **attempt3.3** | Lasso + GridSearchCV | L1 regularization with cross-validation to select optimal alpha and features |
| **attempt3.4** | Correlation Filter | Pearson correlation filter (threshold > 0.25) + manual selection |
| **attempt3.5** | Same as 3.4 | Identical methodology to attempt3.4 |

---

## 3. Test Metrics Comparison

| Notebook | Features | # Features | MSE | RMSE | MAPE |
|----------|----------|------------|-----|------|------|
| **attempt3.1** | Low (forward) | 1 | 428.66 | **20.70** | **4.90%** |
| **attempt3.4** | Close, EMA_50, BB_width, days_since_start | 4 | 577.90 | 24.04 | 5.67% |
| **attempt3.5** | Close, EMA_50, BB_width, days_since_start | 4 | 577.90 | 24.04 | 5.67% |
| **attempt3.2** | 20 features (RFECV) | 20 | 1903.35 | 43.63 | 9.52% |
| **attempt3.3** | Lasso-selected | - | 1904.12 | 43.64 | 9.34% |

### Ranking by RMSE (lower is better):
1. **attempt3.1** (Low): **20.70** — Best overall
2. **attempt3.4/3.5** (4 features): 24.04
3. **attempt3.2** (RFECV 20 features): 43.63
4. **attempt3.3** (Lasso): 43.64 — Worst

### Ranking by MAPE (lower is better):
1. **attempt3.1** (Low): **4.90%**
2. **attempt3.4/3.5**: 5.67%
3. **attempt3.3**: 9.34%
4. **attempt3.2**: 9.52%

---

## 4. Analysis by Notebook

### 4.1 attempt3.1 — Forward Selection (BEST)

**Method**: Greedy forward selection from 34 candidate features (technical indicators, returns, ratios, time features)

**Result**: Selected only `Low` as the best feature. The algorithm stopped after step 1 because no additional feature improved MSE when combined with Low.

**Why it worked**:
- Low has the strongest individual predictive power (MSE=414.51 as single feature)
- Correlation with Close > 0.99
- Single feature reduces overfitting risk
- The greedy algorithm correctly identified that additional features added noise rather than signal

**Individual feature MSE rankings from attempt3.1**:
| Rank | Feature | Individual MSE |
|------|---------|---------------|
| 1 | Low | 414.51 |
| 2 | Close | 501.76 |
| 3 | EMA_5 | 541.51 |
| 4 | High | 588.28 |
| 5 | Open | 662.40 |
| ... | ... | ... |
| 34 | RSI_14 | 53,373 |

---

### 4.2 attempt3.2 — RFECV Selection (WORST)

**Method**: Correlation filter (threshold > 0.25) followed by RFECV with Random Forest estimator

**Result**: 20 features selected, but performance is very poor (RMSE=43.63, MAPE=9.52%)

**Why it failed**:
- RFECV with RF estimator may select features that work well for tree-based models but not for LSTM
- 20 features introduce significant noise and increase model complexity
- The correlation filter may have kept highly correlated features that are redundant for LSTM
- More features = more parameters to train = higher overfitting risk

---

### 4.3 attempt3.3 — Lasso Selection

**Method**: Lasso regression with GridSearchCV for alpha optimization

**Result**: Poor performance (RMSE=43.64, MAPE=9.34%)

**Why it failed**:
- Lasso is designed for linear relationships; LSTM captures non-linear patterns
- Selected features may work for linear prediction but not for sequence modeling
- The regularization may have eliminated important non-linear temporal features

---

### 4.4 attempt3.4/3.5 — Correlation Filter

**Method**: Pearson correlation filter (threshold > 0.25), resulting in 21 features initially, then reduced to 4 features manually

**Selected Features**: Close, EMA_50, BB_width, days_since_start

**Result**: Moderate performance (RMSE=24.04, MAPE=5.67%)

**Why it performed moderately**:
- Close provides direct price information
- EMA_50 captures medium-term trend
- BB_width provides volatility context
- days_since_start adds temporal information
- However, 4 features still introduce more complexity than single feature

---

## 5. Key Insights

### 5.1 Feature Quality vs Quantity

| Finding | Evidence |
|---------|----------|
| **Single feature can outperform multi-feature** | attempt3.1 (1 feature) >> attempt3.2 (20 features) |
| **Low is the strongest predictor** | Selected by forward selection, MSE=414.51 |
| **Price-based features dominate** | Low, Close, EMA_5, High, Open in top 5 |
| **Technical indicators are poor predictors** | RSI, MACD, BB_width have MSE > 15,000 as singles |

### 5.2 Feature Selection Method Comparison

| Method | attempt3.1 (Forward) | attempt3.2 (RFECV) | attempt3.3 (Lasso) | attempt3.4 (Correlation) |
|--------|---------------------|-------------------|-------------------|------------------------|
| Features Selected | 1 | 20 | - | 4 |
| RMSE | **20.70** | 43.63 | 43.64 | 24.04 |
| MAPE | **4.90%** | 9.52% | 9.34% | 5.67% |
| Rating | BEST | WORST | WORST | MODERATE |

### 5.3 Why More Features Performed Worse

1. **Noise amplification**: Additional features introduce noise that confuses the LSTM
2. **Redundancy**: Many selected features are highly correlated (e.g., multiple EMAs)
3. **Overfitting**: More parameters to train with limited data leads to overfitting
4. **Inappropriate for LSTM**: Methods designed for tree models or linear models don't transfer well

---

## 6. Comparison with Previous Attempts (attempt2.x)

From the existing comparison_report.md:

| Notebook | Configuration | RMSE | MAPE |
|----------|---------------|------|------|
| **attempt2.4** (best) | Close + sin/cos year | **20.45** | **4.91%** |
| **attempt3.1** (best in attempt3) | Low (forward) | 20.70 | 4.90% |
| **attempt2.3** | Close only | 21.13 | 4.96% |
| **attempt3.4** | 4 features | 24.04 | 5.67% |
| **attempt2.2** | OHLCV (5 features) | 22.86 | 5.43% |
| **attempt3.2** | 20 features (RFECV) | 43.63 | 9.52% |

**Best overall**: attempt2.4 (Close + time features) with RMSE=20.45

---

## 7. Conclusions

1. **Forward selection from attempt3.1 is the best in this series**, achieving RMSE=20.70 and MAPE=4.90% with a single feature (Low)

2. **More features ≠ better performance**: The 20-feature attempt3.2 performed 2x worse than single-feature attempt3.1

3. **LSTM prefers simple input**: A single well-chosen feature (Low or Close) provides the best results

4. **Time features help**: Adding cyclical time encoding (attempt2.4) provides marginal improvement over price-only features

5. **Feature selection method matters**: Forward selection correctly identified Low as optimal; RFECV and Lasso selected inappropriate features for LSTM

---

## 8. Recommendations for Future Attempts

| Priority | Action | Rationale |
|----------|--------|-----------|
| **HIGH** | Use single price feature (Low or Close) | Proven best in attempt3.1 |
| **HIGH** | Add cyclical time features | Improved results in attempt2.4 |
| **MEDIUM** | Try different lookback periods | 60 may not be optimal |
| **MEDIUM** | Experiment with GRU/attention | May capture patterns better |
| **LOW** | Multi-stock training | Increase dataset diversity |

---

*Report generated: 2026-05-02*
*Data period: 2016-01-04 to 2026-03 (2575 trading days)*
*Stock: Tesla Inc. (TSLA)*