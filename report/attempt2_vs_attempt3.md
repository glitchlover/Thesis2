# Comparative Analysis: Attempt2 vs Attempt3 Series

## Understanding Performance Differences and Declines

---

## 1. Executive Summary

This report compares the attempt2 and attempt3 series of LSTM stock price prediction experiments to understand why some attempt3 configurations performed significantly worse than attempt2, while others achieved comparable results.

### Key Findings:
- **Best performers are comparable**: attempt2.4 (RMSE=20.45) and attempt3.1 (RMSE=20.70) both achieve ~4.9% MAPE
- **Worst decline**: attempt3.2/3.3 degraded from ~20 RMSE to ~43 RMSE (2x worse)
- **Root cause**: Feature selection methods designed for tree models and linear models fail for LSTM

---

## 2. Side-by-Side Comparison

### All Notebooks Ranked by RMSE

| Rank | Notebook | Features | RMSE | MAPE | Series |
|------|----------|----------|------|------|--------|
| 1 | **attempt2.4** | Close + sin/cos year | **20.45** | **4.91%** | 2 |
| 2 | **attempt3.1** | Low (forward) | **20.70** | **4.90%** | 3 |
| 3 | attempt2.3 | Close only | 21.13 | 4.96% | 2 |
| 4 | attempt3.4 | 4 features | 24.04 | 5.67% | 3 |
| 5 | attempt3.5 | 4 features | 24.04 | 5.67% | 3 |
| 6 | attempt2.2 | OHLCV (5) | 22.86 | 5.43% | 2 |
| 7 | attempt2.1 | OHLV (4) | 23.72 | 5.61% | 2 |
| 8 | **attempt3.2** | 20 features | **43.63** | **9.52%** | 3 |
| 9 | **attempt3.3** | Lasso | **43.64** | **9.34%** | 3 |

---

## 3. Why Attempt3.2 and Attempt3.3 Declined Significantly

### 3.1 The Performance Drop

| Metric | Best (attempt2.4) | Worst (attempt3.2) | Change |
|--------|------------------|-------------------|--------|
| RMSE | 20.45 | 43.63 | +113% (2.1x worse) |
| MAPE | 4.91% | 9.52 | +94% (1.9x worse) |

### 3.2 Root Cause Analysis

#### **Problem 1: RFECV Uses Tree-Based Optimizer**

```
attempt3.2 Method: RFECV with Random Forest estimator
```

- RFECV uses Random Forest as the estimator to evaluate feature importance
- Random Forest and LSTM have fundamentally different:
  - **Learning mechanisms**: Trees partition data; LSTM learns sequential patterns
  - **Feature interactions**: Trees handle non-linear interactions automatically; LSTM needs explicit patterns
  - **Temporal awareness**: Trees don't understand sequence; LSTM specifically designed for sequences

**Result**: Features selected for RF may be meaningless for LSTM

#### **Problem 2: Lasso Selects for Linear Relationships**

```
attempt3.3 Method: Lasso regression with GridSearchCV
```

- Lasso performs L1 regularization to select features based on linear correlation with target
- Stock prices are non-linear and LSTM captures:
  - Long-term dependencies
  - Non-linear temporal patterns
  - Complex feature interactions

**Result**: Lasso-selected features work for linear prediction, not for LSTM sequence modeling

#### **Problem 3: Feature Explosion**

| Notebook | # Features | RMSE | Issue |
|----------|------------|------|-------|
| attempt2.4 | 3 | 20.45 | Optimal |
| attempt3.1 | 1 | 20.70 | Optimal |
| attempt3.4 | 4 | 24.04 | Moderate |
| attempt3.2 | **20** | 43.63 | **Too many** |

- 20 features introduce massive noise
- Highly correlated features (multiple EMAs, multiple volatilities) create redundancy
- More parameters = more overfitting = worse generalization

---

## 4. Why Attempt3.1 and Attempt3.4 Performed Well

### 4.1 attempt3.1 — Forward Selection (RMSE=20.70)

**Success factors**:
1. **LSTM-native evaluation**: Each feature tested directly in LSTM, not in a proxy model
2. **Greedy optimization**: Only adds features that actually improve LSTM test MSE
3. **Minimalism**: Correctly identified that single feature (Low) is optimal
4. **Stop condition**: Algorithm correctly stopped when no improvement found

**Key insight**: Forward selection "speaks LSTM's language" by evaluating features in the actual model

### 4.2 attempt3.4 — Correlation Filter (RMSE=24.04)

**Moderate success factors**:
1. **Manual curation**: Human intervention to select meaningful features
2. **Price-based core**: Close is direct price information
3. **Trend capture**: EMA_50 captures medium-term trend
4. **Volatility context**: BB_width provides volatility measure
5. **Temporal anchor**: days_since_start prevents concept drift

**Limitation**: Still 4 features = more complexity than needed

---

## 5. Comparison: Manual vs Automated Feature Selection

### Attempt2 Series (Manual Selection) — Generally Successful

| Notebook | Method | Features | RMSE | Outcome |
|----------|--------|----------|------|---------|
| attempt2.1 | Remove Close | 4 | 23.72 | Poor (removed key feature) |
| attempt2.2 | Use all OHLCV | 5 | 22.86 | Moderate |
| attempt2.3 | Close only | 1 | 21.13 | Good |
| attempt2.4 | Close + time | 3 | **20.45** | **Best** |

**Pattern**: Manual selection works well when it follows domain knowledge:
- Keep Close (most informative)
- Add time features (proven to help)
- Avoid removing key features

### Attempt3 Series (Automated Selection) — Mixed Results

| Notebook | Method | Features | RMSE | Outcome |
|----------|--------|----------|------|---------|
| attempt3.1 | Forward selection | 1 | **20.70** | **Good** |
| attempt3.2 | RFECV (RF) | 20 | 43.63 | **Very Poor** |
| attempt3.3 | Lasso | ? | 43.64 | **Very Poor** |
| attempt3.4 | Correlation filter | 4 | 24.04 | Moderate |

**Pattern**: Automated selection fails when:
- Uses wrong optimizer (RF instead of LSTM)
- Uses wrong assumption (linear vs non-linear)
- Selects too many features

---

## 6. Key Insights

### Insight 1: Feature Selection Must Match Model Type

| Feature Selection Method | Best For | Worst For |
|-------------------------|----------|-----------|
| Forward selection (LSTM) | LSTM, RNNs | - |
| Manual domain knowledge | Any | - |
| Correlation threshold | Linear models, EDA | LSTM |
| RFECV (Random Forest) | Tree models | LSTM |
| Lasso/ElasticNet | Linear regression | LSTM |

### Insight 2: Less is More for LSTM

```
Performance vs Number of Features:
1 feature (Close/Low):  RMSE ~20-21  ✓ BEST
3 features (Close+time): RMSE ~20    ✓ EXCELLENT
4 features:             RMSE ~24     ✓ MODERATE
5 features (OHLCV):     RMSE ~23     ✓ MODERATE
20 features:            RMSE ~43     ✗ DISASTER
```

### Insight 3: The Proxy Model Problem

When using automated feature selection:
- **Don't use a different model type as the evaluator**
- RFECV with RF selects features for RF, not for LSTM
- Lasso selects features for linear regression, not for LSTM
- Only forward selection evaluates in the actual model

### Insight 4: Domain Knowledge Still Matters

The best results came from combining:
- Domain knowledge (Close is key)
- LSTM-native evaluation (forward selection)
- Minimal feature set (1-3 features)

---

## 7. Technical Recommendations

### For Future Attempts

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| HIGH | Use forward selection with LSTM evaluator | Proven best in attempt3.1 |
| HIGH | Limit features to 1-3 | Prevents overfitting |
| HIGH | Evaluate in actual LSTM, not proxy | Avoids RFECV/Lasso trap |
| MEDIUM | Add time features (sin/cos encoding) | Small improvement seen in attempt2.4 |
| MEDIUM | Try different lookback windows | 60 may not be optimal |
| LOW | Experiment with attention mechanisms | May help with feature weighting |

### What to Avoid

| Don't Do | Why |
|----------|-----|
| Use RFECV with tree estimator | Selects wrong features for LSTM |
| Use Lasso for feature selection | Assumes linear relationships |
| Select >5 features | Introduces noise and redundancy |
| Remove Close/Low | Most informative features |
| Use technical indicators as primary | Poor direct predictors |

---

## 8. Conclusion

The performance decline in attempt3.2/3.3 (from ~20 to ~43 RMSE) is not a fundamental flaw with LSTM or the attempt3 series, but rather a result of:

1. **Inappropriate feature selection methods** (RFECV, Lasso) that don't evaluate features in the actual model
2. **Feature explosion** (20 features) that introduces noise and overfitting
3. **Using proxy models** that select features for different model types

The best results (attempt2.4, attempt3.1) both achieved RMSE ~20.45-20.70 using:
- Minimal feature sets (1-3 features)
- Price-based features (Close, Low)
- LSTM-native evaluation (forward selection or manual domain knowledge)
- Optional: cyclical time encoding

**Final takeaway**: For LSTM stock prediction, less is more, and the evaluation method must match the model type.

---

*Report generated: 2026-05-02*
*Data period: 2016-01-04 to 2026-03 (2575 trading days)*
*Stock: Tesla Inc. (TSLA)*