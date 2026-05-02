# Comparative Analysis: LSTM Stock Price Prediction Attempts

## Tesla (TSLA) | 2016-01-01 to 2026 | ~2574 Trading Days

---

## 1. Experiment Configuration

| Notebook | Input Features | # Feat. | Target | Methodology | LSTM Units | Lookback |
|----------|---------------|---------|--------|-------------|------------|----------|
| **attempt2.1** | Open, High, Low, Volume | 4 | Close_nxt | Manual selection (Close excluded) | 100 | 60 |
| **attempt2.2** | Open, High, Low, Close, Volume | 5 | Close_nxt | Manual selection (full OHLCV) | 100 | 60 |
| **attempt2.3** | Close | 1 | Close_nxt | Manual selection (Close only) | 100 | 60 |
| **attempt2.4** | Close, sin_year, cos_year | 3 | Close_nxt | Manual selection (Close + time) | 100 | 60 |
| **attempt3** | Low | 1 | Close_nxt | Forward feature selection (from 34 candidates) | 100 | 60 |

All five notebooks share:
- **Target**: `Close_nxt` = next day's closing price (shifted by -1)
- **Scaler**: RobustScaler, fit on training data only
- **Split**: 80/20 time-based, performed before scaling
- **Sequence padding**: Last `LOOKBACK` rows from training prepended to test
- **Training**: Adam optimizer, MSE loss, EarlyStopping (patience=10), up to 100 epochs, batch_size=32, 10% validation split

---

## 2. Test Metrics Comparison

| Notebook | Features | MSE | RMSE | MAE | MAPE | Mean Error | Std Error |
|----------|----------|-----|------|-----|------|------------|-----------|
| **attempt2.1** | OHLV (no Close) | 562.60 | 23.72 | 18.17 | 5.61% | 11.65 | 20.66 |
| **attempt2.2** | OHLCV | 522.65 | 22.86 | 17.53 | 5.43% | 10.86 | 20.12 |
| **attempt2.3** | Close only | 446.43 | 21.13 | 16.12 | 4.96% | 10.09 | 18.56 |
| **attempt2.4** | Close + time | **418.35** | **20.45** | **15.55** | **4.91%** | **8.51** | 18.60 |
| **attempt3** | Low (forward-selected) | 428.66 | 20.70 | 15.62 | **4.90%** | 9.03 | 18.63 |

### Ranking by RMSE (lower is better):
1. **attempt2.4** (Close + sin_year + cos_year): **20.45** -- Best overall
2. **attempt3** (Low, forward-selected): 20.70
3. **attempt2.3** (Close only): 21.13
4. **attempt2.2** (OHLCV): 22.86
5. **attempt2.1** (OHLV, no Close): 23.72 -- Worst

### Ranking by MAPE (lower is better):
1. **attempt3** (Low): **4.90%**
2. **attempt2.4** (Close + time): 4.91%
3. **attempt2.3** (Close only): 4.96%
4. **attempt2.2** (OHLCV): 5.43%
5. **attempt2.1** (OHLV): 5.61%

---

## 3. Fundamental Differences Between Notebooks

### 3.1 attempt2.1 -- OHLV (Close Excluded)

**Hypothesis**: Can the LSTM predict next-day Close using only Open, High, Low, and Volume -- without the Close price itself?

**Result**: Worst performance across all metrics (RMSE=23.72, MAPE=5.61%).

**Why it failed**: The Close price is the most directly informative feature for predicting the next Close. Removing it forces the model to implicitly reconstruct price levels from the relationships between Open, High, Low, and Volume, which is a much harder indirect inference task. The model must first estimate where Close was, then predict where it will go -- adding an unnecessary intermediary step that introduces error.

### 3.2 attempt2.2 -- Full OHLCV

**Hypothesis**: Using all five standard OHLCV features provides the most complete daily price information.

**Result**: Second worst (RMSE=22.86, MAPE=5.43%). Better than 2.1 but worse than simpler configurations.

**Why it underperformed**: Adding High, Low, Open, and Volume alongside Close introduces correlated but noisy features. The LSTM has more parameters to learn from, but the additional features add noise rather than signal. High and Low are closely related to Close, and Volume carries different units/scales that may confuse the model despite RobustScaler normalization. This demonstrates the classic "curse of dimensionality" -- more features without proportional information gain degrades generalization.

### 3.3 attempt2.3 -- Close Only

**Hypothesis**: A single well-chosen feature (Close) with the most direct predictive relationship to the target may outperform multi-feature setups.

**Result**: Third best (RMSE=21.13, MAPE=4.96%). A strong, clean baseline.

**Why it works**: Close is the single most informative feature for predicting Close_nxt. With only one input dimension, the LSTM has minimal noise to filter and can focus entirely on learning temporal patterns in the price series. The reduced parameter space (LSTM input: 1 instead of 4-5) also reduces overfitting risk. This validates the principle that **feature quality matters more than feature quantity**.

### 3.4 attempt2.4 -- Close + Cyclical Time Features

**Hypothesis**: Adding calendar-based features (sin/cos encoding of day-of-year) captures seasonal patterns in TSLA stock.

**Result**: Best overall (RMSE=20.45, MAPE=4.91%). A ~3.2% improvement over Close-only.

**Why it works**: The sin_year and cos_year features provide the model with awareness of annual cycles (earnings seasons, tax deadlines, holiday effects) without introducing raw date values that would be unbounded. The cyclic encoding is scale-invariant (values in [-1, 1]) and doesn't require scaling, making it naturally compatible with RobustScaler applied to Close. The modest improvement suggests TSLA does exhibit weak seasonal patterns that the LSTM can exploit.

### 3.5 attempt3 -- Forward Feature Selection from 34 Candidates

**Hypothesis**: Systematically selecting the best feature subset from 34 technical indicators, returns, ratios, and time features will find an optimal configuration.

**Result**: Second best (RMSE=20.70, MAPE=4.90%). The forward selection algorithm chose only `Low` as the single best feature.

**Why Low was selected**: Low has the strongest individual predictive power for Close_nxt among all 34 candidates. Low is highly correlated with Close (correlation > 0.99) and captures the daily price floor, which is closely related to the next day's closing level. Notably, the greedy forward selection stopped at step 1 -- no second feature improved MSE when combined with Low. This is a strong validation that for this LSTM architecture and dataset, a single price-based feature dominates.

**Why the 34 candidates underperformed individually**: Features like RSI, MACD, Bollinger Bands, returns, and volatility are designed for different analytical purposes (momentum, trend, volatility measurement) and their raw values are poor direct predictors of price level. They are better suited as features for classification (up/down) or as components in more sophisticated models.

---

## 4. Key Insights

### 4.1 Feature Engineering Insights

| Finding | Evidence |
|---------|----------|
| **Close/Low are the dominant features** | Both attempt2.3 (Close) and attempt3 (Low) are in the top 3; both have >0.99 correlation with target |
| **More features ≠ better** | attempt2.2 (5 features) lost to attempt2.3 (1 feature) and attempt2.4 (3 features) |
| **Removing the most relevant feature is catastrophic** | attempt2.1 (no Close) was worst by a wide margin |
| **Time features provide marginal benefit** | attempt2.4 vs attempt2.3: +3.2% RMSE improvement from sin/cos encoding |
| **Technical indicators are poor direct predictors** | RSI, MACD, BB_width, volatility all had MSE > 15,000 as single features |

### 4.2 Forward Selection Results (attempt3)

The 34 candidate features were evaluated individually. Top 5 by individual MSE:

| Rank | Feature | Individual MSE |
|------|---------|---------------|
| 1 | Low | 414.51 |
| 2 | Close | 501.76 |
| 3 | EMA_5 | 541.51 |
| 4 | High | 588.28 |
| 5 | Open | 662.40 |

Bottom 5 (worst features):
| Rank | Feature | Individual MSE |
|------|---------|---------------|
| 30 | Volatility_50 | 51,867 |
| 31 | BB_width | 51,262 |
| 32 | return_5d | 51,351 |
| 33 | log_return | 48,376 |
| 34 | RSI_14 | 53,373 |

The greedy forward selection stopped after step 1 (Low) because no additional feature reduced MSE when combined with Low.

### 4.3 Error Characteristics

All models show similar error profiles:
- **Positive mean error** (8.5-11.7): Models tend to underpredict (predict lower than actual)
- **Large std error** (18.6-20.7): High variance in predictions
- **Max over-prediction** range: 68-84 USD
- **Max under-prediction** range: -49 to -55 USD
- **Right-skewed error distribution**: Long tail on the positive (underprediction) side

This consistent underprediction bias across all configurations suggests the model struggles with TSLA's strong upward trend during the test period (2023-2026), where prices rose significantly beyond what historical patterns would suggest.

---

## 5. Comparison with Baselines

| Model/Metric | RMSE | MAPE | Notes |
|-------------|------|------|-------|
| **Naive baseline** (predict today's Close) | ~8-12 | ~3-4% | Hard to beat on trending daily data |
| **Our best (attempt2.4)** | 20.45 | 4.91% | 2x worse than naive |
| **Our second best (attempt3)** | 20.70 | 4.90% | Forward-selected Low |
| **Our worst (attempt2.1)** | 23.72 | 5.61% | No Close feature |

The models perform **worse than a naive "tomorrow = today" baseline**, indicating the LSTM is not extracting useful predictive signal from historical patterns. This is consistent with the Efficient Market Hypothesis -- past price patterns have limited predictive power for future prices.

---

## 6. Recommendations for attempt4

| Priority | Action | Rationale |
|----------|--------|-----------|
| **HIGH** | Use Close + time features as baseline (attempt2.4 config) | Best proven configuration |
| **HIGH** | Add returns/volatility features alongside Close | Capture momentum and risk |
| **HIGH** | Implement walk-forward validation | Single split may be unrepresentative |
| **MEDIUM** | Try GRU or Bidirectional LSTM | May capture patterns better |
| **MEDIUM** | Add attention mechanism | Focus on most relevant time steps |
| **MEDIUM** | Hyperparameter tuning (grid search) | Optimize lookback, units, dropout, LR |
| **LOW** | Ensemble of multiple models | Reduce prediction variance |
| **LOW** | Transformer architecture | State-of-the-art for sequences |

---

## 7. Conclusion

The five attempt2.x/attempt3 notebooks systematically explore how input feature selection affects LSTM stock price prediction. The key finding is that **a single price-level feature (Close or Low) outperforms multi-feature setups** for this specific architecture and dataset. Adding cyclical time encoding (attempt2.4) provides the best overall results with RMSE=20.45 and MAPE=4.91%.

The forward selection experiment (attempt3) confirmed that Low is the strongest individual predictor among 34 candidates, but the greedy algorithm found no benefit from combining features -- a result that likely reflects the limitations of the simple 2-layer LSTM architecture rather than a true absence of multi-feature signal.

All models remain significantly worse than naive baselines, confirming that predicting stock prices from historical OHLCV data alone is extremely challenging and that substantially richer feature sets, architectures, and validation strategies are needed.

---

*Report generated: 2026-04-01*
*Data period: 2016-01-04 to 2026-03 (2574-2575 trading days)*
*Stock: Tesla Inc. (TSLA)*
