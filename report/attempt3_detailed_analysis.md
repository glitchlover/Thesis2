# Comparative Analysis: Attempt3 Series Feature Selection Experiments

## Tesla (TSLA) | 2016-01-01 to 2026 | ~2574 Trading Days

---

## 1. Experiment Configuration

| Notebook | Input Features | # Feat. | Target | Methodology | Split | LSTM Units | Lookback |
|----------|---------------|---------|--------|-------------|------------|----------|----------|
| **attempt3.1** | Low | 1 | Close_nxt | individual feature MSE evaluation + greedy forward selection | 80/20 | 100 | 60 |
| **attempt3.2** | Close, High, Low, Open, EMA_5, Volatility_5, EMA_10, Volatility_10, EMA_20, Volatility_20, EMA_50, Volatility_50, MACD, MACD_signal, BB_upper, BB_lower, BB_width, ATR_14, OBV, days_since_start | 20 | Close_nxt | correlation filter + RFECV | 70/15/15 | 100 | 60 |
| **attempt3.3** | Close, High, Low, Open, EMA_5, Volatility_5, EMA_10, Volatility_10, EMA_20, EMA_50, Volatility_50, MACD, MACD_signal, BB_upper, BB_lower, BB_width, ATR_14, OBV, high_low_ratio, days_since_start | 20 | Close_nxt | correlation filter + LASSO | 70/15/15 | 100 | 60 |
| **attempt3.4** | Close, EMA_50, BB_width, days_since_start | 4 | Close_nxt | correlation filter + LASSO | 70/15/15 | 100 | 60 |

All notebooks share:

- **Target**: `Close_nxt` = next day's closing price (shifted by -1), in USD
- **Scaler**: RobustScaler (median-based, robust to outliers), fit on training data only to prevent data leakage
- **Sequence padding**: Last `LOOKBACK` (60) rows from training prepended to test sequences for consistent input length
- **Training**: Adam optimizer, MSE loss, EarlyStopping (patience=10, monitor val_loss), up to 100 epochs, batch_size=32, 10% validation split from training data

Note: Split ratios vary between notebooks (attempt3.1 uses 80/20 train/test, others use 70/15/15 train/val/test), affecting metric comparisons. Feature selection methods include:
- **Correlation filter**: Pre-filters features with absolute Pearson correlation > 0.25 to target
- **Greedy forward**: Wrapper method starting from individual features, greedily adding those maximizing MSE improvement (Kohavi & John, 1997)
- **RFECV**: Recursive Feature Elimination with Cross-Validation, iteratively removing features based on importance scores (Guyon & Elisseeff, 2003)
- **LASSO**: Embedded method with L1 regularization to select sparse models by penalizing coefficients (Tibshirani, 1996)

---

## 2. Differences from Attempt2 Series

The attempt2 series used manual feature selection (OHLCV combinations with domain intuition), while attempt3 series implemented automatic feature selection algorithms.

| Aspect | Attempt2 | Attempt3 |
|--------|----------|----------|
| Feature selection | Manual intuition | Automatic algorithms (greedy, RFECV, LASSO) |
| Best performance | Close + time (RMSE=20.45) | Low only (RMSE=20.70) |
| Feature count | 1-5 features | 1-20 features |
| Methodology | Domain knowledge-driven | Data-driven, algorithmic |

**Key difference**: Automatic methods selected Low as optimal for best performance, while LASSO selected 4 features with poorer results. Manual selection preferred Close. Price-level features dominate, but selection method matters.

---

## 3. Test Metrics Comparison

Metrics are evaluated on test sets, measuring prediction accuracy for TSLA closing prices (in USD). Key definitions:
- **MSE**: Mean Squared Error (average squared prediction errors, in USD²)
- **RMSE**: Root MSE (square root of MSE, in USD, same units as target)
- **MAE**: Mean Absolute Error (average absolute prediction errors, in USD)
- **MAPE**: Mean Absolute Percentage Error (average relative error as percentage)
- **Mean Error**: Average prediction bias (positive = underprediction)
- **Std Error**: Standard deviation of prediction errors (prediction variability)

| Notebook | Features | MSE (USD²) | RMSE (USD) | MAE (USD) | MAPE | Mean Error (USD) | Std Error (USD) |
|----------|----------|------------|------------|-----------|------|------------------|-----------------|
| **attempt3.1** | Low (greedy-selected) | 428.66 | 20.70 | 15.62 | 4.90% | 9.03 | 18.63 |
| **attempt3.2** | 20 features (RFECV) | 1903.35 | 43.63 | 32.05 | 9.52% | 26.54 | 34.63 |
| **attempt3.3** | 20 features (LASSO) | 1904.12 | 43.64 | 31.49 | 9.34% | 23.94 | 36.49 |
| **attempt3.4** | 4 features (LASSO-selected) | 577.90 | 24.04 | 17.85 | 5.67% | 10.80 | 21.48 |

### Ranking by RMSE (lower is better)

1. **attempt3.1** (Low, greedy-selected): **20.70** -- Best overall
2. **attempt3.4** (4 features, LASSO-selected): 24.04 (~1.2x worse than best)
3. **attempt3.2** (20 features, RFECV): 43.63 (~2.1x worse than best)
4. **attempt3.3** (20 features, LASSO): 43.64 -- Worst (~2.1x worse than best)

### Ranking by MAPE (lower is better)

1. **attempt3.1** (Low): **4.90%**
2. **attempt3.4** (4 features, LASSO-selected): 5.67%
3. **attempt3.3** (20 features, LASSO): 9.34%
4. **attempt3.2** (20 features, RFECV): 9.52%

---

## 4. Comparison with Baselines

| Model/Metric | RMSE (USD) | MAPE | Notes |
|-------------|------------|------|-------|
| **Naive baseline** (predict today's Close) | ~8-12 | ~3-4% | Hard to beat on trending data |
| **attempt3.1 (best)** | 20.70 | 4.90% | Single feature, greedy selected, 80/20 split |
| **attempt3.3** | 43.64 | 9.34% | 20 features, LASSO selected, 70/15/15 split |
| **attempt3.4** | 43.64 | 9.34% | 4 features, LASSO selected, 70/15/15 split |
| **attempt2.4 (best overall)** | 20.45 | 4.91% | Manual selection, 80/20 split |

All attempt3 models perform worse than naive baselines, consistent with attempt2 findings. Multi-feature automatic selection (attempt3.2/3.3) performs particularly poorly, ~2x worse than simple methods. attempt3.4 shows LASSO can select effective compact subsets.

---

## 5. Fundamental Differences Between Notebooks

### 3.1 attempt3.1 -- Individual MSE Evaluation + Greedy Forward Selection

**Hypothesis**: Starting from individual feature performance, greedily add features that improve MSE to find an optimal subset.

**Result**: Best performance across all metrics (RMSE=20.70, MAPE=4.90%).

**Why it works**: The greedy forward selection started with Low (best individual MSE=414.51) and stopped after step 1 because no additional feature improved MSE. This confirms that for this dataset and LSTM architecture, a single price-based feature like Low is sufficient and optimal. The simplicity avoids overfitting and noise from correlated features.

### 3.2 attempt3.2 -- Correlation Filter + RFECV

**Hypothesis**: First filter features with low correlation to target (>0.25 threshold), then use Recursive Feature Elimination with Cross-Validation to select optimal subset.

**Result**: Second best (RMSE=43.63, MAPE=9.52%). Significantly worse than 3.1.

**Why it underperformed**: Despite selecting 20 features, the performance is much worse. This suggests that while these features have decent individual correlations, their combination introduces noise or multicollinearity that the LSTM cannot handle effectively. The wrapper method selected features that work well individually but poorly together.

### 3.3 attempt3.3 -- Correlation Filter + LASSO

**Hypothesis**: Use L1 regularization (LASSO) to perform embedded feature selection, penalizing less important features to zero.

**Result**: Third best (RMSE=43.64, MAPE=9.34%). Slightly better than RFECV.

**Why it works better than RFECV**: LASSO (Tibshirani, 1996) performs embedded selection by adding L1 regularization, jointly optimizing prediction and sparsity. This handles correlated features better than RFECV's (Guyon & Elisseeff, 2003) iterative wrapper approach, which may get stuck in local optima. LASSO selected high_low_ratio over some volatility features, potentially reducing multicollinearity noise.

### 3.4 attempt3.4 -- Correlation Filter + LASSO (Selected 4 features)

**Hypothesis**: Use LASSO to select features, potentially finding a different optimal subset.

**Result**: Selected 4 features (Close, EMA_50, BB_width, days_since_start) (RMSE=24.04, MAPE=5.67%), performing much better than attempt3.3.

**Why selected these**: LASSO's L1 penalty (Tibshirani, 1996) shrank irrelevant coefficients to zero, selecting a compact subset mixing price (Close), momentum (EMA_50), volatility (BB_width), and trend (days_since_start) features. This demonstrates embedded methods can find effective, interpretable subsets without exhaustive search.

---

## 6. Key Insights

### 4.1 Feature Engineering Insights

| Finding | Evidence |
|---------|----------|
| **Simple greedy selection outperforms sophisticated methods** | attempt3.1 (greedy, Kohavi & John, 1997) beats all multi-feature approaches |
| **Embedded methods can select compact subsets but may not match greedy performance** | attempt3.4 (LASSO) selected 4 features, similar to attempt3.3 (20 features) but worse than attempt3.1 (1 feature) |
| **More features ≠ better** | All 20-feature models lost badly (~2x worse) to single/4 feature models |
| **Embedded selection (LASSO) slightly better than wrapper (RFECV)** | attempt3.3 (Tibshirani, 1996) better than attempt3.2 (Guyon & Elisseeff, 2003) |
| **Automatic selection with many features leads to overfitting** | Multi-feature models show higher variance (Std Error 34-36 vs 18-19) and bias |
| **Feature redundancy hurts performance** | High correlations among technical indicators (e.g., multiple EMAs) introduce multicollinearity |
| **Bias-variance trade-off in feature selection** | Single/compact subsets reduce variance, avoiding LSTM overfitting |

### 6.1 Feature Engineering Insights

[table as above]

### 6.2 Feature Selection Methodologies Comparison

The attempt3 series explores different feature selection strategies:

- **attempt3.1**: Pure wrapper method (greedy forward, Kohavi & John, 1997) from scratch
- **attempt3.2**: Filter (correlation) + wrapper (RFECV, Guyon & Elisseeff, 2003)
- **attempt3.3/3.4**: Filter (correlation) + embedded (LASSO, Tibshirani, 1996)

Results show that greedy wrapper methods excel at finding optimal single-feature solutions, while embedded methods (LASSO) perform variably depending on subset size—effective for sparsity but not always for performance. Wrappers (RFECV) are computationally intensive and may overfit on limited CV folds.

### 6.3 Limitations and Study Constraints

- **Data split inconsistencies**: Different train/test ratios (80/20 vs 70/15/15) confound direct comparisons; future work should standardize splits.
- **Limited feature diversity**: All features derived from OHLCV; richer data (news sentiment, macroeconomic indicators) might improve results.
- **LSTM hyperparameter sensitivity**: Fixed architecture (100 units, 2 layers) may not be optimal for all feature subsets.
- **Efficient Market Hypothesis**: As noted in attempt2, historical patterns have limited predictive power for stock prices (Fama, 1970).
- **Sample size**: ~2500 samples may be insufficient for robust LSTM training on high-dimensional inputs.

### 4.3 Error Characteristics

All models show similar error profiles to attempt2 series:

- **Positive mean error**: Models tend to underpredict
- **High std error**: Large prediction variance
- **Underprediction bias**: Suggests models struggle with TSLA's upward trends

The single-feature model (3.1) shows better calibration with lower mean and std error.

---

## 5. Differences from Attempt2 Series

The attempt2 series used manual feature selection (OHLCV combinations), while attempt3 series implemented automatic feature selection methods.

| Aspect | Attempt2 | Attempt3 |
|--------|----------|----------|
| Feature selection | Manual intuition | Automatic algorithms |
| Best performance | Close + time (RMSE=20.45) | Low only (RMSE=20.70) |
| Feature count | 1-5 features | 1-20 features |
| Methodology | Domain knowledge | Data-driven |

**Key difference**: Automatic methods selected Low as optimal, while manual selection preferred Close. Both achieve similar performance, validating that price-level features dominate.

---

## 6. Comparison with Baselines

| Model/Metric | RMSE | MAPE | Notes |
|-------------|------|------|-------|
| **Naive baseline** (predict today's Close) | ~8-12 | ~3-4% | Hard to beat on trending data |
| **attempt3.1 (best)** | 20.70 | 4.90% | Single feature, greedy selected, 80/20 split |
| **attempt3.4** | 24.04 | 5.67% | 4 features, LASSO selected, 70/15/15 split |
| **attempt3.3** | 43.64 | 9.34% | 20 features, LASSO selected, 70/15/15 split |
| **attempt2.4 (best overall)** | 20.45 | 4.91% | Manual selection, 80/20 split |

All attempt3 models perform worse than naive baselines, consistent with attempt2 findings.

---

## 7. Recommendations for attempt4

| Priority | Action | Rationale |
|----------|--------|-----------|
| **HIGH** | Standardize data splits for fair comparison | Different splits (80/20 vs 70/15/15) affect metrics |
| **HIGH** | Use automatic methods to find compact subsets | attempt3.4 shows LASSO can select effective 4 features |
| **HIGH** | Validate automatic selection with manual checks | Ensure selected features make domain sense |
| **MEDIUM** | Prefer embedded over wrapper methods | LASSO more stable and finds better subsets than RFECV |
| **MEDIUM** | Implement ensemble feature selection | Combine multiple selection methods for robustness |
| **LOW** | Explore non-price features with better signal | Technical indicators showed poor individual performance |

---

## 8. Conclusion

This study reveals that for LSTM-based stock price prediction, simple automatic selection (greedy/LASSO) outperforms complex methods (RFECV), emphasizing parsimony over quantity (Tibshirani, 1996; Kohavi & John, 1997). The attempt3 series demonstrates the evolution from manual feature selection (attempt2) to algorithmic approaches, showing that embedded methods like LASSO can find compact, effective subsets when data splits are controlled.

Key findings: attempt3.1 (greedy, 80/20 split) achieved the best performance with a single feature. attempt3.4 (LASSO 4 features) performed well (~24 RMSE, ~5.7% MAPE), showing LASSO can select effective compact subsets. attempt3.2 (RFECV) and attempt3.3 (LASSO 20 features) performed poorly (~43.6 RMSE, ~9.3% MAPE), highlighting that subset size matters—too many features introduce multicollinearity/overfitting in LSTMs (Guyon & Elisseeff, 2003).

Future work should standardize splits, combine manual and automatic methods, and explore richer feature sets. Ultimately, predicting stock prices from technical indicators alone remains challenging (Fama, 1970), with all models underperforming naive baselines.

**References**
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.
- Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97(1-2), 273-324.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267-288.

---

*Report generated: 2026-04-25*
*Data period: 2016-01-04 to 2026-03 (2574-2575 trading days)*
*Stock: Tesla Inc. (TSLA)*
