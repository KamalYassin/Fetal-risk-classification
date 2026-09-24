# Final Model Comparison

## Scope

This is the final comparison on `ml_dataset_final_temporal.csv`. The finalized
55-feature representation, 552 patient IDs, labels, leakage exclusions, fixed
split, 30 seeds, and repeated 5x5 folds were unchanged. No feature engineering,
hyperparameter tuning, threshold tuning, model saving, or sequence modelling
was performed.

## Models

- **Logistic Regression:** `C=1, solver="lbfgs", max_iter=2000`
- **Random Forest:** `n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42`
- **XGBoost:** `n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=42`
- **LightGBM:** `n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42`
- **Support Vector Machine:** `kernel="rbf", C=1, gamma="scale", probability=True, random_state=42`

Every pipeline used fold-local median imputation followed by training-only
SMOTE. Logistic Regression and the RBF SVM also used fold-local
`StandardScaler`; tree models were not scaled.

## Fixed test (descriptive only)

| Model | Accuracy | Balanced accuracy | Macro F1 | Path recall | Path precision | False Pathological |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6216 | 0.4108 | 0.4086 | 0.0000 | 0.0000 | 6 |
| Random Forest | 0.6216 | 0.3955 | 0.3908 | 0.0000 | 0.0000 | 3 |
| XGBoost | 0.6396 | 0.4100 | 0.4027 | 0.0000 | 0.0000 | 1 |
| LightGBM | 0.6126 | 0.3806 | 0.3746 | 0.0000 | 0.0000 | 2 |
| Support Vector Machine | 0.6577 | 0.4551 | 0.4344 | 0.0000 | 0.0000 | 0 |

The fixed test was evaluated once and was not used to choose the recommendation.

## Thirty predefined splits

### Macro F1

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.4862 ± 0.0527 | — | — |
| Random Forest | 0.4772 ± 0.0565 | — | — |
| Random Forest minus Logistic Regression | -0.0089 | [-0.0304, +0.0133] | 11/19/0 |
| XGBoost | 0.4608 ± 0.0458 | — | — |
| XGBoost minus Logistic Regression | -0.0254 | [-0.0474, -0.0037] | 11/19/0 |
| LightGBM | 0.4519 ± 0.0525 | — | — |
| LightGBM minus Logistic Regression | -0.0343 | [-0.0596, -0.0096] | 12/18/0 |
| Support Vector Machine | 0.4835 ± 0.0561 | — | — |
| Support Vector Machine minus Logistic Regression | -0.0027 | [-0.0239, +0.0176] | 17/13/0 |

### Balanced accuracy

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.5224 ± 0.0688 | — | — |
| Random Forest | 0.4800 ± 0.0607 | — | — |
| Random Forest minus Logistic Regression | -0.0425 | [-0.0700, -0.0161] | 10/20/0 |
| XGBoost | 0.4597 ± 0.0480 | — | — |
| XGBoost minus Logistic Regression | -0.0627 | [-0.0895, -0.0374] | 7/23/0 |
| LightGBM | 0.4493 ± 0.0509 | — | — |
| LightGBM minus Logistic Regression | -0.0731 | [-0.1009, -0.0468] | 5/25/0 |
| Support Vector Machine | 0.4785 ± 0.0519 | — | — |
| Support Vector Machine minus Logistic Regression | -0.0439 | [-0.0689, -0.0208] | 5/25/0 |

### Pathological recall

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.4000 ± 0.2081 | — | — |
| Random Forest | 0.2389 ± 0.1733 | — | — |
| Random Forest minus Logistic Regression | -0.1611 | [-0.2500, -0.0833] | 3/15/12 |
| XGBoost | 0.2389 ± 0.1431 | — | — |
| XGBoost minus Logistic Regression | -0.1611 | [-0.2333, -0.0944] | 2/18/10 |
| LightGBM | 0.2111 ± 0.1512 | — | — |
| LightGBM minus Logistic Regression | -0.1889 | [-0.2611, -0.1167] | 2/20/8 |
| Support Vector Machine | 0.2333 ± 0.1356 | — | — |
| Support Vector Machine minus Logistic Regression | -0.1667 | [-0.2335, -0.1056] | 0/17/13 |

### Pathological precision

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.2329 ± 0.1743 | — | — |
| Random Forest | 0.2364 ± 0.1610 | — | — |
| Random Forest minus Logistic Regression | +0.0035 | [-0.0580, +0.0689] | 16/12/2 |
| XGBoost | 0.2632 ± 0.1692 | — | — |
| XGBoost minus Logistic Regression | +0.0303 | [-0.0292, +0.0870] | 20/10/0 |
| LightGBM | 0.2454 ± 0.1642 | — | — |
| LightGBM minus Logistic Regression | +0.0125 | [-0.0574, +0.0793] | 18/12/0 |
| Support Vector Machine | 0.3512 ± 0.2212 | — | — |
| Support Vector Machine minus Logistic Regression | +0.1183 | [+0.0397, +0.2033] | 19/11/0 |

### False Pathological predictions

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 9.1333 ± 3.7114 | — | — |
| Random Forest | 4.7333 ± 1.9989 | — | — |
| Random Forest minus Logistic Regression | -4.4000 | [-5.5667, -3.3325] | 27/2/1 |
| XGBoost | 4.4333 ± 1.8511 | — | — |
| XGBoost minus Logistic Regression | -4.7000 | [-5.8333, -3.6667] | 29/1/0 |
| LightGBM | 3.7667 ± 1.3566 | — | — |
| LightGBM minus Logistic Regression | -5.3667 | [-6.6000, -4.2000] | 29/1/0 |
| Support Vector Machine | 2.9333 ± 1.9286 | — | — |
| Support Vector Machine minus Logistic Regression | -6.2000 | [-7.2667, -5.1667] | 29/1/0 |

## Repeated 5x5 cross-validation

### Macro F1

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.4663 ± 0.0559 | — | — |
| Random Forest | 0.4822 ± 0.0739 | — | — |
| Random Forest minus Logistic Regression | +0.0160 | [-0.0121, +0.0449] | 14/11/0 |
| XGBoost | 0.4516 ± 0.0761 | — | — |
| XGBoost minus Logistic Regression | -0.0146 | [-0.0434, +0.0143] | 11/14/0 |
| LightGBM | 0.4562 ± 0.0867 | — | — |
| LightGBM minus Logistic Regression | -0.0101 | [-0.0412, +0.0206] | 13/12/0 |
| Support Vector Machine | 0.4862 ± 0.0833 | — | — |
| Support Vector Machine minus Logistic Regression | +0.0199 | [-0.0062, +0.0464] | 15/10/0 |

### Balanced accuracy

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.5041 ± 0.0803 | — | — |
| Random Forest | 0.4839 ± 0.0772 | — | — |
| Random Forest minus Logistic Regression | -0.0201 | [-0.0527, +0.0102] | 11/14/0 |
| XGBoost | 0.4551 ± 0.0764 | — | — |
| XGBoost minus Logistic Regression | -0.0490 | [-0.0804, -0.0198] | 7/18/0 |
| LightGBM | 0.4541 ± 0.0839 | — | — |
| LightGBM minus Logistic Regression | -0.0500 | [-0.0827, -0.0197] | 5/20/0 |
| Support Vector Machine | 0.4807 ± 0.0763 | — | — |
| Support Vector Machine minus Logistic Regression | -0.0233 | [-0.0500, +0.0015] | 10/15/0 |

### Pathological recall

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.3773 ± 0.2056 | — | — |
| Random Forest | 0.2360 ± 0.1962 | — | — |
| Random Forest minus Logistic Regression | -0.1413 | [-0.2187, -0.0747] | 1/15/9 |
| XGBoost | 0.2320 ± 0.2058 | — | — |
| XGBoost minus Logistic Regression | -0.1453 | [-0.2187, -0.0800] | 1/15/9 |
| LightGBM | 0.2027 ± 0.2050 | — | — |
| LightGBM minus Logistic Regression | -0.1747 | [-0.2573, -0.1013] | 1/17/7 |
| Support Vector Machine | 0.2493 ± 0.1748 | — | — |
| Support Vector Machine minus Logistic Regression | -0.1280 | [-0.2040, -0.0573] | 3/15/7 |

### Pathological precision

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 0.1891 ± 0.0905 | — | — |
| Random Forest | 0.2587 ± 0.2433 | — | — |
| Random Forest minus Logistic Regression | +0.0696 | [-0.0148, +0.1635] | 12/9/4 |
| XGBoost | 0.2541 ± 0.2456 | — | — |
| XGBoost minus Logistic Regression | +0.0650 | [-0.0194, +0.1591] | 15/9/1 |
| LightGBM | 0.2910 ± 0.2997 | — | — |
| LightGBM minus Logistic Regression | +0.1019 | [-0.0028, +0.2157] | 13/11/1 |
| Support Vector Machine | 0.3640 ± 0.2710 | — | — |
| Support Vector Machine minus Logistic Regression | +0.1749 | [+0.0723, +0.2850] | 18/5/2 |

### False Pathological predictions

| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |
|---|---:|---:|---:|
| Logistic Regression | 9.2800 ± 3.3853 | — | — |
| Random Forest | 4.1200 ± 2.1079 | — | — |
| Random Forest minus Logistic Regression | -5.1600 | [-6.7600, -3.4400] | 23/1/1 |
| XGBoost | 3.8800 ± 2.1471 | — | — |
| XGBoost minus Logistic Regression | -5.4000 | [-7.0800, -3.5990] | 23/2/0 |
| LightGBM | 2.8400 ± 1.8184 | — | — |
| LightGBM minus Logistic Regression | -6.4400 | [-7.9200, -4.9200] | 23/0/2 |
| Support Vector Machine | 2.6400 ± 1.6299 | — | — |
| Support Vector Machine minus Logistic Regression | -6.6400 | [-8.0000, -5.2800] | 25/0/0 |

Pairwise differences are model minus Logistic Regression. For false
Pathological predictions, a negative difference is favourable and the
improved/worsened counts use that direction. Confidence intervals crossing zero
are treated as statistical uncertainty, not significance.

## Final ranking

| Rank | Model | Macro F1 | Path recall | Path precision | Balanced accuracy | False Pathological |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Support Vector Machine | 0.4862 | 0.2493 | 0.3640 | 0.4807 | 2.64 |
| 2 | Random Forest | 0.4822 | 0.2360 | 0.2587 | 0.4839 | 4.12 |
| 3 | Logistic Regression | 0.4663 | 0.3773 | 0.1891 | 0.5041 | 9.28 |
| 4 | LightGBM | 0.4562 | 0.2027 | 0.2910 | 0.4541 | 2.84 |
| 5 | XGBoost | 0.4516 | 0.2320 | 0.2541 | 0.4551 | 3.88 |

The ranking is lexicographic: repeated-CV Macro F1, then Pathological recall,
Pathological precision, and balanced accuracy. Stability across the 30 splits,
false alarms, paired uncertainty, and simplicity were then used for the final
recommendation.

## Recommendation

**Support Vector Machine.** Support Vector Machine ranked first in repeated-CV Macro F1 (0.4862); its paired Macro F1 comparison with Logistic Regression had 95% CI [-0.0062, +0.0464]; versus Logistic Regression it changed Pathological recall by -0.1280, precision by +0.1749, and false Pathological predictions by -6.64; across the 30 predefined splits its Macro F1 was 0.4835 versus 0.4862 for Logistic Regression; 30-split Macro F1 SD was 0.0561, Pathological recall was 0.2493, precision was 0.3640, and mean false Pathological predictions were 2.64.

This recommendation follows the requested primary Macro-F1 ranking. The paired
tables must still be used to judge its recall, precision, and false-alarm
trade-offs. If the deployment objective prioritizes sensitivity over precision
and false alarms, the model with stronger Pathological recall may be the more
appropriate operational alternative.

No trained artifact was saved or replaced. The result does not establish
clinical significance.
