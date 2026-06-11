# Sephora: From Trees to Boosting Power

Applying the decision-tree-to-ensemble-boosting pipeline from a prior bank-marketing analysis to Sephora product and review data. The goal is to surface **actionable marketing recommendations** — which product attributes, customer skin profiles, and price tiers drive `is_recommended`, high ratings, and purchase intent.

---

## Dataset

| File | Rows | Key columns |
|---|---|---|
| `data/product_info.csv` | 8,494 products | `brand_name`, `price_usd`, `rating`, `loves_count`, `primary_category`, `highlights`, `limited_edition`, `sephora_exclusive` |
| `data/reviews_0-250.csv` … `reviews_750-1250.csv` | ~1.8M reviews total | `is_recommended`, `rating`, `skin_tone`, `skin_type`, `hair_color`, `helpfulness`, `review_text` |

---

## Analysis Scaffold

### 0. Setup
- Merge all review shards into one DataFrame; left-join on `product_id` with `product_info`
- Define the target variable: `is_recommended` (binary) — mirrors the `y` subscription flag from the bank dataset
- Secondary target to explore: `rating >= 4` as a high-satisfaction flag

### 1. Data Preparation

**1a. Preliminary EDA**
- Plot distributions of all numerical features (`price_usd`, `loves_count`, `rating`, `helpfulness`, feedback counts)
- Plot bar charts for categorical features (`skin_tone`, `skin_type`, `primary_category`, `brand_name`, `highlights`)
- Check class balance on `is_recommended` — expect imbalance similar to the bank dataset; use stratified K-Fold
- Identify placeholders: `"None"`, empty strings, and sparse categories in `skin_tone`, `eye_color`, `hair_color`

**1b. Handling missing / placeholder values**
- Create indicator columns (`skin_tone_placeholder`, `skin_type_placeholder`, etc.) for rows where skin/hair attributes are missing — same pattern as `job_placeholder`, `default_placeholder` in the bank pipeline
- Impute or flag `sale_price_usd` and `value_price_usd` nulls (most products have no sale price)

**1c. Feature engineering**
- `discount_rate = (value_price_usd - sale_price_usd) / value_price_usd` where available
- `helpfulness_ratio = total_pos_feedback_count / (total_feedback_count + 1)`
- `review_length = len(review_text)`
- Binary flags already present: `limited_edition`, `new`, `online_only`, `out_of_stock`, `sephora_exclusive`
- Parse `submission_time` → `review_year`, `review_month` for temporal effects

**1d. Encoding**
- **High-cardinality categoricals** (`brand_name`, `primary_category`, `secondary_category`, `highlights`) → **target encoding** (same as `job`, `month`, `day_of_week` in the bank pipeline)
- **Low-cardinality categoricals** (`skin_tone`, `skin_type`, `eye_color`, `hair_color`, `variation_type`) → **one-hot encoding**

**1e. Train / val / test split**
- 70 / 15 / 15 stratified split on `is_recommended`, then wrap in `StratifiedKFold(n_splits=4)`

---

### 2. Baseline: Decision Tree

- `DecisionTreeClassifier(criterion='gini')`
- Grid-search: `max_depth`, `min_samples_split`, `ccp_alpha`
- Plot learning curve (train vs. val accuracy across `max_depth`)
- Report: Accuracy, Precision, Recall, F1, ROC-AUC, AUCPR, top-10 feature importances
- **Marketing lens:** Which features (price tier, skin type, category) show up at the top splits?

---

### 3. Ensemble Boosting

#### 3a. Gradient Boosting (`sklearn`)
- `RandomizedSearchCV` over `learning_rate`, `n_estimators`, `max_depth`, `subsample`
- Plot: training vs. validation log-loss across boosting iterations (per learning rate)
- Plot: bias-variance curve across `n_estimators`

#### 3b. XGBoost
- `objective='binary:logistic'`, `eval_metric='auc'`, early stopping on val set
- Same hyperparameter search as GB
- Plot: training vs. validation AUC (highlight best LR, gray out others)
- Plot: bias-variance across `n_estimators` and across `max_depth × learning_rate`
- Use `plot_importance` for feature importance

#### 3c. LightGBM
- `objective='binary'`; measure wall-clock training time vs. XGBoost
- Pass categorical feature names natively instead of OHE (compare vs. encoded pipeline)
- Same diagnostic plots as GB / XGBoost

---

### 4. Evaluation & Marketing Recommendations

**Metrics table** (reproduce for all four models):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AUCPR |
|---|---|---|---|---|---|---|
| Decision Tree | | | | | | |
| Gradient Boost | | | | | | |
| XGBoost | | | | | | |
| LightGBM | | | | | | |

**Interpretation questions to answer:**
1. Which product categories / price tiers most strongly predict recommendation?
2. Do skin-profile features (`skin_type`, `skin_tone`) materially improve recall for the minority class?
3. Does `sephora_exclusive` or `limited_edition` flag change predicted recommendation likelihood?
4. Which brand clusters score highest on predicted `is_recommended` — target those for co-marketing or loyalty campaigns?
5. What is the marginal value of `review_length` and `helpfulness_ratio` — can Sephora prompt reviewers for richer text to improve future model signal?

---

### 5. Limitations & Next Steps
- Review text is available but unused — a sentiment feature or TF-IDF top-n signal could improve recall on the minority class
- Price is a single scalar; bucketing into tiers (`budget / mid / luxury`) may produce cleaner splits
- Temporal drift: reviews span multiple years; a time-stratified split would test generalization more honestly
- Class imbalance on `is_recommended` should also be addressed with `scale_pos_weight` (XGBoost) or `class_weight='balanced'` (sklearn), not just stratified folds

---

## Requirements

```
pandas numpy scipy matplotlib seaborn
scikit-learn category_encoders xgboost lightgbm
```

## Prior Work
Methodology adapted from *From Dirty Data to Predictive Models* (A1) and *From Trees to Boosting Power* (A2), applied to the UCI Bank Marketing dataset.
