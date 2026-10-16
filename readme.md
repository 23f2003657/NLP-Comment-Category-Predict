# Comment Category Prediction Challenge: Pipeline & Model Report

This repository contains the end-to-end machine learning pipeline developed for the **Comment Category Prediction Challenge** on Kaggle. The project focuses on multiclass text classification across 4 classes using a combination of comment text, engagement statistics, and user metadata.

---

## 1. Project Overview & Architecture

* **Dataset Scale:** 198,000 training records and 102,000 test records.
* **Task:** Multiclass text classification (4 distinct categories: `0`, `1`, `2`, and `3`).
* **Pipeline Structure:**
  1. **EDA & Visualizations:** Class imbalance analysis, comment character length evaluation, metadata distribution checks, and missing value auditing.
  2. **Preprocessing & Feature Engineering:** Text vectorization via TF-IDF with n-grams, demographic categorical encoding via One-Hot Encoding, missing value imputation, and numeric scaling via `StandardScaler`.
  3. **Modeling:** Progression from linear baselines (`SGDClassifier`, `LogisticRegression`) to gradient-boosted decision trees (`LightGBM`).
  4. **Inference & Post-Processing:** Aligning test feature transformations, preventing target leakage, and generating index-mapped submission files.

---

## 2. Exploratory Data Analysis (EDA) & Statistical Insights

Key statistical findings derived from the training data (`train.csv`):

* **Severe Class Imbalance:**
  * **Label 0:** 114,173 samples (~57.7%)
  * **Label 2:** 62,440 samples (~31.5%)
  * **Label 1:** 15,918 samples (~8.0%)
  * **Label 3:** 5,469 samples (~2.8%)
  * *Strategy:* Required class-weighted cost functions (`class_weight='balanced'`) to prevent gradient updates from ignoring minority classes.
* **Missing Value Auditing:**
  * Categorical demographic tags (`race`, `religion`, `gender`) contained 145,423 missing values (~73.4%).
  * Text comments contained 1 missing value, which was imputed with empty strings prior to tokenization.
* **Numerical Distribution & Heavy Tails:**
  * Engagement metrics (`upvote`, `downvote`, interaction flags `if_1`, `if_2`) exhibited heavy skewness and extreme outliers (e.g., `if_1` max is 1860 with a mean of 1.91; `upvote` max is 201 with a 75th percentile of 3).
  * Scaling was required to stabilize variance and prevent high-magnitude features from dominating linear models.
* **Text Length Distributions:**
  * Kernel Density Estimation (KDE) demonstrated consistent comment length distributions across train and test splits, verifying no major structural text drift between sets.

---

## 3. Feature Engineering & Preprocessing Pipeline

To eliminate data leakage, all vectorizers, encoders, and scalers were fitted **strictly on the training data** and used only in `.transform()` mode on the test data.

1. **Text Features (TF-IDF):**
   * Extracted unigrams and bigrams (`ngram_range=(1, 2)`) to capture contextual compound phrases (e.g., *"not good"*).
   * Filtered using document frequency thresholds (`min_df=5`) to suppress single-occurrence typos and noise.
2. **Categorical Features (One-Hot Encoding):**
   * Encoded demographic attributes (`race`, `religion`, `gender`) via `OneHotEncoder(handle_unknown='ignore')`.
   * Boolean indicators (e.g., `disability`) were converted into binary integer features (`0` / `1`).
3. **Numeric Features (StandardScaler):**
   * Normalized continuous metrics (`upvote`, `downvote`, `emoticon_1`, `emoticon_2`, `emoticon_3`, `if_1`, `if_2`) using `StandardScaler()`.
4. **Feature Fusion:**
   * Sparse TF-IDF matrices and dense scaled numeric/OHE matrices were concatenated into a unified feature matrix using `scipy.sparse.hstack`.

---

## 4. Machine Learning Models & Results

The primary evaluation metric was **Macro-averaged F1 Score** due to the severe class imbalance.

| Model | Configuration / Key Hyperparameters | Macro F1 Score | Notes |
| :--- | :--- | :--- | :--- |
| **SGDClassifier** | Linear SVM (`loss='hinge'`, default parameters) | **0.63** | Fastest training time; struggled on minority classes without class weighting. |
| **Logistic Regression** | Linear classifier, L2 regularization | **0.67** | Strong baseline; better class probability calibration than default SGD. |
| **LightGBM (Baseline)** | GBDT, default tree parameters, TF-IDF unigrams | **0.6341** | Addressed initial 1.0 F1 target leakage artifact; realistic baseline. |
| **LightGBM (Tuned)** | `ngram_range=(1, 2)`, `n_estimators=500`, `num_leaves=63` | **0.6470** | Bigrams captured key multi-word sentiment and toxic phrases. |
| **LightGBM (Optimized)**| GBDT, `class_weight='balanced'`, expanded vocabulary | **0.83** | **Best model.** Leaf-wise tree growth captured non-linear interactions across text and metadata. |

---

## 5. Engineering Challenges & Pipeline Fixes

* **Eliminating Target Leakage:** Early iterations produced an artificial 1.0 F1 score caused by inadvertent leakage of target-correlated features. Decoupling all preprocessors and fitting strictly on train splits restored valid evaluation metrics.
* **Submissions Alignment (Row Count & ID Mappings):**
  * The dataset's `post_id` column contained repeating thread IDs (e.g., ID `123` appearing across multiple rows), causing duplicate ID rejection errors on the platform.
  * Applying `drop_duplicates(subset='ID')` slashed valid test entries from 102,000 down to 45 rows.
  * Inspection of `Sample.csv` confirmed the platform expected a 1-indexed contiguous integer sequence (`ID: 1 to 102,000`). The mapping was resolved by using an explicit sequence range:

```python
# Final submission generation matching exact test requirements
submission = pd.DataFrame({
    'ID': range(1, len(test_preds) + 1),
    'label': test_preds
})

# Verify constraints before saving
assert len(submission) == 102000, "Row count mismatch!"
assert submission['ID'].nunique() == 102000, "Duplicate IDs detected!"

submission.to_csv('submission.csv', index=False)
