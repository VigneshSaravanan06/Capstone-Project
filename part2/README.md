# Applied AI & ML Essentials — Capstone Project

## Part 2: Supervised Machine Learning Model

## Project Overview

This project uses the cleaned Heart Attack dataset created in Part 1 to build and evaluate one regression model and one binary classification model.

The project includes:

* Categorical feature encoding
* Leak-free train-test splitting and scaling
* Linear Regression
* Ridge Regression
* Logistic Regression
* Class-imbalance handling
* Confusion-matrix analysis
* ROC curve and AUC
* Decision-threshold sensitivity
* Logistic Regression regularization comparison
* Bootstrap confidence interval for the AUC difference

## Dataset

The dataset was loaded from:

```text
cleaned_data.csv
```

It contains demographic, clinical, diagnostic, and lifestyle variables related to patient health and heart-attack risk.

## Label Definitions

### Regression Label

The regression label was:

```text
cholesterol
```

`cholesterol` is a continuous numeric medical measurement. The regression models attempt to predict a patient's cholesterol level from the remaining features.

### Classification Label

The classification label was:

```text
heart_attack_risk
```

This is a natural binary target:

* `0` represents the lower-risk class.
* `1` represents the higher-risk class.

Since the dataset already contains a natural binary target, it was not necessary to create a classification label by binarizing the regression target.

## Feature Matrix

The feature matrix contained all useful predictor columns except:

* `cholesterol`
* `heart_attack_risk`
* `patient_id`

`cholesterol` and `heart_attack_risk` were excluded because they were the target variables.

`patient_id` was excluded because it is only an identifier and does not contain meaningful predictive information.

## Categorical Encoding

### Ordered Category

`physical_activity` has a natural order:

```text
Low < Moderate < High
```

It was mapped as:

```text
Low = 0
Moderate = 1
High = 2
```

This ordering preserves the meaningful progression from lower to higher physical activity.

### Unordered Categories

Other categorical variables, such as gender, chest-pain type, resting ECG, ST slope, thalassemia, smoking status, and alcohol consumption, do not have a meaningful numerical order.

They were encoded using one-hot encoding with:

```python
pd.get_dummies(drop_first=True)
```

The first dummy category was dropped to reduce perfect multicollinearity.

One-hot encoding was chosen because assigning integer labels to nominal categories would introduce a false ordinal relationship. For example, assigning arbitrary numbers to different chest-pain categories could incorrectly suggest that one category is numerically greater than another.

## Train-Test Split

The dataset was divided into:

* 80% training data
* 20% test data

The split used:

```python
train_test_split(
    test_size=0.2,
    random_state=42
)
```

Stratification was applied using the classification target to preserve approximately the same class distribution in the training and test sets.

## Leak-Free Standard Scaling

`StandardScaler` was fitted only on the training features:

```python
scaler.fit(X_train)
```

The fitted scaler was then used to transform both the training and test features.

Fitting the scaler on the full dataset would cause data leakage because the scaler would use the test set's means and standard deviations during training. This would allow information from the unseen test set to influence model preprocessing and could produce unrealistically optimistic evaluation results.

## Linear Regression

A `LinearRegression` model was trained using the scaled training features.

The results were:
Mean Squared Error: **853.0717**
R²: **0.1410**

MSE measures the average squared difference between the actual and predicted cholesterol values. Lower MSE indicates smaller prediction errors.

R² measures the proportion of variation in cholesterol explained by the model. An R² closer to 1 indicates stronger explanatory performance. A value near zero indicates that the model performs similarly to predicting the average target value. A negative R² indicates performance worse than the mean baseline.

## Linear Regression Coefficients

The three features with the largest absolute coefficients were:

1. **age** — coefficient: **10.3477**
2. **bmi** — coefficient: **5.8498**
3. **diabetes** — coefficient: **1.1506**

Because the input features were standardized, a one-unit increase means a one-standard-deviation increase in the original feature.

A large positive coefficient means that increasing the scaled feature by one unit is associated with an increase of approximately the coefficient value in predicted cholesterol, while holding the other features constant.

A large negative coefficient means that increasing the scaled feature by one unit is associated with a decrease of approximately the absolute coefficient value in predicted cholesterol, while holding the other features constant.

These coefficient interpretations describe associations and do not prove causation.

## Ridge Regression

A Ridge Regression model was trained with:

```python
Ridge(alpha=1.0)
```

The comparison was:
| Model             |          MSE |         R² |
| ----------------- | -----------: | ---------: |
| Linear Regression | **853.0717** | **0.1410** |
| Ridge Regression  | **853.0714** | **0.1410** |


Ridge Regression adds an L2 penalty to the ordinary least-squares objective. This penalty discourages very large coefficient values and can produce a more stable coefficient profile when predictors are correlated.

The `alpha` parameter controls the strength of regularization:

* A small `alpha` makes Ridge similar to ordinary Linear Regression.
* A larger `alpha` applies stronger coefficient shrinkage.
* Extremely large values may underfit the data.

On this dataset, Ridge Regression produced almost identical performance to ordinary Linear Regression. The difference in MSE and R² was negligible, indicating that regularization had very little impact.
## Class-Imbalance Analysis

The classification training-set class distribution before balancing was:

| Class |    Count | Percentage |
| ----- | -------: | ---------: |
| 0     | **3248** |  **58.0%** |
| 1     | **2352** |  **42.0%** |


The minority class represented **42.0%** of the training samples.
Because this was above the 35% threshold, resampling was not required.
When used, SMOTE was applied only to the training set. It was not applied to the test set because the test set must preserve the original real-world class distribution for unbiased evaluation.

The class counts after handling imbalance were:

| Class |   Before |    After |
| ----- | -------: | -------: |
| 0     | **3248** | **3248** |
| 1     | **2352** | **2352** |


## Baseline Logistic Regression

The baseline model used:

```python
LogisticRegression(
    C=1.0,
    max_iter=1000
)
```

The results were:

Accuracy: **0.7821**
Precision: **0.7675**
Recall: **0.6905**
F1-score: **0.7269**
AUC: **0.8662**

## Confusion Matrix

The confusion matrix was:

```text
[[689, 123],
 [182, 406]]
```

For this model:

True Negatives: **689**
False Positives: **123**
False Negatives: **182**
True Positives: **406**

A false negative occurs when a genuinely high-risk patient is predicted as low risk. A false positive occurs when a lower-risk patient is predicted as high risk.

## Precision and Recall

Precision is calculated as:

```text
Precision = TP / (TP + FP)
```

Precision measures the proportion of positive predictions that were correct.

Recall is calculated as:

```text
Recall = TP / (TP + FN)
```

Recall measures the proportion of actual positive cases correctly identified.

For heart-attack-risk prediction, recall is generally more important than precision because false negatives can be more costly than false positives. Missing a genuinely high-risk patient may delay further investigation or treatment, while a false positive may result in additional screening.

This model should not be used for real medical diagnosis without clinical validation.

## ROC Curve and AUC

The baseline Logistic Regression AUC was:

0.8662

The ROC curve shows the relationship between the true-positive rate and false-positive rate across different decision thresholds.

AUC represents the model's ability to rank positive cases above negative cases:

* `0.50` indicates performance similar to random guessing.
* Values above `0.50` indicate some discriminatory ability.
* Values closer to `1.00` indicate stronger class separation.

An AUC of **0.8662** indicates that the model has good ability to distinguish between patients with high and low heart attack risk.
## Decision-Threshold Sensitivity

Predicted probabilities were generated using:

```python
model.predict_proba(X_test_scaled)[:, 1]
```

The following thresholds were tested:

| Threshold |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: |
|      0.30 | **0.6757** | **0.8469** | **0.7517** |
|      0.40 | **0.7150** | **0.7721** | **0.7424** |
|      0.50 | **0.7675** | **0.6905** | **0.7269** |
|      0.60 | **0.8157** | **0.6020** | **0.6928** |
|      0.70 | **0.8799** | **0.4983** | **0.6363** |


The threshold that maximized F1-score was:

```text
0.30```

Its F1-score was:

```text
0.7517```

For this heart-attack-risk task, recall is more important because false negatives may allow high-risk patients to be missed.

To increase recall, the decision threshold should generally be lowered. Lowering the threshold causes more patients to be classified as positive, which reduces false negatives.

The cost of lowering the threshold is reduced precision because more lower-risk patients may be incorrectly predicted as high risk, increasing false positives.

## Logistic Regression Regularization Experiment

A second Logistic Regression model was trained using stronger L2 regularization:

```python
LogisticRegression(
    C=0.01,
    max_iter=1000
)
```

The comparison was:

| Model                        |  Precision |     Recall |   F1-score |        AUC |
| ---------------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression (C=1.0)  | **0.7675** | **0.6905** | **0.7269** | **0.8662** |
| Logistic Regression (C=0.01) | **0.7743** | **0.6769** | **0.7223** | **0.8670** |


The `C` parameter controls the inverse strength of regularization:

* A larger `C` means weaker regularization.
* A smaller `C` means stronger regularization.
* Stronger regularization shrinks coefficient values more aggressively.
* Excessively strong regularization can cause underfitting.

Reducing C from 1.0 to 0.01 had very little effect on overall performance. The strongly regularized model achieved slightly higher precision and AUC but slightly lower recall and F1-score. Overall, both models performed similarly.
## Bootstrap Confidence Interval for AUC Difference

The bootstrap experiment compared:

```text
AUC(C=1.0) − AUC(C=0.01)
```

A total of 500 bootstrap samples were drawn from the test set using sampling with replacement.

The results were:

Mean AUC difference: **-0.000783**
2.5th percentile: **-0.002180**
97.5th percentile: **0.000820**
95% confidence interval: **(-0.002180, 0.000820)**

The confidence interval includes zero.

If the interval excludes zero, the performance advantage of one model is likely consistent across different samples.

If the interval includes zero, the observed difference may not be reliable and may be due to sampling variation.

For this dataset, the confidence interval includes zero, indicating that the difference in AUC between the Logistic Regression models (C=1.0 and C=0.01) is not statistically reliable. Both models perform almost equally well, and neither model demonstrates a consistent performance advantage across different bootstrap samples.
## Generated Files

The notebook generates:

```text
plots/
├── confusion_matrix.png
└── roc_curve.png
```

It also generates:

```text
results/
├── linear_regression_coefficients.csv
├── regression_comparison.csv
├── class_balance_comparison.csv
├── threshold_comparison.csv
├── logistic_regularization_comparison.csv
├── bootstrap_auc_difference.csv
└── part2_summary_results.csv
```

## How to Run

1. Open `Heart_Attack_Part2.ipynb` in Google Colab.
2. Install `imbalanced-learn` when prompted by the first cell.
3. Upload `cleaned_data.csv`.
4. Select **Runtime → Run all**.
5. Verify that all cells finish without errors.
6. Confirm that the plot and result files are generated.

## Dependencies

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## Limitations

The models demonstrate supervised-learning techniques for an educational project and must not be used for real clinical diagnosis.

The regression model may have limited predictive performance if cholesterol is only weakly associated with the available features.

Linear and Logistic Regression assume relatively simple relationships and may not capture complex interactions.

SMOTE creates synthetic training observations and may not perfectly represent real patients.

AUC and classification metrics are based on one test split and may vary with a different sample.

Correlation, model coefficients, and feature importance indicate associations rather than medical causation.

