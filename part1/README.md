# Applied AI & ML Essentials — Capstone Project

## Part 1: Heart Attack Risk Data Cleaning and Exploratory Analysis

## Dataset Description

This project uses a structured heart-attack-risk dataset containing 7,000 patient records and 22 columns. The dataset contains demographic, clinical, lifestyle, and diagnostic variables, including age, gender, chest-pain type, resting blood pressure, cholesterol, maximum heart rate, BMI, smoking status, physical activity, family history, diabetes, stress level, and heart-attack risk.

The dataset was selected because it contains more than 500 rows, multiple numeric and categorical variables, missing values, and a numeric target column called `heart_attack_risk`. These characteristics make it suitable for data cleaning, exploratory analysis, correlation analysis, and later supervised machine-learning tasks.

## Dataset Loading and Inspection

The dataset was loaded into a pandas DataFrame using `pd.read_csv()`. The first five rows, inferred column data types, and DataFrame shape were printed.

The original dataset contained 7,000 rows and 22 columns.

## Missing-Value Analysis

Missing-value counts were calculated using:

```python
df.isnull().sum()
```

Missing-value percentages were calculated using:

```python
(df.isnull().sum() / df.shape[0]) * 100
```

The analysis showed that **No columns exceed 20% missing values.
resting_bp Mean: 116.45 Median: 116.0
cholesterol Mean: 184.65 Median: 184.0
fasting_blood_sugar Mean: 0.25 Median: 0.0
max_heart_rate Mean: 173.94 Median: 174.0
oldpeak Mean: 1.27 Median: 1.0
bmi Mean: 27.11 Median: 27.1
stress_level Mean: 5.44 Median: 5.0**.

Numeric columns with missing percentages below or equal to 20% were imputed using their median.

The median was selected instead of the mean because it is less sensitive to extreme values and skewed distributions. In medical datasets, variables such as cholesterol, resting blood pressure, BMI, maximum heart rate, and stress level may contain extreme but valid observations. Such values can pull the mean upward or downward, while the median remains a more stable representation of the centre of the distribution.

Categorical missing values were filled using the most frequent value, or mode, before saving the final cleaned dataset.

## Duplicate Detection and Removal

Duplicate rows were counted using:

```python
df.duplicated().sum()
```

The dataset contained 0 duplicate rows.

Duplicates were removed using:

```python
df.drop_duplicates()
```

A total of 0 rows were removed.

After duplicate removal, the missing-value percentages **DID NOT** change.

## Data-Type Correction

The `patient_id` column was initially stored as an integer. However, it is an identifier rather than a measurable numeric variable. It was converted to string dtype using:

```python
df["patient_id"] = df["patient_id"].astype("string")
```

Repetitive text columns, including gender, chest-pain type, resting ECG, ST slope, thalassemia, smoking status, alcohol consumption, and physical activity, were converted to category dtype.

Memory usage before conversion was **3945851 bytes bytes**, while memory usage after conversion was **1156348 bytes**.

The conversion saved **2789503 bytes**, representing a reduction of approximately **70.69 %**.

Category dtype reduces memory usage because pandas stores each unique category once and represents repeated values using smaller internal category codes.

## Descriptive Statistics and Skewness

Descriptive statistics were generated for all numeric columns using `df.describe()`.

Skewness was calculated for every numeric variable using `df[col].skew()`.

The column with the highest absolute skewness was **oldpeak**, with a skewness value of **1.5059**.

A positive skew means that the distribution has a longer right tail and contains some unusually high values. These high values pull the mean upward.

A negative skew means that the distribution has a longer left tail and contains some unusually low values. These low values pull the mean downward.

Because the mean is affected by extreme observations, median imputation is more appropriate for strongly skewed variables.

## Imputation Strategy Comparison

The two numeric columns with the highest absolute skewness were:

1. **oldpeak**
2. **diabetes**

For **oldpeak**, the mean before imputation was **1.260771**, while the median was **1.0**.

For **diabetes**, the mean before imputation was **0.205714**, while the median was **0.0**.

Median imputation was selected for these columns.

For a positively skewed column, extreme high values pull the mean upward, making the median a more representative measure of central tendency.

For a negatively skewed column, extreme low values pull the mean downward, again making the median more representative.

After median imputation, `isnull().sum()` confirmed that no missing values remained in the two most skewed numeric columns.

## IQR Outlier Analysis

Outlier detection was performed on `cholesterol` and `resting_bp`.

The interquartile range was calculated using:

```text
IQR = Q3 − Q1
Lower bound = Q1 − 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

### Cholesterol

For cholesterol:

* Q1: **164.0**
* Q3: **206.0**
* IQR: **42.0**
* Lower bound: **101.0**
* Upper bound: **269.0**
* Outlier count: **68**

The cholesterol outliers were retained. Extremely high or low cholesterol values may represent real patients and may contain important predictive information about cardiovascular risk.

### Resting Blood Pressure

For resting blood pressure:

* Q1: **108.0**
* Q3: **125.0**
* IQR: **17.0**
* Lower bound: **82.5**
* Upper bound: **150.5**
* Outlier count: **69**

The resting-blood-pressure outliers were also retained because unusual blood-pressure measurements may be clinically meaningful rather than incorrect.

In Part 2, robust scaling and tree-based models will be considered because they are less sensitive to extreme values. Capping will only be considered if the observations are found to be impossible, incorrect, or harmful to model performance.

## Line Plot

The line plot displays maximum heart rate across patient records using the row index.

The graph shows how maximum heart rate varies between patients and helps identify unusually high or low measurements.

The row index does not represent time, so the graph should not be interpreted as a time-series trend.

## Bar Chart

The bar chart compares the mean heart-attack-risk value across chest-pain categories.

Since `heart_attack_risk` is binary, the mean represents the proportion of patients classified as high risk within each chest-pain category.

The category with the highest mean heart-attack risk was **Typical Angina**.

Differences between the bars suggest that chest-pain type may carry predictive information. However, these differences do not establish causation because age, cholesterol, blood pressure, diabetes, exercise response, and other factors may influence the result.

## Histogram

The histogram displays the distribution of **MOST SKEWED COLUM**, which was identified as the numeric column with the highest absolute skewness.

The distribution was **POSITIVELY** skewed.

A longer right tail indicates positive skew, while a longer left tail indicates negative skew.

The visible shape of the histogram agrees with the calculated skewness value. Since the distribution is skewed, median imputation was selected instead of mean imputation.

## Scatter Plot

The scatter plot compares age and maximum heart rate.

The Pearson correlation coefficient was **-0.6093**, indicating a **STRONG NEGATIVE** relationship.

A negative relationship means that maximum heart rate tends to decrease as age increases. A positive relationship means that both variables tend to increase together.

The spread of points shows that age alone does not fully explain maximum heart rate. Other factors such as physical fitness, medication, cardiovascular health, and exercise capacity may also affect the result.

## Box Plot

The box plot compares maximum heart rate between patients with heart-attack-risk values of 0 and 1.

The median maximum heart rate for risk group 0 was **178.0**, while the median for risk group 1 was **169.0**.

The difference in medians suggests that maximum heart rate May* carry predictive information. However, visible overlap and within-group variation indicate that maximum heart rate alone is insufficient for reliably predicting heart-attack risk.

## Pearson Correlation Heat Map

A Pearson correlation matrix was calculated for all numeric columns and displayed using a heat map.

The pair with the highest absolute Pearson correlation was **age** and **max_heart_rate**, with a correlation coefficient of **-0.6093**.

This association does not prove that one variable causes the other.

A third variable may influence both measurements. Plausible alternative explanations include age, physical fitness, medication, diabetes, smoking status, body composition, exercise capacity, or another underlying medical condition.

Therefore, the correlation will be used for exploratory feature-selection guidance rather than as evidence of causation.

## Spearman Rank Correlation

The Spearman rank-correlation matrix was calculated using:

```python
df.corr(method="spearman")
```

Spearman correlation works with ranked values and can detect monotonic relationships that are not necessarily linear.

The three pairs with the largest absolute difference between Spearman and Pearson correlation were:

1. **exercise_angina** and **oldpeak**
2. **resting_bp** and **bmi**
3. **age** and **max_heart_rate**

For **exercise_angina** and **oldpeak**, the Pearson correlation was **0.329246**, while the Spearman correlation was **0.418358**. Since the **Spearman** correlation had the larger absolute value, the relationship appears to be **monotonic but potentially non-linear**. Therefore, **Spearman correlation** will provide better feature-selection guidance for this pair.

For **resting_bp** and **bmi**, the Pearson correlation was **0.448480**, while the Spearman correlation was **0.426843**. Since the **Pearson** correlation had the larger absolute value, the relationship appears to be **approximately linear**. Therefore, **Pearson correlation** will provide better feature-selection guidance for this pair.

For **age** and **max_heart_rate**, the Pearson correlation was **-0.609304**, while the Spearman correlation was **-0.590494**. Since the **Pearson** correlation had the larger absolute value, the relationship appears to be **approximately linear**. Therefore, **Pearson correlation** will provide better feature-selection guidance for this pair.

In Part 2, **Spearman correlation** will be preferred for monotonic but non-linear relationships and variables that may be influenced by outliers. **Pearson correlation** will be preferred when the relationship is approximately linear.

Correlation alone will not be used to select or remove features. Clinical relevance, multicollinearity, feature importance, cross-validation results, and overall model performance will also be considered during feature selection.


## Final Cleaned Dataset

After cleaning, all remaining categorical missing values were filled using their mode.

A final missing-value check was performed to confirm that the cleaned dataset contained no unresolved missing values.

The cleaned dataset was saved using:

```python
df.to_csv("cleaned_data.csv", index=False)
```

The file `cleaned_data.csv` will be used in Parts 2 and 3.

## Limitations

This analysis is exploratory and does not establish clinical causality.

The dataset may contain simulated or observational records, and the results should not be used for real medical diagnosis.

Median and mode imputation can reduce natural variation and may hide patterns associated with missingness.

Correlation measures do not capture all possible interactions between variables.

Outliers were retained because they may represent genuine clinical cases, but their effect on model performance must be evaluated in Part 2.
