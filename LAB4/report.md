# LAB 4 Report — Data Pipeline and Quality

**Course:** Machine Learning  
**Laboratory:** LAB 4  
**Dataset:** UCI Adult (Census Income), OpenML Adult version 2  
**Student ID:** 72119

## 1. Objective

The objective of this laboratory was to implement a reproducible, leakage-aware data processing and quality validation pipeline for binary income classification. The task predicts whether annual income exceeds USD 50,000 using the UCI Adult dataset. In addition to evaluating a predictive model, the laboratory focuses on schema validation, missing-value handling, duplicate auditing, train-only preprocessing, feature engineering, and reproducible output artifacts.

## 2. Dataset and Problem Definition

The Adult dataset contains **48,842 records and 15 columns**: 14 input features and one binary target (`income`, `<=50K` or `>50K`). It combines numerical attributes such as age, capital gain, capital loss, and weekly working hours with categorical demographic, education, and employment attributes.

The data originate from the 1994 US Census and are used here for educational benchmarking, not consequential individual decisions.

## 3. Data Quality Validation

The generated `results/validation_report.json` records the following checks:

| Check | Recorded result |
|---|---:|
| Rows | 48,842 |
| Columns | 15 |
| Schema violations | 0 for expected columns |
| Range violations (age, capital gain/loss, hours/week) | 0 |
| Missing values in `workclass` | 2,799 |
| Missing values in `occupation` | 2,809 |
| Missing values in `native-country` | 857 |
| Duplicate complete rows | 52 |

Missing tokens (`?`) were treated as missing values. The pipeline reports quality issues rather than silently discarding records during initial validation.

## 4. Methodology

### 4.1 Train/test split

A stratified **70%/30%** split was performed with random seed **42**, before fitting preprocessing steps. The held-out test set contains **14,653 rows** and was not modified by training-set duplicate removal. The initial training set contained **34,189 rows**; **26 duplicate feature rows** were removed, leaving **34,163 training rows**.

### 4.2 Preprocessing

The pipeline uses training-fitted transformations to reduce information leakage:

- Numerical missing values: median imputation.
- Categorical missing values: most-frequent-category imputation.
- Numerical outlier handling: clipping at the training set's 1st and 99th percentiles.
- Numerical scaling and categorical one-hot encoding.

The fitted preprocessing objects are saved as `results/preprocessor.joblib`.

### 4.3 Models and feature engineering

A logistic regression classifier serves as the baseline. A second logistic regression model uses two engineered features:

- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

Both approaches are evaluated on the same held-out test set. The primary reported metrics are accuracy and positive-class F1-score.

## 5. Experimental Results

The following numbers are taken directly from the supplied `results/metrics.json` file.

| Model | Test accuracy | Positive-class F1 |
|---|---:|---:|
| Logistic regression baseline | 0.854160 (85.42%) | 0.660740 |
| Logistic regression with engineered features | 0.858118 (85.81%) | 0.670575 |
| **Absolute improvement** | **+0.003958 (+0.40 percentage points)** | **+0.009835** |

The engineered-feature model improved both reported metrics. The increase is modest, but consistent across accuracy and F1 in this recorded run. The comparison plot is stored in `results/model_comparison.png`.

## 6. Label Audit and Reproducibility

The project includes a label-consistency audit (`results/label_audit.json`) and a 50-row review sample (`results/label_review_50.csv`). Automated checks are included, but the project documentation states that independent human review of the sampled rows against the official data dictionary remains outstanding.

Reproducibility artifacts include:

- `results/processed_train.csv` — processed training dataset.
- `results/processed_train.sha256` — SHA256 checksum for the exported training data.
- `results/run_metadata.yaml` — configuration and software versions.
- `results/preprocessor.joblib` and `results/model.joblib` — fitted pipeline artifacts.
- `results/validation_report.json` and `results/metrics.json` — recorded quality checks and performance.

The recorded environment used Python **3.13.0**, NumPy **2.5.4**, pandas **2.3.3**, and scikit-learn **1.9.1**. The configuration uses seed **42**, test fraction **0.30**, and clipping quantiles **0.01** and **0.99**.

## 7. Discussion and Limitations

The pipeline demonstrates a structured approach to validation and train-only fitting. In the supplied results, engineered features improved held-out performance, although the magnitude is small and no repeated-run variability or confidence intervals were reported. These results therefore do not establish that the engineered model will outperform the baseline across other splits or populations.

The dataset is historical, with a non-inflation-adjusted income threshold. Its demographic attributes may reflect social inequalities and raise fairness concerns. Accuracy and F1 alone are not evidence of fairness or suitability for deployment. The automated 50-row label audit does not replace manual review.

## 8. Conclusion

LAB 4 produced a reproducible data-quality and preprocessing workflow for the UCI Adult income classification task. The recorded run validated the 48,842-row dataset, identified missing values and duplicates, preserved the test set during training deduplication, and compared baseline logistic regression with an engineered-feature variant. The engineered model achieved **85.81% accuracy** and **0.6706 F1**, compared with **85.42% accuracy** and **0.6607 F1** for the baseline. The remaining documented quality task is manual review of the 50 sampled audit rows.

## 9. References

1. UCI Machine Learning Repository — Adult: https://archive.ics.uci.edu/dataset/2/adult
2. OpenML Adult dataset: https://www.openml.org/search?type=data&status=active&id=1590
3. Project artifacts: `README.md`, `problem_definition.md`, `data_card.md`, `config.yaml`, and `results/` in the supplied LAB4 archive.
