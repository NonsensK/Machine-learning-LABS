# Data Card — UCI Adult (Census Income)

## Source and purpose

UCI Machine Learning Repository, **Adult / Census Income** dataset; loaded from OpenML as `fetch_openml(name='adult', version=2, as_frame=True)`. Original data were extracted from the 1994 US Census database. Used here for a teaching exercise in binary income classification and leakage-free data engineering.

## License and attribution

The UCI Adult dataset is distributed through the UCI Machine Learning Repository, which lists a **CC BY 4.0** license for the dataset. Attribute UCI and the original creators when redistributing. Check the current repository record for any updated terms: https://archive.ics.uci.edu/dataset/2/adult . OpenML may also display its own dataset metadata.

## Size and schema

48,842 observations; 15 columns: 14 predictors plus `income` (`<=50K` or `>50K`). Numerical: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`. Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`. The missing-value token `?` is treated as missing, not as a substantive category in model fitting.

## Processing and splits

Stratified 70% train / 30% test split, seed 42. Remove exact duplicate feature rows from train only. Fit categorical mode imputation, numeric median imputation, training-set percentile clipping, standardization, and one-hot encoding on train only. Derived attributes: `capital_net = capital-gain - capital-loss`, `hours_per_age = hours-per-week / (age - 16)`. Train a logistic regression baseline with and without these two features. Validation, audit, metric, and SHA256 artifacts are generated on execution.

## Known limitations and biases

Data reflect US census records from 1994 and may not generalize to today's populations, incomes, or labor markets. `race`, `sex`, `native-country`, and other features may correlate with protected characteristics or structural inequality. The USD 50,000 threshold is historical and not inflation-adjusted. Dataset labels may contain errors or ambiguous cases. Class imbalance affects metric interpretation. Predictive performance does not establish fairness.

## Privacy and intended use

Although this is a widely published research dataset, records contain potentially sensitive demographic and employment information. Avoid attempts to re-identify individuals or combine rows with external personal data. Use only for education and benchmarking; do not deploy for hiring, credit, eligibility, or other consequential individual decisions without appropriate legal, ethical, fairness, and privacy review.

## Quality and reproducibility

See `results/validation_report.json` for type, range, missingness, and duplicate counts; `results/label_audit.json` for conflicting labels and the 50-row audit sample; `results/metrics.json` for test performance; `results/run_metadata.yaml` for package versions and run settings; and `results/processed_train.sha256` for the canonical processed training CSV hash. The 50-row sample receives automated checks; independent human inspection against the data dictionary is still recommended.

## References

- UCI Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
- OpenML dataset: https://www.openml.org/search?type=data&status=active&id=1590
