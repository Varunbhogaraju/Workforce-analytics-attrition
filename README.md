# Workforce Analytics & Employee Attrition Prediction

A fresher-oriented end-to-end data science project designed around an Associate Data Scientist workflow.

## Business Problem

Employee attrition creates hiring costs, productivity loss, and workforce planning challenges. This project explores employee data, identifies workforce patterns, and builds classification models to estimate attrition risk.

## Workflow

1. Data quality assessment
2. Exploratory data analysis
3. Statistical hypothesis testing
4. Feature engineering
5. Logistic Regression baseline
6. Random Forest comparison
7. Model evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC
8. 5-fold cross-validation
9. Model interpretation with permutation importance
10. Optional LLM/NLP employee-feedback analysis

## Run

Place the dataset at:

`data/raw/employee_attrition.csv`

Then:

```bash
python src/train_pipeline.py
```

Outputs are generated under `reports/`.

## Important interview points

- Accuracy can be misleading when the target is imbalanced.
- Recall is useful when missing a genuinely at-risk employee is costly.
- Precision/recall trade off against each other.
- Statistical association does not establish causation.
- Cross-validation estimates how consistently the model generalizes.
- Feature importance indicates predictive contribution, not causal effect.
- Class weighting is used to reduce the impact of class imbalance.
- The preprocessing steps are inside the sklearn pipeline to avoid train/test leakage.

## Ethical consideration

Attrition prediction should support workforce planning and employee support, not automatic adverse employment decisions. Predictions should be treated as risk signals rather than facts about individual employees.
