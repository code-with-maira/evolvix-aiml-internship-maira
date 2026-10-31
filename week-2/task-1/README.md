# Week 2 - Task 1: Model Comparison on Titanic Survival

## Overview
Trained and compared three classification models to predict whether a Titanic passenger survived, using a confusion matrix, precision, recall, F1-score and ROC-AUC. All three ROC curves are plotted on one chart for a direct comparison.

## Dataset
Titanic dataset (891 passengers) from [Data Science Dojo](https://github.com/datasciencedojo/datasets). The data is split 80/20 with stratification (712 training rows, 179 test rows).

## Models
- **Logistic Regression** (median imputation + feature scaling)
- **Decision Tree** (max_depth = 4)
- **Random Forest** (200 trees, max_depth = 6)

Missing `Age` values are filled inside each model pipeline using the median of the training data only, which avoids data leakage.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV ROC-AUC (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.843 | 0.848 |
| Decision Tree | 0.788 | 0.844 | 0.551 | 0.667 | 0.821 | 0.842 |
| Random Forest | 0.793 | 0.808 | 0.609 | 0.694 | 0.843 | 0.868 |

## Recommendation
I would deploy the **Random Forest**. It has the highest cross-validated ROC-AUC (0.868), ties for the best ROC-AUC on the test set (0.843) and is more stable than a single decision tree. Logistic Regression is almost as good and easier to explain, so it is a strong alternative if interpretability matters more.

## Key Findings
- The three models are close on a single test split, so 5-fold cross-validation was used to confirm the ranking.
- The Decision Tree has the highest precision (0.844) but the lowest recall (0.551), meaning it misses many survivors.
- The most important features are `Sex`, `Fare`, `Age` and `Pclass`, which matches the Week 1 EDA.

## Live Demo
[Open the Streamlit app](https://YOUR-APP-LINK.streamlit.app)

## Files
- [titanic_models.ipynb](titanic_models.ipynb): full notebook with confusion matrices, ROC curves and conclusion
- [app.py](app.py): Streamlit app to compare the models and predict survival
- [requirements.txt](requirements.txt): libraries needed to run the app

## Tools
Python, pandas, scikit-learn, seaborn, matplotlib, Streamlit, Google Colab
