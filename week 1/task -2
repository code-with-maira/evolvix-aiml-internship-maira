# Week 1 · Task 2: First Machine Learning Model

## What the model predicts
Whether a Titanic passenger **survived** (1) or **died** (0), using Logistic Regression (scikit-learn). This builds on my [Task 1 EDA](https://github.com/code-with-maira/evolvix-aiml-internship-maira/tree/main/week-1/task-1) of the same dataset (891 passengers).

## Files
- `titanic_logreg_model.ipynb`: full notebook (split, train, evaluate, save)
- `titanic_logreg_model.joblib`: saved model (preprocessing + Logistic Regression)

## Method
- **Features:** Pclass, Age, SibSp, Parch, Fare, Sex, Embarked. I dropped PassengerId, Name, Ticket and Cabin.
- **Split:** 80/20 train/test, `random_state=42`, stratified on `Survived`.
- **No data leakage:** I split the data first. Filling missing values (median Age, most common Embarked), scaling and one-hot encoding are all inside a scikit-learn `Pipeline`, so they are learned from the training set only. In Task 1 I filled Age using the whole dataset, which is fine for EDA but would leak test information into a model.
- **Baseline:** a dummy model that always predicts "died", to check that the model really learns something.

## Results (test set, 179 passengers)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Always predict "died" (baseline) | 0.615 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | **0.804** | **0.793** | **0.667** | **0.724** |

5-fold cross-validation accuracy on the training set: **0.796 ± 0.019**, close to the test score, so the result is stable.

## What the metrics mean
- **Accuracy:** the share of all passengers predicted correctly. It can mislead when classes are unbalanced, so I compare it with the baseline.
- **Precision:** when the model says "survived", how often it is right.
- **Recall:** of the people who really survived, how many the model found.
- **F1:** one number that balances precision and recall.

## Takeaways
The model beats the baseline by about 19 percentage points of accuracy. It is more precise (0.79) than complete (recall 0.67): when it predicts survival it is usually right, but it misses about a third of the real survivors. Sex and passenger class are the strongest signals, which matches what I found in Task 1.

## How to load the model
```python
import joblib
model = joblib.load("titanic_logreg_model.joblib")
model.predict(new_passengers_df)  # DataFrame with the 7 feature columns
```
