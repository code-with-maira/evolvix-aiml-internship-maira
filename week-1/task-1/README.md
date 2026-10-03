# Week 1 - Task 1: Titanic Survival EDA

## Overview
Exploratory Data Analysis (EDA) of the Titanic dataset (891 passengers) using Python. The goal is to find out which factors were linked to a passenger's chance of survival.

## Dataset
Titanic dataset from [Data Science Dojo](https://github.com/datasciencedojo/datasets). It is loaded directly from a public URL inside the notebook, so no file upload is needed.

## What I Did
- Loaded the data with pandas
- **Cleaned the data:**
  - Dropped `Cabin` (about 77% missing)
  - Filled 177 missing `Age` values with the median age of each passenger class
  - Filled 2 missing `Embarked` values with the most common port (S)
  - Checked for duplicates and impossible values (none found)
  - Converted `Sex` and `Embarked` to category type
- Calculated summary statistics using `describe()` and `groupby()`
- Created visualizations with seaborn: missing values heatmap, count plots, histogram, bar plots and a correlation heatmap

## Key Insights
1. **Gender was the strongest factor:** 74.2% of women survived compared to only 18.9% of men.
2. **Passenger class mattered:** survival was 63.0% in 1st class, 47.3% in 2nd class and 24.2% in 3rd class.
3. **Fare and class are strongly linked:** `Pclass` and `Fare` have a correlation of -0.55. The average fare fell from 84.15 in 1st class to 13.68 in 3rd class.

## Files
- - [titanic_eda.ipynb](https://github.com/code-with-maira/evolvix-aiml-internship-maira/blob/main/week-1/task-1/titanic_eda.ipynb): full notebook with code, charts and insights: full notebook with code, charts and insights

## Tools Used
Python, pandas, seaborn, matplotlib, Google Colab
