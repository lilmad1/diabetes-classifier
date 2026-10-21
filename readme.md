# Diabetes Classifier

I built this project to learn how to train, compare, and evaluate classification models. It uses the Pima Indians Diabetes dataset and includes a small Streamlit app for trying the trained model.

## What I tried

I compared logistic regression, random forest, and XGBoost. I used 80% of the records for development and kept 20% aside for a final test.

Some measurements were recorded as zero where they should have been missing. I marked those as missing and used a pipeline to keep preprocessing inside each training round.

| Model | Cross-validation accuracy | Recall |
|---|---:|---:|
| Logistic regression | 78.2% | 58.4% |
| Random forest | 76.9% | 58.4% |
| XGBoost | 75.7% | 59.4% |
| XGBoost after tuning | 77.2% | 54.2% |

Logistic regression had the highest average accuracy among the settings I tested. Tuning it didn’t improve that score. Tuning XGBoost improved its accuracy, but reduced its recall.

## Choosing a threshold

I also tried changing the cutoff for predicting diabetes. Lowering it caught more cases, but produced more false alarms.

I chose 0.40 as a compromise for this experiment, then kept it fixed for the final test. It isn’t a medically validated cutoff.

## Final results

On the 154 records held aside for testing, logistic regression achieved:

- **73.4% accuracy**
- **64.8% recall**
- **61.4% precision**

It caught 35 diabetes cases and missed 19. It also flagged 22 records without diabetes and correctly classified the other 78.

The biggest lesson was that accuracy alone hides a lot. A model can get most predictions right while still missing many positive cases.

## Try the app

The project was built with Python 3.12.10 on Windows.

Create an environment:

`python -m venv .venv`

Install the packages:

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Start the app:

`.\.venv\Scripts\python.exe -m streamlit run app.py`

The app loads the saved model from `artifacts/diabetes_model.joblib`. The experiments are in `analysis.ipynb`.

## Limitations

This is a learning project, not a medical diagnostic tool. The dataset is small, covers a specific population, and has many missing measurements. I haven’t tested whether the model works on other populations.

It missed roughly one-third of the diabetes cases in the final test. The app’s score should not be treated as someone’s personal medical risk.

## Data

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The notebook reads the local ZIP archive stored at `data/dataset`.