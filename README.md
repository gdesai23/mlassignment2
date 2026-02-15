# ML Assignment 2 - Classification Model Comparison

## a) Problem Statement
The objective is to build and compare six classification models on a real-world dataset, evaluate them using six required metrics, and deploy an interactive Streamlit app where users can upload test data and inspect model performance.

## b) Dataset Description
- **Dataset Name:** Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/adult
- **Task Type:** Binary classification (`income` <=50K vs >50K)
- **Instances:** 48,842 rows before cleaning (after removing missing values represented as `?`, remaining rows are used for training/testing)
- **Features:** 14 input features (mix of numerical and categorical), satisfies assignment requirement of at least 12 features
- **Target Column:** `income`

## Project Structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/
|   |-- adult_train.csv
|   |-- adult_test.csv
|-- model/
|   |-- train_models.py
|   |-- model_metrics.csv
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest.pkl
|   |-- xgboost.pkl
```

## c) Models Used With Comparison Table
All models were trained using an 80:20 stratified train-test split with `random_state=42`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8450 | 0.9020 | 0.7344 | 0.5870 | 0.6525 | 0.5601 |
| Decision Tree | 0.8100 | 0.7470 | 0.6154 | 0.6222 | 0.6188 | 0.4922 |
| kNN | 0.8387 | 0.8852 | 0.7007 | 0.6097 | 0.6520 | 0.5500 |
| Naive Bayes | 0.6193 | 0.8389 | 0.3877 | 0.9242 | 0.5462 | 0.3891 |
| Random Forest | 0.8569 | 0.9134 | 0.7908 | 0.5749 | 0.6658 | 0.5895 |
| XGBoost | 0.8690 | 0.9259 | 0.7852 | 0.6490 | 0.7106 | 0.6317 |

## d) Observations Table
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with good AUC and balanced precision-recall; performs consistently on this dataset. |
| Decision Tree | Easy to interpret, but lower AUC and MCC compared to ensemble methods; likely overfits relative to simpler linear model. |
| kNN | Competitive with Logistic Regression in F1, but slightly weaker than ensemble methods in overall discrimination (AUC). |
| Naive Bayes | Very high recall but poor precision and low accuracy; predicts many positives, causing more false positives. |
| Random Forest | Better accuracy, F1, and MCC than single-tree and linear baseline; robust performance due to ensemble averaging. |
| XGBoost | Best overall metrics in this run (Accuracy, AUC, F1, MCC), showing strongest classification quality across metrics. |

## Streamlit App Features Implemented
- CSV upload option for test dataset
- Model selection dropdown (all 6 models)
- Display of required metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report display
- Quick download button for `test_data.csv`

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

GitHub Repository Link: https://github.com/gdesai23/mlassignment2
Live Streamlit App Link: https://mlassignment2-2025aa05217.streamlit.app/


## Notes
- Trained model files (`.pkl`) are included for app inference.
- Core training logic is available in `model/train_models.py`.
- Use your own BITS Lab screenshot and your own repository/app links before final submission.
