# Student Performance Analysis

This Data Science project aims to analyze and predict student exam scores based on various influencing factors. The project compares three different Machine Learning algorithms and highlights the most important features contributing to a student's academic success.

## 📊 Project Features

- **Data Preprocessing**: Handling missing values, scaling for numerical features, and one-hot encoding for categorical features.
- **Machine Learning Model Comparison**: Comparing three models:
  - Linear Regression
  - Random Forest
  - XGBoost
- **Model Evaluation**: Evaluating model performance using Root Mean Squared Error (RMSE) and R-squared ($R^2$) metrics.
- **Feature Importance Analysis**: Automatically extracting the best-performing model and visualizing the most impactful factors on student scores in a bar chart (`feature_importance.png`).

## 🛠️ Prerequisites

Make sure you have [Python](https://www.python.org/) installed on your system. 

You will need to install the required Python libraries listed in the `requirements.txt` file.

Run the following command in your terminal/command prompt:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

1. Ensure the dataset is correctly placed in `Dataset/StudentPerformanceFactors.csv`.
2. Run the main processing script:
```bash
python analyze.py
```
3. The script will train all three algorithms, output the RMSE and R-squared statistics in the terminal, and save the feature importance plot as `feature_importance.png` in the root directory.

## 📁 Repository Structure

- `analyze.py`: The main script containing the logic for data loading, preprocessing, model comparison, and plotting.
- `requirements.txt`: The list of third-party Python dependencies.
- `Dataset/StudentPerformanceFactors.csv`: The dataset file to be analyzed (*make sure to put it here before running the script*).
- `feature_importance.png`: The resulting feature importance graph (*created after a successful run*).
