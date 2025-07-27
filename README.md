# Student-Performance-Prediction
# 🧠 Student Performance Prediction using Machine Learning

This project predicts students' **math scores** based on various features such as gender, parental education, lunch type, and test preparation course.

## 🔧 Tools Used
- Python 🐍
- pandas, numpy
- seaborn, matplotlib
- scikit-learn

## 📁 Dataset
- Source: [Students Performance Dataset (UCI)](https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv)

## 📊 Features
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- reading score
- writing score

## 🎯 Target
- math score (Regression)

## 📈 Results
- **Root Mean Squared Error (RMSE)**: ~9–10
- **R² Score**: ~0.75–0.80

## 📌 Steps
1. Data Cleaning and Encoding
2. Exploratory Data Analysis (EDA)
3. Linear Regression Modeling
4. Evaluation and Visualization

## 📷 Sample Output
- Scatter plot: Actual vs Predicted scores
- Heatmap: Feature Correlation
- Feature Importance list

## 🚀 How to Run
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
python student_performance_prediction.py
