Applied AI & ML Essentials – Capstone Project (Part 3)
Overview

This project improves the heart attack risk prediction models developed in Part 2 by using advanced machine learning techniques.

Objectives
Build and compare ensemble models.
Improve model performance through hyperparameter tuning.
Create a complete machine learning pipeline.
Save and reload the best model for deployment.
Dataset
Dataset: cleaned_data.csv
Target Variable: heart_attack_risk
0 = Low Risk
1 = High Risk
Data Preprocessing
Removed unnecessary columns.
Encoded categorical features.
Applied one-hot encoding.
Split data into training and testing sets (80:20).
Standardized numerical features.
Models Used
Decision Tree
Controlled Decision Tree
Random Forest
Gradient Boosting
Logistic Regression (comparison)
Techniques Applied
Gini vs Entropy comparison
Feature Importance
Feature Ablation
5-Fold Cross Validation
Hyperparameter Tuning (GridSearchCV)
Machine Learning Pipeline
Learning Curve
Model Serialization (Joblib)
Model Evaluation

Models were evaluated using:

Accuracy
ROC-AUC
Cross-validation score
Model stability

The best-performing model was selected for deployment.

Pipeline

The final pipeline includes:

Missing value imputation
Feature scaling
Random Forest classifier

This ensures consistent preprocessing and prevents data leakage.

Model Saving

The best model was saved using:

joblib.dump(best_pipeline, "best_model.pkl")

It was successfully reloaded for future predictions.

Files Generated
Plots
Feature Importance
Learning Curve
Results
Decision Tree comparison
Gini vs Entropy comparison
Feature importance
Feature ablation
Cross-validation results
Grid search results
Learning curve
Final model comparison
Requirements
pandas
numpy
matplotlib
scikit-learn
joblib
Limitations
Intended for educational purposes only.
Not suitable for clinical diagnosis.
Performance depends on dataset quality.
Feature importance does not imply causation.
Conclusion

This project compared multiple machine learning models for heart attack risk prediction. After evaluating performance using cross-validation and ROC-AUC, the best model was selected, saved, and prepared for deployment using a complete machine learning pipeline.

This version is about 60–70% shorter while keeping all the important information.
