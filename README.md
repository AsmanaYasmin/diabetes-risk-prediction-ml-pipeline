# Diabetes Risk Prediction: Diagnostic ML Pipeline

A robust, clinical diagnostic framework built on the Pima Indians Diabetes Dataset to identify high-risk individuals using benchmarked machine learning classifiers.

##   Methodology & Technical Highlights
- **Leak-Free Scikit-Learn Pipeline:** Missing data imputation (handling biologically impossible zero values for Glucose, BMI, Insulin, etc.) and feature standardization are executed strictly inside stratified folds to prevent data leakage.
- **Algorithmic Benchmarking:** Evaluated 6 core classifiers via 5-Fold Stratified Cross-Validation:
  - Logistic Regression
  - Support Vector Machines (SVM - RBF Kernel)
  - Random Forest Classifier
  - Gradient Boosting
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- **Optimization:** Grid search hyperparameter tuning applied to ensemble methods.
- **Clinical Explainability:** Tree-based feature importance extracted, highlighting **Glucose** and **BMI** as the primary risk determinants.

##   Performance Summary (5-Fold CV / Test)
- **Top Models:** SVM (RBF) & Logistic Regression demonstrated balanced precision-recall trade-offs.
- **Tuned Random Forest:** Yielded superior test recall (>72%) on the minority diabetic class, critical for reducing false negatives in clinical triage.

##   Installation & Running
```bash
git clone [https://github.com/](https://github.com/)<your-username>/diabetes-risk-prediction-ml-pipeline.git
cd diabetes-risk-prediction-ml-pipeline
pip install -r requirements.txt
