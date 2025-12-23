# Explainable Credit Risk Scoring System

## Problem Statement
The goal of this project is to predict the probability of credit card default
for a customer based on demographic and financial attributes.

### Dataset used-
- UCI Credit Card Default Dataset
- 30,001 samples, 25 features
- Binary classification (default vs non-default)
- Moderately imbalanced (~22% default rate)

### Approach
- Feature engg based on payment behaviour and credit usage
- Baseline model: Logistic Regression(interpretable, low variance)
- Non linear model: Gradient Boosting to capture non linear interactions
- Probability calibration using isotonic regression
- Explainability using SHAP(global + local explanations)

### Tech Stack
- Python3 (Core implementation)
- Pandas, Numpy (Data manpulation)
- Matplotlib, Seaborn (Visualization)
- Scikit learn (Modelling and evaluation)
- SHAP (Explainability)
- FastAPI (Model deployment as API)

This is formulated as a binary classification problem where the model outputs
a calibrated probability of default, not just a class label.

Models used - 1. Logistic Regression, Gradient boosting0.7827557529981515

| Model               | ROC-AUC               | Calibration | Interpretability |
| ------------------- | -------               | ----------- | ---------------- |
| Logistic Regression | 0.709246838167149     | Good        | High             |
| Gradient Boosting   | 0.7824828162748017    | Poor        | Medium           |
| Calibrated GB       | 0.7827557529981515    | Good        | Medium           |

* Gradient Boosting improves ranking performance, while calibration significantly improves
* the reliability of predicted probabilities without affecting ROC-AUC.

### Explainability
- SHAP is used to explain feature contributions
- Key drivers - recent payment delays, credit utilization
- Local explanations show why individual customers are classified as low/high risk

### Deployment
- Model served via FastAPI
- /predit endpoint returns calibrated default probability and risk label
- Interactive API documentation via Swagger(/docs)

### How to Run
- pip3 install -r requirements.txt (or pip install -r requirements.txt)
- uvicorn app.main:app -- reload 

The model is exposed via a FastAPI service that returns calibrated default probabilities. 
Input validation ensures consistency with training features, 
and outputs are suitable for downstream decision systems.

## Conclusion
This project implements a production-oriented credit risk modeling pipeline, progressing from interpretable linear baselines to non-linear ensemble methods. Gradient Boosting significantly improves ranking performance, while isotonic calibration corrects probability misalignment without affecting ROC-AUC. SHAP-based global and local explanations provide faithful feature attributions, ensuring model transparency. The calibrated model is deployed via FastAPI, enabling consistent, validated probability outputs suitable for integration into downstream financial decision systems.

> [!NOTE]
> Models are not committed. Train using notebook 02 or download from [here](https://drive.google.com/drive/folders/1V6ck4YbjVsdDYagZ5wbKSnqyUh1lyOn7?usp=sharing).
