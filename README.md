
# Classification Comparison Lab

### Machine Learning — Assignment 2
**BITS Pilani | M.Tech in Artificial Intelligence and Machine Learning**

## 1. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms on a publicly available binary classification dataset.

All models are trained and evaluated using a consistent stratified train-test split. Their performance is compared using six evaluation metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive web application is developed using Streamlit to allow users to upload labeled test datasets, select trained models, view evaluation results, visualize confusion matrices, and compare all five classifiers.

The application loads previously trained models and performs predictions without retraining.

---

## 2. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Access Method:** `sklearn.datasets.load_breast_cancer`

### Dataset Characteristics

| Property | Description |
|---|---|
| Total Samples | 569 |
| Input Features | 30 |
| Target Variable | Diagnosis |
| Classification Type | Binary Classification |
| Benign Samples | 357 |
| Malignant Samples | 212 |
| Training Split | 80% |
| Testing Split | 20% |
| Random State | 42 |
| Stratification | Enabled |

### Target Variable

The project uses the following target encoding:

- **0 — Benign**
- **1 — Malignant**

The original scikit-learn dataset uses the opposite encoding. The labels are converted so that malignant cases represent the positive class for evaluation.

### Why This Dataset Was Selected

The Wisconsin Diagnostic Breast Cancer dataset was selected because:

1. It is a publicly available and widely used benchmark for binary classification.
2. It contains numerical features suitable for comparing different machine learning algorithms.
3. It provides a meaningful real-world classification problem.
4. It allows comparison of linear, nonlinear, distance-based, probabilistic, and tree-based classifiers.
5. The unequal class distribution makes evaluation metrics such as recall, F1 score, and MCC particularly relevant.

The dataset contains 357 benign samples and 212 malignant samples.

A stratified split is used to preserve approximately the same class proportions in the training and testing datasets.

---

## 3. Data Preprocessing

The dataset is divided into training and testing subsets using:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Based on target labels

### Feature Scaling

StandardScaler is applied to algorithms that are sensitive to feature magnitude.

| Algorithm | Standardization |
|---|---|
| Logistic Regression | Yes |
| Decision Tree | No |
| K-Nearest Neighbors | Yes |
| Gaussian Naive Bayes | No |
| Support Vector Machine | Yes |

For Logistic Regression, KNN, and SVM, standardization is incorporated into scikit-learn pipelines.

The pipelines are fitted only on the training dataset, preventing information leakage from the held-out test data.

Decision Tree and Gaussian Naive Bayes use the original unscaled numerical features.

---

## 4. Machine Learning Models Implemented

Five supervised classification algorithms are implemented.

### 4.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a sample belonging to a particular class.

Standardized features are used to support stable optimization.

### 4.2 Decision Tree

Decision Tree classifies samples using a sequence of feature-based decision rules.

The tree uses a maximum depth of 5 to restrict model complexity and help reduce overfitting.

### 4.3 K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm.

It predicts the class of a sample based on the classes of its nearest neighbors.

Feature standardization is performed because distance calculations are affected by differences in feature scales.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem.

It assumes that features are conditionally independent given the class and that individual feature values follow Gaussian distributions within each class.

### 4.5 Support Vector Machine (SVM)

Support Vector Machine identifies a decision boundary that separates classes.

An RBF kernel is used to support nonlinear classification.

Probability estimates are enabled to calculate ROC-AUC.

---

## 5. Model Evaluation Metrics

The following six metrics are used to evaluate classification performance.

### Accuracy

Measures the proportion of correctly classified samples.

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

### AUC

Area Under the Receiver Operating Characteristic Curve measures the ability of a classifier to rank positive samples above negative samples across classification thresholds.

A higher AUC indicates better discrimination.

### Precision

Measures the proportion of predicted malignant cases that are actually malignant.

`Precision = TP / (TP + FP)`

### Recall

Measures the proportion of actual malignant cases correctly identified.

`Recall = TP / (TP + FN)`

Recall is especially important in this dataset because false negatives correspond to missed malignant cases.

### F1 Score

The harmonic mean of precision and recall.

`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

### Matthews Correlation Coefficient (MCC)

MCC measures the correlation between predicted and actual classifications using all four confusion-matrix outcomes.

`MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))`

MCC is useful when class distributions are unequal.

All precision, recall, and F1 values are calculated with the malignant class as the positive class.

---

## 6. Model Performance Comparison

The following results were obtained from the held-out labeled test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| **Support Vector Machine** | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Performance Highlights

- **Highest Accuracy:** Support Vector Machine — 97.37%
- **Highest AUC:** Logistic Regression — 0.9960
- **Highest Precision:** SVM and Gaussian Naive Bayes — 1.0000
- **Highest Recall:** SVM and Logistic Regression — 0.9286
- **Highest F1 Score:** SVM — 0.9630
- **Highest MCC:** SVM — 0.9442

---

## 7. Model-wise Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 96.49% and the highest AUC of 0.9960.

Its F1 score of 0.9512 indicates a strong balance between precision and recall.

The results suggest that the standardized dataset supports effective linear classification.

### Decision Tree

Decision Tree achieved an accuracy of 92.11% and an F1 score of 0.8861.

It recorded the lowest recall of 0.8333, indicating that it missed more malignant cases than the leading models.

The maximum depth constraint of 5 helps limit model complexity, although the resulting test performance was lower than that of the other classifiers.

### K-Nearest Neighbors

KNN achieved an accuracy of 95.61% and an F1 score of 0.9383.

Feature standardization is particularly important because KNN relies on distance calculations.

KNN performed competitively but obtained slightly lower scores than Logistic Regression and SVM.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 93.86% and an AUC of 0.9934.

It obtained perfect precision of 1.0000, meaning that none of its malignant predictions were false positives on this test set.

However, recall was 0.8333, indicating that some actual malignant cases were classified as benign.

Its conditional independence assumption may limit its ability to model relationships between correlated features.

### Support Vector Machine

SVM achieved the highest accuracy of 97.37%, F1 score of 0.9630, and MCC of 0.9442.

It also achieved perfect precision of 1.0000 and recall of 0.9286.

The RBF kernel allows nonlinear decision boundaries and produced the strongest overall results under the selected F1 and MCC criteria.

---

## 8. Overall Model Comparison

Support Vector Machine emerged as the best overall classifier based on F1 score and Matthews Correlation Coefficient.

SVM achieved:

- Accuracy: 0.9737
- AUC: 0.9947
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC: 0.9442

Logistic Regression achieved a slightly higher AUC of 0.9960 compared with SVM's 0.9947.

Both Logistic Regression and SVM achieved the highest malignant-class recall of 0.9286.

Since missing malignant cases is an important concern in this classification problem, recall is considered alongside F1 and MCC rather than relying only on accuracy.

**Selected Best Overall Model: Support Vector Machine (SVM)**

This selection is based on its highest F1 score and MCC, together with its strong malignant-class recall.

---

## 9. Interactive Streamlit Application

An interactive application is developed using Streamlit to evaluate labeled test datasets with already-trained classification models.

### Application Features

**1. Upload Test Dataset**

Users can upload a CSV file containing the required input features and true target labels.

**2. Select Classification Model**

Users can choose among:

- Logistic Regression
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Support Vector Machine

**3. Evaluate Model Performance**

The application calculates six classification metrics for the uploaded labeled dataset.

**4. Confusion Matrix Visualization**

The application displays a confusion matrix to help analyze correct and incorrect classifications.

**5. All-Model Comparison**

Users can compare all five trained classifiers on the same uploaded dataset.

**6. Pretrained Model Loading**

Saved model artifacts are loaded using Joblib.

The application does not retrain models during evaluation.

---

## 10. Project Structure

```text
ML_Assignment2/
|
|-- train_models.py
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- test_data.csv
|-- comparison_metrics.csv
|
|-- model/
|   |-- *.joblib
|   |-- metadata.json
|
|-- .gitignore
```

### File Descriptions

| File | Purpose |
|---|---|
| `train_models.py` | Loads dataset, trains classifiers, evaluates performance, and saves artifacts |
| `app.py` | Streamlit application for interactive model evaluation |
| `requirements.txt` | Python dependencies |
| `test_data.csv` | Labeled held-out test dataset |
| `comparison_metrics.csv` | Model comparison results |
| `model/*.joblib` | Saved trained models and preprocessing pipelines |
| `model/metadata.json` | Feature names, feature order, and target mapping |
| `README.md` | Project documentation |

---

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/MrunmayeeManmadkar/ML_Assignment2.git
cd ML_Assignment2
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train and Evaluate Models

```bash
python train_models.py
```

This script:

1. Loads the WDBC dataset.
2. Prepares the target labels.
3. Performs a stratified train-test split.
4. Trains five classification algorithms.
5. Calculates six evaluation metrics.
6. Saves trained models.
7. Exports the held-out test dataset.
8. Generates the model comparison CSV file.

### Step 4: Launch the Streamlit Application

```bash
streamlit run app.py
```

### Step 5: Evaluate Test Data

1. Open the Streamlit application.
2. Upload `test_data.csv`.
3. Select a classification model.
4. View the calculated evaluation metrics.
5. Examine the confusion matrix.
6. Open the all-model comparison section to compare all classifiers.

---

## 12. GitHub Repository

**Repository:** https://github.com/MrunmayeeManmadkar/ML_Assignment2

The repository contains the implementation, application code, dependencies, and project documentation.

---

## 13. Streamlit Deployment

The application can be deployed using Streamlit Community Cloud.

**Deployment Platform:** https://share.streamlit.io/

### Deployment Configuration

- Repository: `MrunmayeeManmadkar/ML_Assignment2`
- Branch: `main`
- Main application file: `app.py`

The application requires the trained model artifacts and metadata to be available in the deployed repository.

---

## 14. Conclusion

This project successfully compares five supervised machine learning classification algorithms using a consistent evaluation procedure on the Wisconsin Diagnostic Breast Cancer dataset.

The experimental results demonstrate that:

1. SVM achieved the highest accuracy, F1 score, and MCC.
2. Logistic Regression achieved the highest AUC.
3. KNN demonstrated competitive classification performance.
4. Gaussian Naive Bayes achieved perfect precision but comparatively lower recall.
5. Decision Tree recorded the lowest overall performance among the evaluated models.

SVM was selected as the best overall classifier based on F1 score and MCC.

The Streamlit application provides an interactive interface for evaluating uploaded labeled datasets using saved trained models without retraining.

The project demonstrates supervised classification, preprocessing pipelines, model evaluation, performance comparison, model persistence, and interactive application deployment.

**Note:** The dataset is used for educational machine learning experiments. The trained models are not intended for clinical diagnosis.
