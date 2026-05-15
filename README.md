# Predictive Analysis of Internet User Demographics
### A Comparative Machine Learning Benchmark Study

## 📌 Project Overview
This research project conducts a rigorous comparative analysis of five distinct machine learning architectures to predict user segments based on demographic characteristics. Utilizing a dataset of over 10,000 records, the study examines how features such as **income, education, and age** correlate with technology diffusion and digital adoption patterns.

The primary objective was to benchmark classical statistical models against more complex architectures (Neural Networks) to determine the most effective approach for structured demographic data.

---

## 🛠️ Technical Stack & Methodology
### The Machine Learning Pipeline:
* **Data Preprocessing:** Handled missing values, implemented **Label Encoding** for categorical features, and applied **Standard Scaling** to ensure feature parity across models.
* **Evaluation Metrics:** Models were evaluated not just on raw accuracy, but on **ROC/AUC, Precision-Recall trade-offs, and Confusion Matrices**.
* **Model Benchmarking:** * **SVM (Support Vector Machine):** Maximizing the margin between demographic segments.
    * **Decision Trees:** Providing a hierarchical look at decision logic.
    * **Logistic Regression:** Baseline statistical classification for interpretability.
    * **Neural Networks (MLP):** Exploring non-linear patterns in user behavior.
    * **K-Nearest Neighbors (KNN):** Classification based on demographic proximity.

---

## 🚀 Key Performance Results
The project revealed that for structured demographic data, classical models often rival or outperform deep learning approaches in both speed and accuracy.

| Model | Accuracy (%) | AUC Score | Verdict |
| :--- | :---: | :---: | :--- |
| **SVM** | **66.93%** | 0.742 | **Top Overall Performer** |
| **Logistic Regression** | 65.12% | **0.757** | **Best Interpretability/AUC** |
| **Neural Network (MLP)** | 64.85% | 0.731 | High complexity, moderate results |
| **Decision Tree** | 62.40% | 0.690 | High variance, prone to overfitting |
| **KNN** | 61.15% | 0.685 | Computationally expensive |

### Visualizations Included:
* **ROC Curves:** Comparison of True Positive vs. False Positive rates across all 5 models.
* **Feature Correlation Heatmaps:** Identifying the strongest demographic drivers of technology adoption.

---

## 📈 Business & Social Insights
1.  **Classical vs. Complex:** Demonstrated that classical models like SVM and Logistic Regression can outperform Neural Networks on structured demographic datasets, offering better resource efficiency.
2.  **The Digital Divide:** Identified income and education as the primary predictors in user segmentation, providing a framework for understanding technology diffusion.
3.  **Predictive Power:** Confirmed that while demographic data provides a strong foundation (0.75+ AUC), behavioral data is likely required to push accuracy beyond the 70% threshold.

---

## 📂 Repository Structure
* `Demographic_Analysis.ipynb`: Full Python implementation from EDA to Model Evaluation.
* `plots/`: Exported ROC curves and model comparison charts.
* `data/`: (If public) The demographic dataset used for the study.
