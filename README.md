# Five-Model Classification Benchmark on the UCI Internet Usage Data

An individual data mining project (SGH Warsaw School of Economics, MSc Advanced Analytics & Big Data, 2025/26) that compares five scikit-learn classifiers (Logistic Regression, Decision Tree, k-Nearest Neighbours, Support Vector Machine and a Multi-Layer Perceptron) on the same binary task, using the UCI Internet Usage Data (GVU WWW User Survey, 1997).

## Project context

The aim was to practise a complete classification workflow (data understanding, preprocessing, training several model families and comparing them on a common set of metrics), with charts for each step.

## What this project does

* Loads `final_general.dat.gz` (space-separated, numerically coded survey answers) and keeps the 10,066 rows with exactly 72 fields
* Fills missing values with the column median (numeric) or mode (categorical)
* Uses the **last column of the file as the target** and, because it has 7,391 distinct values, converts it to binary by splitting at its median (6,371 records in class 0, 3,695 in class 1)
* Uses the other 71 columns as features, with an 80/20 train/test split (8,052 / 2,014) and standard scaling
* Trains Logistic Regression, a Decision Tree (max depth 10), KNN, an RBF-kernel SVM and an MLP (100→50 hidden units)
* Compares models on accuracy, precision, recall, F1 and ROC-AUC, with confusion matrices, ROC curves and comparison charts

## Results (test set)

| Model | Train accuracy | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| SVM (RBF) | 80.1% | 66.9% | 0.557 | 0.751 |
| Decision Tree | 73.6% | 66.9% | 0.557 | 0.737 |
| Logistic Regression | 66.9% | 66.5% | 0.548 | **0.757** |
| MLP | 99.7% | 65.1% | 0.499 | 0.739 |
| KNN | 77.1% | 64.9% | 0.524 | 0.726 |

**How to read these numbers**

* **The models only slightly beat the simplest baseline.** 63.3% of test records are in class 0, so always predicting class 0 already gives 63.3% accuracy. The best model reaches 66.9%.
* **Logistic Regression had the best AUC (0.757)** and almost no gap between train and test accuracy. The more flexible models overfit: the MLP reached 99.7% training accuracy but only 65.1% on the test set.
* On this data, more complex models brought no benefit over a linear baseline.

## Tech stack

* Python, Jupyter Notebook
* pandas, NumPy
* scikit-learn: `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`, `MLPClassifier`, metrics
* Matplotlib, seaborn

## Project structure

```
Internet-usage-multi-model-benchmarking/
├── internet-usage-multi-model-benchmarking.ipynb   # Full pipeline in one cell: data → preprocessing → 5 models → comparison
└── README.md
```

To run it, download `final_general.dat.gz` from the [UCI Internet Usage Data page](https://kdd.ics.uci.edu/databases/internet_usage/internet_usage.data.html) and update `file_path` in the notebook. Running the notebook saves nine PNG charts (one per step plus the final comparison).

## Honest limitations

* **The target is not a defined survey question.** It is simply the last column of the file, split at its median. The notebook does not identify what that column measures, so the results show how the five methods compare on this task, not anything about internet users.
* **Survey codes treated as numbers.** Answers are numeric codes (including special codes such as 98 and 99), and they were scaled as continuous values rather than one-hot encoded.
* **Feature names not attached**, so the models cannot be interpreted in terms of specific survey questions.
* **Default hyperparameters and a single split**; no cross-validation or tuning beyond a quick check of k for KNN.
* **Accuracy close to the majority-class baseline**, as explained above.
