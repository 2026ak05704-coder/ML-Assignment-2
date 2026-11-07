# Online Shoppers Purchasing Intention — Classifier Comparison

An end-to-end machine learning project that trains **five classification models**
to predict whether an online browsing session results in a purchase, and serves
the comparison through an interactive **Streamlit** app.

---

## a. Problem Statement

E-commerce sites want to know, **during a session**, whether a visitor is likely
to complete a purchase so they can trigger interventions (offers, assistance,
retargeting). This is a **binary classification** problem: predict the `Revenue`
flag (`True` = purchase, `False` = no purchase) from session-level behavioural
and page-analytics features.

The business challenge is the **class imbalance** — only about 15% of sessions
end in a purchase — so headline accuracy is misleading and metrics such as
**F1** and **MCC** matter far more.

## b. Dataset Description

| Property      | Value                                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Name          | Online Shoppers Purchasing Intention Dataset                                                                                    |
| Source        | UCI Machine Learning Repository (ID 468) — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances     | 12,330 sessions                                                                                                                 |
| Features      | 17 (10 numeric + 7 categorical)                                                                                                 |
| Target        | `Revenue` (binary: purchase / no purchase)                                                                                      |
| Class balance | ≈ 15.5% positive (purchase), ≈ 84.5% negative                                                                                   |

**Feature groups**

- _Page-visit counts & durations:_ `Administrative`, `Administrative_Duration`,
  `Informational`, `Informational_Duration`, `ProductRelated`,
  `ProductRelated_Duration`
- _Google Analytics metrics:_ `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
- _Categorical context:_ `Month`, `OperatingSystems`, `Browser`, `Region`,
  `TrafficType`, `VisitorType`, `Weekend`

**Preprocessing (inside each saved pipeline)**

- Numeric features → `StandardScaler` (essential for KNN, SVM, Logistic Regression)
- Categorical features → `OneHotEncoder(handle_unknown="ignore")`
- Stratified 80/20 train/test split (`random_state=42`); the 20% test split is
  saved as `test_data.csv` and is the file uploaded to the app.

## c. GitHub Repository Link

> `https://github.com/<your-username>/online-shoppers-intention` _(replace with your repo URL)_

## d. Models Used

All five classifiers are trained on the **same** dataset and split, each wrapped
in a `Pipeline` so the app applies identical preprocessing without retraining.

### Comparison Table (evaluated on the held-out test set, ~2,466 sessions)

| ML Model Name          | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.8808     | 0.8828     | **0.7366** | 0.3586     | 0.4824     | 0.4592     |
| Decision Tree          | **0.8909** | **0.8835** | 0.6865     | 0.5445     | **0.6073** | **0.5499** |
| KNN                    | 0.8731     | 0.8267     | 0.7143     | 0.3010     | 0.4236     | 0.4086     |
| Naive Bayes            | 0.2729     | 0.7334     | 0.1726     | **0.9738** | 0.2933     | 0.1289     |
| Support Vector Machine | 0.8877     | 0.8824     | 0.7234     | 0.4450     | 0.5511     | 0.5099     |

_(Metrics are computed with scikit-learn; precision/recall/F1 use the positive
class. AUC uses predicted probabilities.)_

### Observations

| ML Model Name                         | Observation about model performance                                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**               | Highest precision (0.737) and AUC (0.883, just below the tree), but recall is 0.359. This thresholded model catches only a minority of purchases, so its moderate F1 (0.482) is weaker than its probability ranking suggests.                                                                                                             |
| **Decision Tree**                     | Best accuracy (0.891), AUC (0.884), F1 (0.607) and MCC (0.550). A depth-8 tree captures non-linear patterns while limiting complexity; its recall (0.545) is notably stronger than Logistic Regression and KNN on the positive class.                                                                                                     |
| **KNN**                               | Recall (0.301), F1 (0.424), and AUC (0.827) are the weakest among the non-Naive-Bayes models. Scaling helps distance comparisons, but one-hot categories create a high-dimensional feature space and the class imbalance makes majority-class neighbours common.                                                                          |
| **Naive Bayes**                       | Highest recall (0.974), but precision is only 0.173; the model labels many non-purchasing sessions as purchases. Its 0.273 accuracy and 0.129 MCC show that this recall comes at a severe false-positive cost. GaussianNB's distribution and independence assumptions are poorly matched to this mix of behavioural and one-hot features. |
| **Support Vector Machine**            | Second-best F1 (0.551) and MCC (0.510), with accuracy 0.888 and AUC 0.882. Scaling and the RBF kernel capture non-linear structure; recall (0.445) remains below the decision tree.                                                                                                                                                       |
| **Overall, Winner for this dataset?** | **Decision Tree.** It leads on accuracy, AUC, F1 and MCC, and has the best F1/MCC balance for the imbalanced target. Its recall (0.545) is still far from perfect, so a real deployment should select a decision threshold based on the cost of missed purchases versus false alarms.                                                     |

---

## Project Structure

```
project-folder/
├── app.py                      # Streamlit dashboard (loads models, no retraining)
├── requirements.txt            # Pinned dependencies
├── README.md
├── test_data.csv               # Held-out test split (features + Revenue) to upload
└── model/
    ├── train_models.py         # Source: trains the 5 pipelines, writes artifacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── metadata.json           # Target name, feature list, model file map
    └── metrics.json            # Metrics table used in this README
```

## How to Run Locally

```bash
pip install -r requirements.txt
# (re)train models and regenerate artifacts + test_data.csv
python model/train_models.py
# launch the app
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar, pick a model, and view the metrics,
confusion matrix, classification report, and the all-models comparison tab.
The `ucimlrepo` dependency is used only by the training script to fetch the
public dataset; the deployed app loads the saved pipelines and does not fetch or
retrain on data.

## Streamlit App Features

- **CSV upload** of test data (sidebar)
- **Model selection dropdown** (all five classifiers)
- **Evaluation metrics** display (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix** heatmap + **classification report**
- **Compare-all-models** tab with a highlighted table and a metric bar chart
- **Download sample test CSV** button

> The app loads the pre-trained `.pkl` pipelines and computes metrics on the
> uploaded data. It never retrains, so the numbers match this README.

## Live App

> `https://<your-app-name>.streamlit.app` _(replace with your deployed URL)_
# ML-Assignment-2
