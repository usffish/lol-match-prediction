# LoL Match Prediction

ML classification pipeline predicting League of Legends 2024 competitive match outcomes using player and team performance data from Oracle's Elixir.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

## Overview

Built on 12,276 rows and 123 features from the Oracle's Elixir 2024 competitive dataset, this project trains and evaluates three classifiers — KNN, SVM, and MLP — to predict match win/loss outcomes. The pipeline handles heavy class imbalance, high missingness (up to 97% in some columns), and mixed feature types across player-level and team-level statistics.

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| KNN (k=11, Manhattan, distance-weighted) | 91.5% | 0.916 | 0.974 |
| SVM (RBF kernel, tuned C & gamma) | 97.9% | 0.979 | — |
| MLP (Adam, tuned hidden layers) | 99.85% | — | — |

## Pipeline

1. Data collection — downloaded via kagglehub from Kaggle, auto-detects CSV delimiter
2. Cleaning — column normalization, null analysis, median imputation
3. Feature engineering — encodes categorical features (side, position, league), scales numerics with StandardScaler
4. Imbalance handling — imbalanced-learn for class balancing
5. Model training — GridSearchCV with StratifiedKFold for hyperparameter tuning on all three classifiers
6. Evaluation — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report

## Getting Started

The notebook runs in Google Colab. Open it and run all cells — it downloads the dataset automatically via kagglehub.

Dependencies are pinned and auto-installed in the first cell:

```
numpy==2.1.3, scikit-learn==1.4.2, imbalanced-learn==0.12.3, pandas==2.2.3, matplotlib==3.9.2
```

Or install locally:

```bash
git clone https://github.com/usffish/lol-match-prediction.git
cd lol-match-prediction
pip install -r requirements.txt
jupyter notebook MLProject.ipynb
```

## Dataset

League of Legends 2024 Competitive Game Dataset — Oracle's Elixir, via Kaggle.
12,276 rows, 123 features covering kills, deaths, assists, gold differentials, objective control, and early-game stats.

## Author

**Ismail Jhaveri** — [LinkedIn](https://www.linkedin.com/in/ismail-jhaveri-2021/) · [ismailj@usf.edu](mailto:ismailj@usf.edu)
