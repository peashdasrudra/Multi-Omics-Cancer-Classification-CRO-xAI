<div align="center">

# 🧬 Multi-Omics Breast Cancer Classification

### *DC-CRO Feature Selection × Explainable AI*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-Academic-purple?style=for-the-badge)](#license)

<br/>

> **Classifying breast cancer subtypes from integrated multi-omics data using Diversity-Controlled Chemical Reaction Optimization (DC-CRO) for feature selection and SHAP-based Explainable AI for model interpretability.**

<br/>

```
                  ╔══════════════════════════════════════════════════════════════════╗
                ║   705 Patients  ·  1,837 Features  ·  4 Omics Layers  ·  3 Targets   ║
                  ╚══════════════════════════════════════════════════════════════════╝
```

---

</div>

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [🔬 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [⚙️ Methodology](#️-methodology)
- [🏆 Baseline Results](#-baseline-results)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🗺️ Roadmap](#️-roadmap)
- [🔄 Reproducibility](#-reproducibility)
- [📜 License](#-license)

</details>

---

## 🔬 Overview

High-dimensional multi-omics data holds immense promise for **precision oncology**, yet the sheer number of features (1,800+ molecular markers) introduces noise, redundancy, and the curse of dimensionality.

This thesis tackles the challenge through a **three-stage pipeline**:

<table>
<tr>
<td width="33%" align="center">

### 🎯 Stage 1
**Baseline Benchmarking**

Evaluate RF, SVM & XGBoost on the full feature set with leakage-free cross-validation

</td>
<td width="33%" align="center">

### ⚡ Stage 2
**DC-CRO Feature Selection**

Discover compact, high-performance feature subsets via metaheuristic optimization

</td>
<td width="33%" align="center">

### 🔍 Stage 3
**Explainable AI**

SHAP-based interpretation for biologically meaningful clinical insights

</td>
</tr>
</table>

---

## 📊 Dataset

<table>
<tr><td>

| | Detail |
|:---|:---|
| 🏥 **Source** | TCGA Breast Cancer (BRCA) Cohort |
| 👥 **Samples** | 705 patients |
| 🔬 **Omics Layers** | RNA-Seq · CNV · Somatic Mutations · Protein (RPPA) |
| 📐 **Raw Features** | 1,936 molecular features |
| ✂️ **After Dedup** | **1,837** unique features (99 content-duplicate CNV columns removed) |

</td><td>

**Clinical Targets:**

| Target | Type | Classes |
|:---|:---|:---|
| `ER.Status` | Binary | Positive / Negative |
| `HER2.Final.Status` | Binary | Positive / Negative |
| `histological.type` | Multi-class | IDC / ILC / Other |

</td></tr>
</table>

> 📂 Raw data: `data/brca_data_w_subtypes.csv` — integrates all four omics layers with clinical annotations.

---

## ⚙️ Methodology

### Preprocessing Pipeline

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              PREPROCESSING PIPELINE                     │
                    └─────────────────────────────────────────────────────────┘

    ┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │ Raw CSV  │────▶│  Content    │────▶│   Target     │────▶│  80 / 20    │
    │ (1,936)  │     │  Dedup      │     │  Encoding    │     │  Stratified │
    └──────────┘     │  (→ 1,837)  │     └──────────────┘     │  Split      │
                     └─────────────┘                          └──────┬──────┘
                                                                     │
                                                    ┌────────────────┼────────────────┐
                                                    ▼                                 ▼
                                            ┌──────────────┐                 ┌──────────────┐
                                            │  Train Set   │                 │  Test Set    │
                                            └──────┬───────┘                 │  (held out)  │
                                                   │                         └──────────────┘
                                                   ▼
                                            ┌──────────────┐
                                            │ StandardScale│
                                            │ (fit on train│
                                            │  only)       │
                                            └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │    SMOTE     │
                                            │ (per CV fold │
                                            │  via ImbPipe)│
                                            └──────────────┘
```

### 🛡️ Scientific Rigor Guarantees

| Principle | Implementation |
|:---|:---|
| 🚫 **No Data Leakage** | Scaling fit exclusively on training data; SMOTE applied inside CV folds via `ImbPipeline` |
| 📊 **Stratified Splitting** | Class proportions preserved in both train and test sets |
| 🧹 **Content-Based Dedup** | Removes columns with identical *values* (not just names) — eliminates 99 redundant CNV features from shared chromosomal loci |
| 🏷️ **Feature Name Tracking** | Column names saved as `.npy` arrays for downstream SHAP explainability |
| 🔒 **Seed Locking** | Fixed seed (42) across NumPy, Python `random`, and `PYTHONHASHSEED` |

### 🤖 Baseline Classifiers

| Classifier | Configuration | Key Strength |
|:---|:---|:---|
| 🌲 **Random Forest** | 100 estimators, parallelized | Robust to noise |
| 🎯 **SVM** | RBF kernel, probability estimates | Strong on high-dim data |
| ⚡ **XGBoost** | Logloss metric, parallelized | Best overall performance |

---

## 🏆 Baseline Results

<div align="center">

> 📈 *Stratified 5-Fold Cross-Validation with per-fold SMOTE · Values: **mean ± std***

</div>

### 🔹 ER Status — *Binary Classification*

<table>
<tr>
<th>Model</th>
<th>Accuracy</th>
<th>F1 (Weighted)</th>
<th>F1 (Macro)</th>
<th>ROC-AUC</th>
<th>MCC</th>
</tr>
<tr>
<td>⚡ <b>XGBoost</b></td>
<td><b>93.16 ± 2.98</b></td>
<td><b>93.23 ± 2.85</b></td>
<td><b>90.97 ± 3.64</b></td>
<td>95.47 ± 2.82</td>
<td><b>0.822 ± 0.070</b></td>
</tr>
<tr>
<td>🌲 Random Forest</td>
<td>92.94 ± 2.43</td>
<td>92.83 ± 2.42</td>
<td>90.22 ± 3.20</td>
<td><b>95.69 ± 3.18</b></td>
<td>0.808 ± 0.065</td>
</tr>
<tr>
<td>🎯 SVM</td>
<td>91.57 ± 1.54</td>
<td>91.21 ± 1.78</td>
<td>87.76 ± 2.60</td>
<td>95.23 ± 3.51</td>
<td>0.769 ± 0.043</td>
</tr>
</table>

### 🔹 HER2 Final Status — *Binary Classification*

<table>
<tr>
<th>Model</th>
<th>Accuracy</th>
<th>F1 (Weighted)</th>
<th>F1 (Macro)</th>
<th>ROC-AUC</th>
<th>MCC</th>
</tr>
<tr>
<td>⚡ <b>XGBoost</b></td>
<td><b>92.86 ± 2.33</b></td>
<td><b>92.47 ± 2.73</b></td>
<td><b>85.32 ± 5.65</b></td>
<td><b>92.47 ± 5.67</b></td>
<td><b>0.717 ± 0.106</b></td>
</tr>
<tr>
<td>🌲 Random Forest</td>
<td>90.56 ± 2.93</td>
<td>88.95 ± 4.16</td>
<td>76.79 ± 9.50</td>
<td>89.04 ± 4.93</td>
<td>0.592 ± 0.152</td>
</tr>
<tr>
<td>🎯 SVM</td>
<td>86.87 ± 2.77</td>
<td>82.53 ± 4.65</td>
<td>60.57 ± 11.45</td>
<td>88.89 ± 5.33</td>
<td>0.330 ± 0.217</td>
</tr>
</table>

### 🔹 Histological Type — *Multi-class Classification*

<table>
<tr>
<th>Model</th>
<th>Accuracy</th>
<th>F1 (Weighted)</th>
<th>F1 (Macro)</th>
<th>ROC-AUC</th>
<th>MCC</th>
</tr>
<tr>
<td>⚡ <b>XGBoost</b></td>
<td><b>91.67 ± 1.63</b></td>
<td><b>91.64 ± 1.46</b></td>
<td><b>86.16 ± 2.14</b></td>
<td><b>93.77 ± 2.49</b></td>
<td><b>0.726 ± 0.043</b></td>
</tr>
<tr>
<td>🎯 SVM</td>
<td>89.36 ± 3.30</td>
<td>88.52 ± 3.58</td>
<td>79.78 ± 6.35</td>
<td>90.56 ± 5.09</td>
<td>0.615 ± 0.129</td>
</tr>
<tr>
<td>🌲 Random Forest</td>
<td>88.30 ± 2.18</td>
<td>87.44 ± 2.29</td>
<td>77.99 ± 3.99</td>
<td>91.40 ± 4.24</td>
<td>0.578 ± 0.083</td>
</tr>
</table>

<br/>

<div align="center">

### 💡 Key Findings

</div>

> **XGBoost dominates** across all three clinical targets, achieving the highest accuracy, F1, and MCC scores consistently. Notably:
> - **ER Status** is the easiest target (93.2% accuracy) — strong biological signal from hormone receptor expression.
> - **HER2 Status** shows the most room for improvement (MCC: 0.717) — class imbalance makes this a prime candidate for **DC-CRO feature selection**.
> - **Histological Type** benefits from multi-class handling (91.7% accuracy) — XGBoost's gradient-based optimization handles the 3-class problem well.

---

## 📁 Project Structure

```
🧬 Multi Omics Cancer - CRO xAI/
│
├── 📂 data/
│   └── brca_data_w_subtypes.csv          # Raw TCGA BRCA multi-omics dataset
│
├── 📂 src/
│   ├── __init__.py                       # Package initializer
│   ├── preprocess.py                     # Data loading, deduplication, encoding, SMOTE
│   ├── baseline.py                       # Stratified CV evaluation with ImbPipeline
│   └── utils.py                          # Reproducibility (seed locking)
│
├── 📂 outputs/
│   ├── baseline_metrics/                 # Per-target CSV results (pivoted)
│   │   ├── ER.Status_baseline.csv
│   │   ├── HER2.Final.Status_baseline.csv
│   │   └── histological.type_baseline.csv
│   ├── preprocessed/                     # .npy arrays (train/test, feature names)
│   └── figures/                          # Plots (populated in later weeks)
│
├── 📂 notebooks/                         # Jupyter notebooks for exploration
├── 📂 reports/                           # Generated thesis figures & reports
│
├── 🚀 run_week1.py                      # Master script — full Week 1 pipeline
├── 📋 requirements.txt                  # Python dependencies
├── 📋 .gitignore
└── 📋 README.md
```

---

## 🚀 Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-green?logo=pypi&logoColor=white)

### Installation

```bash
# Clone the repository
git clone https://github.com/peashdasrudra/Multi-Omics-Cancer-Classification---CRO-xAI.git
cd Multi-Omics-Cancer-Classification---CRO-xAI

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python run_week1.py
```

<details>
<summary>📝 <b>What does it do?</b> (click to expand)</summary>

<br/>

1. 📥 Loads and deduplicates the raw multi-omics dataset (1,936 → 1,837 features)
2. For each of the 3 clinical targets:
   - 🏷️ Encodes labels and splits data (80/20 stratified)
   - 📏 Scales features (StandardScaler, fit on train only)
   - 💾 Saves pre-SMOTE data for leakage-free CV evaluation
   - ⚖️ Applies SMOTE and saves resampled data for future DC-CRO
   - 🤖 Runs baseline evaluation (RF, SVM, XGBoost) with Stratified 5-Fold CV
   - 📊 Exports metrics to CSV

</details>

### Dependencies

| Package | Purpose |
|:---|:---|
| `numpy` · `pandas` | Data manipulation & analysis |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `imbalanced-learn` | SMOTE oversampling & ImbPipeline |
| `xgboost` | Gradient boosted tree classifier |
| `matplotlib` · `seaborn` | Visualization & plotting |
| `joblib` | Model serialization & persistence |

---

## 🗺️ Roadmap

<table>
<tr>
<th>Week</th>
<th>Milestone</th>
<th>Status</th>
</tr>
<tr>
<td align="center"><b>1</b></td>
<td>📊 Data preprocessing, content-based deduplication, baseline evaluation (RF, SVM, XGBoost)</td>
<td align="center">✅ <b>Complete</b></td>
</tr>
<tr>
<td align="center"><b>2</b></td>
<td>⚡ DC-CRO metaheuristic skeleton & feature selection engine</td>
<td align="center">🔄 In Progress</td>
</tr>
<tr>
<td align="center"><b>3</b></td>
<td>🔗 DC-CRO integration with classifiers & hyperparameter tuning</td>
<td align="center">📅 Planned</td>
</tr>
<tr>
<td align="center"><b>4</b></td>
<td>🔍 SHAP-based explainability analysis & feature importance</td>
<td align="center">📅 Planned</td>
</tr>
<tr>
<td align="center"><b>5</b></td>
<td>📝 Final evaluation, comparison tables, thesis write-up & figures</td>
<td align="center">📅 Planned</td>
</tr>
</table>

---

## 🔄 Reproducibility

All experiments use a **fixed random seed (`42`)** locked across:

- 🔢 **NumPy** — `np.random.seed(42)`
- 🐍 **Python** — `random.seed(42)`
- #️⃣ **Hash Seed** — `PYTHONHASHSEED=42`

Results are **100% reproducible** given the same environment, dataset, and library versions.

---

## 📜 License

This project is part of an academic thesis. All rights reserved.

---

<div align="center">

<br/>

**Department of Computer Science & Engineering**

**Northern University of Business & Technology Khulna**

<br/>

*Made with 🧬 for precision oncology research*

</div>
