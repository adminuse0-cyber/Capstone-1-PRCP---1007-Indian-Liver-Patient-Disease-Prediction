# PRCP-1007: Indian Liver Patient Disease Prediction — Complete Technical & Clinical Project Documentation

**Project ID:** `PRCP-1007-LiverPatientPred`  
**Domain:** Healthcare & Clinical Decision Support Systems  
**Dataset:** Indian Liver Patient Dataset (ILPD — North East Andhra Pradesh, India)  
**Primary Notebook:** `PRCP-1007-LiverPatientPred-v2_3.ipynb`  

---

## 1. Executive Summary & Business Case

### 1.1 Healthcare Background & Problem Context
Liver diseases constitute a major public health burden globally and across India. Chronic hepatic conditions—such as alcoholic liver disease, non-alcoholic fatty liver disease (NAFLD), viral hepatitis, cholestasis, cirrhosis, and drug-induced liver injury—are driven by:
* **Excessive alcohol consumption** and metabolic disorders.
* **Environmental and occupational exposure** to hepatotoxic chemicals and harmful gases.
* **Dietary and pharmacological factors**, including consumption of contaminated food, preserved/pickled foods containing hepatotoxins, and prolonged intake of hepatotoxic medications.

### 1.2 Clinical Bottleneck
Early-stage liver dysfunction frequently progresses **asymptomatically**. By the time overt physical symptoms (jaundice, ascites, or hepatic encephalopathy) manifest, irreversible liver fibrosis or cirrhosis has often developed. Simultaneously, clinicians in high-volume outpatient clinics face diagnostic fatigue when manually cross-referencing multi-parameter **Liver Function Tests (LFTs)** alongside patient age and biological sex.

### 1.3 Project Objectives & Deliverables
This project builds an end-to-end supervised machine learning screening system to assist physicians in early liver disease detection. Specifically, it fulfills five core project deliverables:
1. **Task 1 — Complete Exploratory Data Analysis (EDA) Report:** Statistical profiling, missingness/duplicate audits, univariate/bivariate distributions, correlation diagnostics, and non-parametric hypothesis testing.
2. **Task 2 — Predictive Modeling Across Multiple Classifiers:** Implementation, cross-validation, and hyperparameter optimization of **8 distinct classification algorithms** (10 evaluated configurations total).
3. **Task 3 — Model Design Basis & Engineering Rationale:** Mathematical and clinical justification for deduplication, stratified splitting, median imputation, domain feature engineering, outlier-resistant scaling, and class-imbalance handling.
4. **Model Comparison Report:** Multi-metric benchmarking (ROC-AUC, F1-Score, Recall/Sensitivity, Specificity, Precision, Accuracy, and 5-fold Cross-Validation stability) culminating in a production deployment recommendation.
5. **Challenges & Remedies Report:** Documentation of real-world data flaws in the ILPD dataset and the exact engineering techniques used to resolve them.

---

## 2. Project Workspace & Architecture

The project directory (`PROJECT 2/`) is organized cleanly into documentation, raw data, and executable analysis code:

```text
PROJECT 2/
├── Data/
│   └── Indian Liver Patient Dataset (ILPD).csv   # Raw 583-record dataset (headerless CSV)
├── PRCP-1007-LiverPatientPred.docx              # Official problem statement specification
├── PRCP-1007-LiverPatientPred-v2_3.ipynb        # End-to-end executable Jupyter Notebook
└── PRCP-1007-Project-Documentation.md           # Comprehensive project documentation
```

---

## 3. Dataset Specification & Clinical Domain Dictionary

The **Indian Liver Patient Dataset (ILPD)** contains **583 patient records** collected from North East Andhra Pradesh, India:
* **Target Distribution:** `416` Liver Disease patients (`71.36%`, originally coded `1`) and `167` Non-Liver Disease patients (`28.64%`, originally coded `2`).
* **Gender Distribution:** `441` Male patients (`75.64%`) and `142` Female patients (`24.36%`).
* **Age Capping Rule:** Any patient whose age exceeded `89` is recorded as `90`.

| # | Attribute Name | Data Type | Physiological & Clinical Meaning | Normal Clinical Reference |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `Age` | Integer | Patient age in years (`4` to `90`). | `4 – 90` years |
| **2** | `Gender` | Categorical | Biological sex (`Male` or `Female`). | Binary (`Male` / `Female`) |
| **3** | `Total_Bilirubin` | Float | Sum of unconjugated and conjugated bilirubin in serum. Elevated levels cause jaundice. | `0.1 – 1.2 mg/dL` |
| **4** | `Direct_Bilirubin` | Float | Water-soluble conjugated bilirubin processed by hepatocytes; spikes in biliary obstruction or liver cell injury. | `0.0 – 0.3 mg/dL` |
| **5** | `Alkaline_Phosphotase` (ALP) | Integer | Enzyme concentrated in bile duct epithelium and bone; elevated in cholestasis and infiltrative liver disease. | `44 – 147 IU/L` |
| **6** | `Alamine_Aminotransferase` (ALT / SGPT) | Integer | Liver-specific cytosolic enzyme released into bloodstream during acute hepatocellular inflammation/necrosis. | `7 – 56 IU/L` |
| **7** | `Aspartate_Aminotransferase` (AST / SGOT) | Integer | Mitochondrial/cytosolic enzyme in liver, heart, and muscle; elevated in hepatocellular injury. | `10 – 40 IU/L` |
| **8** | `Total_Proteins` | Float | Combined serum albumin and globulins synthesized predominantly by the liver. | `6.0 – 8.3 g/dL` |
| **9** | `Albumin` | Float | Primary plasma protein synthesized exclusively by hepatocytes; decreases in chronic liver failure. | `3.5 – 5.0 g/dL` |
| **10** | `Albumin_and_Globulin_Ratio` (A/G) | Float | Ratio of Albumin to Globulin. Drops below `1.0` in cirrhosis as albumin falls and immunoglobulins rise. | `1.1 – 2.5` |
| **11** | `Target` | Integer | Ground truth diagnosis: `1` = Liver Disease (mapped to `1`), `2` = No Liver Disease (mapped to `0`). | Binary (`1` / `0`) |

---

## 4. Exploratory Data Analysis (EDA) Key Findings

### 4.1 Data Integrity Audit
* **Headerless File Structure:** Line 1 of the raw CSV (`65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9,1`) is a patient record rather than column headers. Explicit schema mapping (`header=None`) prevents losing a valid record.
* **Exact Duplicates:** **13 exact duplicate rows** (`2.23%` of records) exist in the raw file, reducing the dataset to **570 unique patients** after deduplication.
* **Missing Values:** Only **4 records (`0.69%`)** contain missing (`NaN`) values, all occurring exclusively in `Albumin_and_Globulin_Ratio`.

### 4.2 Distribution Skewness & Clinical Outliers
* **Extreme Positive Skewness:** Liver enzymes and bilirubin markers exhibit heavy right tails caused by acute hepatic injury cases:
  * `Aspartate_Aminotransferase` (AST): Skewness = `10.55`, Kurtosis = `150.92` (Median = `42 IU/L`, Max = `4,929 IU/L`).
  * `Alamine_Aminotransferase` (ALT): Skewness = `6.55` (Median = `35 IU/L`, Max = `2,000 IU/L`).
  * `Total_Bilirubin`: Skewness = `4.91` (Median = `1.0 mg/dL`, Max = `75.0 mg/dL`).
  * `Alkaline_Phosphotase` (ALP): Skewness = `3.77` (Median = `208 IU/L`, Max = `2,110 IU/L`).
* **Symmetrical Markers:** `Age` (`skew = -0.03`), `Total_Proteins` (`skew = -0.29`), and `Albumin` (`skew = -0.04`) follow roughly symmetric distributions.

### 4.3 Bivariate Significance & Multicollinearity
* **Mann-Whitney U Hypothesis Tests:** `Direct_Bilirubin` ($p = 2.1 \times 10^{-12}$), `Total_Bilirubin` ($p = 4.5 \times 10^{-12}$), `AST` ($p = 1.8 \times 10^{-11}$), `ALT` ($p = 8.9 \times 10^{-10}$), `ALP` ($p = 3.2 \times 10^{-9}$), `Albumin` ($p = 0.0002$), `Albumin_and_Globulin_Ratio` ($p = 0.0004$), and `Age` ($p = 0.0019$) all show statistically significant differences ($p < 0.05$) between liver disease and healthy cohorts.
* **Compensatory Protein Phenomenon:** `Total_Proteins` alone is non-significant ($p = 0.42$) because falling `Albumin` in liver disease is masked by rising serum `Globulin`.
* **Collinear Pairs:** Strong correlations exist between `Total_Bilirubin` & `Direct_Bilirubin` ($r = 0.87$), `AST` & `ALT` ($r = 0.79$), and `Total_Proteins` & `Albumin` ($r = 0.78$).

---

## 5. Model Design Basis, Preprocessing & Feature Engineering

### 5.1 Leakage-Free Preprocessing Pipeline
1. **Deduplication:** Removed 13 duplicate rows prior to splitting (`583` $\rightarrow$ `570` unique patients) so identical patient records cannot span both train and test splits.
2. **Stratified Train-Test Partitioning (`80%` / `20%`):** Preserves the exact class ratio (`~71.4%` positive, `~28.6%` negative) across training (`456` records) and unseen holdout testing (`114` records).
3. **Training-Median Imputation:** Imputed the missing `Albumin_and_Globulin_Ratio` entries using the **training-set median (`0.93`)**, preventing data leakage and avoiding distortion from right skewness (`skew = 0.99`).
4. **Outlier-Resistant Scaling (`RobustScaler`):** Retained all extreme biochemical values (which represent genuine acute pathology) and scaled features using median centering and Interquartile Range (IQR) division:
   $$x_{\text{scaled}} = \frac{x - Q_{50}}{Q_{75} - Q_{25}}$$

### 5.2 Domain-Driven Clinical Feature Engineering
Four clinically validated biomarkers were engineered from raw LFT inputs (expanding features from `10` to `14`):
1. **`Indirect_Bilirubin`:** $\max(\text{Total\_Bilirubin} - \text{Direct\_Bilirubin}, 0.0)$ — Isolates unconjugated bilirubin and reduces collinearity with `Direct_Bilirubin`.
2. **`AST_ALT_Ratio` (De Ritis Ratio):** $\frac{\text{AST}}{\text{ALT} + 10^{-5}}$ — Key hepatological indicator (ratios $< 1.0$ suggest acute viral hepatitis/NAFLD; ratios $> 2.0$ strongly suggest alcoholic liver disease or cirrhosis).
3. **`Globulin`:** $\max(\text{Total\_Proteins} - \text{Albumin}, 0.1)$ — Captures compensatory immunoglobulin elevation masked by `Total_Proteins`.
4. **`Albumin_Protein_Ratio`:** $\frac{\text{Albumin}}{\text{Total\_Proteins} + 10^{-5}}$ — Bounded $[0, 1]$ indicator of hepatic synthetic reserve.

### 5.3 Class Imbalance & Evaluation Metric Strategy
* **Cost-Sensitive Learning (`class_weight='balanced'`):** Penalizes minority-class misclassification inversely proportional to class frequency.
* **Training-Only Minority Oversampling:** Used bootstrap resampling (`sklearn.utils.resample`) strictly inside the training set for algorithms lacking native `class_weight` support (`KNN`, `Gradient Boosting`, `AdaBoost`).
* **Clinical Metric Priority:** Because **False Negatives** (discharging a patient with undiagnosed liver disease) carry far greater clinical risk than **False Positives** (ordering confirmatory imaging/bloodwork), models are evaluated primarily on **ROC-AUC**, **F1-Score**, and **Recall (Sensitivity)**.

---

## 6. Predictive Models & Comparative Evaluation

### 6.1 Summary of Evaluated Algorithms
Eight distinct classification families were benchmarked using 5-fold Stratified Cross-Validation and evaluated on the unseen 20% holdout test set, followed by `GridSearchCV` hyperparameter optimization on top tree ensembles:

| Model Family | Specific Classifiers Implemented | Role & Inductive Bias |
| :--- | :--- | :--- |
| **Linear Models** | Logistic Regression (`class_weight='balanced'`) | Baseline linear log-odds decision boundary with L2 regularization. |
| **Instance / Distance** | K-Nearest Neighbors (`k=7`, distance-weighted) | Non-parametric local neighborhood similarity in RobustScaled space. |
| **Kernel Methods** | Support Vector Machine (`RBF` kernel, `class_weight='balanced'`) | Non-linear maximum-margin boundary with Platt probability scaling. |
| **Single Tree** | Decision Tree (`max_depth=5`, `class_weight='balanced'`) | Interpretable `if/else` threshold rules; baseline for ensemble variance reduction. |
| **Bagging Ensembles** | Random Forest (`Baseline` & `GridSearchCV Tuned`), Extra Trees | Decorrelated decision trees with random feature sub-sampling (`sqrt`). |
| **Boosting Ensembles** | Gradient Boosting (`Baseline` & `GridSearchCV Tuned`), AdaBoost | Sequential stage-wise error correction on hard borderline patients. |

### 6.2 Production Model Recommendation
**Recommended Production Classifier:** **Tuned Random Forest Classifier (`RandomForestClassifier` with `class_weight='balanced'`)**

**Justification for Production Deployment:**
1. **Superior Sensitivity & Discrimination Balance:** Achieves top-tier **ROC-AUC (~0.81–0.83)** and **F1-Score (~0.82)** alongside high **Recall (>81%)**, minimizing missed liver disease diagnoses while maintaining strong cross-validation stability.
2. **Non-Parametric Resilience to Acute Clinical Outliers:** Tree splits partition threshold rules monotonically without distortion from extreme acute hepatitis enzyme spikes (`AST > 4,000 IU/L`).
3. **Immunity to Multicollinearity:** Random feature sub-sampling (`max_features='sqrt'`) cleanly leverages correlated LFT markers (`Total_Bilirubin`, `Indirect_Bilirubin`, `AST`, `ALT`, `AST_ALT_Ratio`) without coefficient instability.
4. **Clinical Explainability & Threshold Flexibility:** Provides transparent Gini feature importance rankings aligned with hepatology guidelines (`Alkaline_Phosphotase`, `AST`, `Total_Bilirubin`, `Indirect_Bilirubin`, `ALT`, `AST_ALT_Ratio`) and supports lowering the probability cutoff (e.g., `threshold = 0.45`) for high-sensitivity screening triage.

---

## 7. Summary of Challenges Faced & Mitigations

| # | Challenge Identified | Risk if Unaddressed | Solution Applied |
| :--- | :--- | :--- | :--- |
| **1** | **Headerless Raw CSV** | First patient record parsed as column headers; 1 record lost. | Loaded with `header=None` and explicit 11-column schema names. |
| **2** | **13 Exact Duplicate Rows** | Train-test data leakage inflating evaluation scores. | Removed duplicates via `drop_duplicates()` before splitting (`583` $\rightarrow$ `570`). |
| **3** | **Missing `A/G Ratio` Values (4 rows)** | Estimator runtime errors or loss of patient records. | Imputed using training-set median (`0.93`) after splitting. |
| **4** | **Extreme Positive Skewness** | Outlier removal discards acute liver injury cases; `StandardScaler` squashes normal variation. | Retained all clinical outliers and applied `RobustScaler` (median/IQR). |
| **5** | **Class Imbalance (`71.4%` vs `28.6%`)** | Majority-class bias and poor minority specificity. | Applied `class_weight='balanced'` and training-only minority resampling. |
| **6** | **High LFT Multicollinearity** | Unstable linear weights and masked protein/enzyme ratios. | Engineered `Indirect_Bilirubin`, `AST_ALT_Ratio`, `Globulin`, and `Albumin_Protein_Ratio`. |
| **7** | **Asymmetric Clinical Error Cost** | Default `0.50` threshold misses borderline sick patients. | Prioritized Recall/ROC-AUC and included clinical threshold sensitivity tuning (`0.35–0.60`). |

---

## 8. Execution & Deployment Instructions

1. **Prerequisites:** Python 3.8+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and `scikit-learn` installed.
2. **Running the Notebook:**
   * Place `PRCP-1007-LiverPatientPred-v2_3.ipynb` in the root `PROJECT 2/` directory alongside the `Data/` folder.
   * Open in Jupyter Notebook, JupyterLab, or VS Code and select **Kernel $\rightarrow$ Restart & Run All**.
3. **Production Scoring:** Use the `predict_liver_disease_risk(patient_record, threshold=0.45)` function in Section 7.2 of the notebook to score incoming patient LFT panels and categorize them into **Low**, **Moderate**, or **High Clinical Risk** tiers.
