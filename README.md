# datathon-2025-lifeline-team76 
## Team ID: TM-76
### Team Members  
- Vanessa Stefany Arifianto  
- Menezes Leisha Pritika
- Alexandra Naomira Krisdianto

## Project Overview  
This repository contains our submission for **Datathon 2025 – Lifeline**, which focuses on automated classification of fetal health states using **Cardiotocography (CTG)** data. The objective is to predict fetal state categories — *Normal, Suspect,* or *Pathologic* — from fetal heart rate (FHR) and uterine contraction (UC) features, providing an interpretable decision-support tool for clinicians.

## Dataset  
- **Source:** [UCI Machine Learning Repository – Cardiotocography (CTG)](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
- **Samples:** 2,126 records  
- **Target column:** `NSP` (Normal = 0, Suspect = 1, Pathologic = 2)  
- **Features used:**  
  `LB, AC, FM, UC, ASTV, mSTV, ALTV, mLTV, DL, DS, DP, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency`

## Data Processing Pipeline  
1. **Loading:** Automatically iterates through all sheets in `CTG.xls` to locate the label column (`NSP` or `CLASS`) and normalizes headers.  
2. **Cleaning:** Removes missing values and ensures all selected columns are valid numeric features.  
3. **Transformation:** Encodes class labels {1, 2, 3} → {0, 1, 2} and applies `StandardScaler` to normalize features.  
4. **Balancing:** Addresses class imbalance using the Synthetic Minority Oversampling Technique (SMOTE) within the cross-validation pipeline to prevent data leakage (Chawla et al., 2002).  
5. **Splitting:** Uses a stratified 80/20 train–test split with consistent random state (42).

## Model Design  
### Pipeline  
`StandardScaler → SMOTE → Classifier`

### Models Implemented  
- Logistic Regression: `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`  
- Random Forest: `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`

### Model Selection  
- **Method:** `GridSearchCV` applied to Random Forest  
- **Parameter Grid:**  
  - `n_estimators ∈ {200, 300, 500}`  
  - `max_depth ∈ {None, 6, 10}`  
  - `min_samples_split ∈ {2, 5}`  
- **Scoring Metric:** Macro-F1 (robust against class imbalance)  
- **Validation:** 5-fold `StratifiedKFold` cross-validation  

## Evaluation Metrics  
- **Primary:** Macro-F1 (equal weighting across classes)  
- **Secondary:** Balanced Accuracy  
- **Diagnostics:** Confusion matrix, per-class recall, and classification report  

Example results (replace with actual metrics):  
| Model | Macro-F1 | Balanced Accuracy |
|--------|-----------|------------------|
| Logistic Regression | 0.74 | 0.76 |
| Random Forest (300 trees) | 0.82 | 0.83 |
| Best Random Forest (GridSearchCV) | 0.84 | 0.85 |

## Explainability  
- **Feature Importances:** Variability-related features (ASTV, ALTV, mSTV) and deceleration types (DS, DP) were most influential.  
- **SHAP Analysis:** Confirmed that reduced variability and frequent decelerations increase the likelihood of Pathologic classification, consistent with CTG interpretation guidelines (Ayres-de-Campos et al., 2000).

---

## How to Run  
```bash
# Clone the repository
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]

# Install dependencies
pip install -r requirements.txt

# Run the main script
python ctg_model_pipeline.py
# or open ctg_model_pipeline.ipynb in Jupyter/Colab
