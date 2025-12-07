import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif

# Import Pipeline, SMOTE, and undersampling from imblearn to prevent data leakage
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline

# Configuration
DATA_DIR = "./data/processed" # Ensure this matches your actual folder
TARGET_COL = "alc_y"

# FULL LIST OF COGNITIVE TRAITS (From your data loader)
base_cog_candidates = [
    "PMAT24_A_CR", "PMAT24_A_SI", "PMAT24_A_RTCR",
    "ReadEng_Unadj", "ReadEng_AgeAdj", "PicVocab_Unadj", "PicVocab_AgeAdj",
    "IWRD_TOT", "IWRD_RTC", "ProcSpeed_Unadj", "ProcSpeed_AgeAdj",
    "DDisc_SV_1mo_200", "DDisc_SV_6mo_200", "DDisc_SV_1yr_200", "DDisc_SV_3yr_200",
    "DDisc_SV_5yr_200", "DDisc_SV_10yr_200", "DDisc_SV_6mo_40K", "DDisc_SV_1yr_40K",
    "DDisc_SV_3yr_40K", "DDisc_SV_5yr_40K", "DDisc_SV_10yr_40K", "DDisc_AUC_200", "DDisc_AUC_40K",
    "VSPLOT_TC", "VSPLOT_CRTE", "VSPLOT_OFF",
    "SCPT_TP", "SCPT_TN", "SCPT_FP", "SCPT_FN", "SCPT_TPRT", "SCPT_SEN", "SCPT_SPEC", "SCPT_LRNR",
    "ListSort_Unadj", "ListSort_AgeAdj", "PicSeq_Unadj", "PicSeq_AgeAdj",
    "SSAGA_Income", "SSAGA_Educ",
    "CardSort_Unadj", "CardSort_AgeAdj", "Flanker_Unadj", "Flanker_AgeAdj",
]

def run_svm_pipeline(df, dataset_name, sex_label):
    print(f"\n{'='*60}")
    print(f"RUNNING SVM FOR {sex_label} | DATASET: {dataset_name}")
    print(f"{'='*60}")

    # 1. Feature Selection
    # Auto-detect traits + neuro features
    present_cog_traits = [col for col in base_cog_candidates if col in df.columns]
    
    neuro_features = [col for col in df.columns if 
                      col.startswith("PC") or 
                      col.startswith("Struct_PC") or
                      col.startswith("Func_PC") or
                      col.startswith("VAE_Struct") or 
                      col.startswith("VAE_Func") or
                      col.startswith("Raw_Struct") or 
                      col.startswith("Raw_Func")]
    
    extras = ["Age"] 
    final_extras = [c for c in extras if c in df.columns]
    
    selected_features = present_cog_traits + neuro_features + final_extras
    
    X = df[selected_features]
    y = df[TARGET_COL]
    
    # 2. Split Data FIRST (before any preprocessing to prevent leakage)
    # Stratify ensures consistent ratio of alcohol abuse
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    print(f"Train Shape: {X_train.shape} | Abuse Cases: {sum(y_train)}")
    
    # Check class distribution
    class_dist = y_train.value_counts()
    print(f"Class Distribution (Train): {dict(class_dist)}")

    # 3. Pipeline Setup - Test multiple imbalance handling strategies
    # We'll test: SMOTE (oversampling), Undersampling, and Class Weighting only
    # Note: Resampling MUST go inside the pipeline to avoid CV leakage
    
    # Calculate safe k_neighbors for SMOTE (must be less than minority class size)
    min_class_size = int(class_dist.min())
    smote_k_neighbors = min(5, max(1, min_class_size - 1))  # At least 1, but less than minority class
    
    # Base steps common to all pipelines
    # Order: Impute → Scale → Select Features → (Resample) → Model
    # All preprocessing in pipeline prevents data leakage in CV
    base_steps = [
        ('imputer', SimpleImputer(strategy="mean")),  # Handle missing values (fit only on training folds)
        ('scaler', StandardScaler()),  # Standardize features (mean=0, std=1) - CRITICAL for SVM
        ('selector', SelectKBest(f_classif, k=50)), # Pre-select top 50 features to reduce noise
    ]
    
    # Create three different pipelines to test
    def create_pipeline(resampling_strategy='weighting'):
        """
        resampling_strategy: 'smote', 'undersample', or 'weighting'
        """
        steps = base_steps.copy()
        
        if resampling_strategy == 'smote':
            steps.append(('resampler', SMOTE(random_state=123, k_neighbors=smote_k_neighbors)))
            steps.append(('svm', SVC(probability=True, class_weight='balanced', random_state=123)))
            return ImbPipeline(steps)
        elif resampling_strategy == 'undersample':
            steps.append(('resampler', RandomUnderSampler(random_state=123)))
            steps.append(('svm', SVC(probability=True, class_weight='balanced', random_state=123)))
            return ImbPipeline(steps)
        else:  # 'weighting' - no resampling, just class weights
            steps.append(('svm', SVC(probability=True, class_weight='balanced', random_state=123)))
            return SklearnPipeline(steps)
    
    # Test all three approaches
    print("\nTesting imbalance handling strategies:")
    print("  1. SMOTE (oversampling)")
    print("  2. Undersampling")
    print("  3. Class weighting only (no resampling)")
    
    pipelines = {
        'smote': create_pipeline('smote'),
        'undersample': create_pipeline('undersample'),
        'weighting': create_pipeline('weighting')
    }

    # 4. Grid Search
    # We reference pipeline steps using double underscores (e.g., svm__C)
    param_grid = {
        'selector__k': [20, 50, 100], # Test different feature counts
        'svm__C': [0.1, 1, 10, 50, 100],  # Added lower C values for better regularization
        'svm__gamma': ['scale', 0.01, 0.001, 0.0001],
        'svm__kernel': ['rbf', 'poly', 'linear']  # Added linear kernel
    }
    
    # Use StratifiedKFold for better class balance in CV
    cv_folds = min(5, class_dist.min())  # Ensure each fold has at least 1 sample per class
    cv_folds = max(3, cv_folds)  # At least 3 folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=123)
    
    # Test all three imbalance handling strategies
    best_grid = None
    best_score = -np.inf
    best_strategy = None
    
    for strategy_name, pipeline in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Testing {strategy_name.upper()} strategy...")
        print(f"{'='*60}")
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            refit='roc_auc', 
            verbose=0, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        print(f"  Best CV AUC: {grid.best_score_:.4f}")
        print(f"  Best Params: {grid.best_params_}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_grid = grid
            best_strategy = strategy_name
    
    print(f"\n{'='*60}")
    print(f"BEST STRATEGY: {best_strategy.upper()} (CV AUC: {best_score:.4f})")
    print(f"{'='*60}")
    
    grid = best_grid
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")

    # 6. Evaluate
    y_prob = grid.predict_proba(X_test)[:, 1]
    
    # Threshold Moving
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\nOptimal Threshold: {best_thresh:.4f}")
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred_opt)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
        
    print(f"Accuracy (Opt): {acc:.4f} | AUC: {auc:.4f} | F1 (Opt): {best_f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_opt))

    # 7. Feature Importance
    # Permutation importance on the BEST pipeline
    perm = permutation_importance(grid.best_estimator_, X_test, y_test, n_repeats=5, random_state=123, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()[::-1][:10]
    
    feature_names = list(X.columns) 
    print("\nTop 5 Predictors:")
    for i in sorted_idx[:5]:
        if i < len(feature_names):
            print(f"  {feature_names[i]}: {perm.importances_mean[i]:.4f}")
        
    return {
        "sex": sex_label,
        "name": dataset_name,
        "strategy": best_strategy,
        "accuracy": acc,
        "auc": auc,
        "f1": best_f1
    }

# Main Execution Flow
search_pattern = os.path.join(DATA_DIR, "ebm_input*.csv")
file_list = glob.glob(search_pattern)
results = []
dfs = {}

if not file_list:
    print(f"ERROR: No files found matching {search_pattern}")
else:
    for filepath in file_list:
        name = os.path.basename(filepath).replace(".csv", "").replace("ebm_input_", "")
        df_temp = pd.read_csv(filepath)
        dfs[name] = df_temp
        
        # Split by Sex
        df_m = df_temp[df_temp["Gender"] == "M"]
        df_f = df_temp[df_temp["Gender"] == "F"]
        
        res_m = run_svm_pipeline(df_m, name, "MALE")
        res_f = run_svm_pipeline(df_f, name, "FEMALE")
        results.append(res_m)
        results.append(res_f)

    # Grandmaster Dataset
    keys = list(dfs.keys())
    if keys:
        df_final = dfs[keys[0]].copy()
        for k in keys[1:]:
            potential_merge_keys = ["Subject", TARGET_COL, "Gender", "Age"] + base_cog_candidates
            actual_merge_keys = [c for c in potential_merge_keys if c in df_final.columns and c in dfs[k].columns]
            df_final = pd.merge(df_final, dfs[k], on=actual_merge_keys, how="inner")
        
        df_final_m = df_final[df_final["Gender"] == "M"]
        df_final_f = df_final[df_final["Gender"] == "F"]
        
        res_final_m = run_svm_pipeline(df_final_m, "ALL_COMBINED", "MALE")
        res_final_f = run_svm_pipeline(df_final_f, "ALL_COMBINED", "FEMALE")
        
        results.append(res_final_m)
        results.append(res_final_f)

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY (SEX-STRATIFIED)")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results)[["sex", "name", "strategy", "accuracy", "auc", "f1"]]
    print(summary_df.sort_values(["sex", "auc"], ascending=False))