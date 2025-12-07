import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------
# 0. Paths and configs
# ---------------------------------------------------------

INPUT_DIR = "ebm_inputs"
OUT_DIR = "ebm_alcohol_models"
os.makedirs(OUT_DIR, exist_ok=True)

variant_files = {
    "tnpca": os.path.join(INPUT_DIR, "ebm_input_tnpca.csv"),
    "vae":   os.path.join(INPUT_DIR, "ebm_input_vae.csv"),
    "pca":   os.path.join(INPUT_DIR, "ebm_input_pca.csv"),
}

sex_values = ["M", "F"]   # adjust if Gender coding differs

# How many non-cognitive features to keep after selection
TOP_K_NONCOG = 40

# Cognitive feature template (same as in make_ebm_datasets.py)
BASE_COG_CANDIDATES = [
    # Fluid intelligence
    "PMAT24_A_CR",
    "PMAT24_A_SI",
    "PMAT24_A_RTCR",

    # Reading & vocabulary
    "ReadEng_Unadj",
    "ReadEng_AgeAdj",
    "PicVocab_Unadj",
    "PicVocab_AgeAdj",

    # Immediate & delayed word recall
    "IWRD_TOT",
    "IWRD_RTC",

    # Processing speed
    "ProcSpeed_Unadj",
    "ProcSpeed_AgeAdj",

    # Delay discounting
    "DDisc_SV_1mo_200",
    "DDisc_SV_6mo_200",
    "DDisc_SV_1yr_200",
    "DDisc_SV_3yr_200",
    "DDisc_SV_5yr_200",
    "DDisc_SV_10yr_200",
    "DDisc_SV_6mo_40K",
    "DDisc_SV_1yr_40K",
    "DDisc_SV_3yr_40K",
    "DDisc_SV_5yr_40K",
    "DDisc_SV_10yr_40K",
    "DDisc_AUC_200",
    "DDisc_AUC_40K",

    # Visuospatial / mental rotation
    "VSPLOT_TC",
    "VSPLOT_CRTE",
    "VSPLOT_OFF",

    # Sustained attention (SCPT)
    "SCPT_TP",
    "SCPT_TN",
    "SCPT_FP",
    "SCPT_FN",
    "SCPT_TPRT",
    "SCPT_SEN",
    "SCPT_SPEC",
    "SCPT_LRNR",

    # Working memory / list sorting
    "ListSort_Unadj",
    "ListSort_AgeAdj",

    # Episodic memory (picture sequence)
    "PicSeq_Unadj",
    "PicSeq_AgeAdj",

    # Socioeconomic covariates
    "SSAGA_Income",
    "SSAGA_Educ",

    # Executive function & attention
    "CardSort_Unadj",
    "CardSort_AgeAdj",
    "Flanker_Unadj",
    "Flanker_AgeAdj",
]


# ---------------------------------------------------------
# Helper: univariate feature selection on non-cognitive features
# ---------------------------------------------------------

def select_top_k_non_cog(X_train_df, y_train, non_cog_features, k=TOP_K_NONCOG):
    """
    Compute absolute Pearson correlation between each non-cognitive feature
    and y on the TRAIN split, and keep the top k.
    """
    scores = []
    y = y_train.astype(float)

    for col in non_cog_features:
        x = X_train_df[col].values.astype(float)
        mask = ~np.isnan(x)
        if mask.sum() < 10:
            scores.append((col, 0.0))
            continue
        x_mask = x[mask]
        y_mask = y[mask]
        if np.all(x_mask == x_mask[0]):
            scores.append((col, 0.0))
            continue
        corr = np.corrcoef(x_mask, y_mask)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        scores.append((col, abs(corr)))

    scores.sort(key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in scores[:k]]
    return selected, scores


# ---------------------------------------------------------
# 1. Training function for one sex + one variant
# ---------------------------------------------------------

def train_ebm_variant(df_sex, feature_names_all, cog_features, sex_label, variant_label):
    """
    Train an EBM classifier for a given sex + feature set with:
      - feature selection (keep all cognitive features + top K non-cog)
      - CV tuning
      - simple class-imbalance handling via sample weights
    """
    cols = feature_names_all + ["alc_y"]
    sub = df_sex[cols].dropna().copy()

    if sub.empty:
        raise ValueError(f"No rows left after dropping NA for {sex_label}, {variant_label}.")

    # Split into X (DataFrame) and y
    X_df = sub[feature_names_all]
    y = sub["alc_y"].values.astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError(f"Not enough positive/negative cases for {sex_label}, {variant_label}.")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"\n=== Training EBM ({variant_label}) for sex={sex_label} ===")
    print("Train size:", len(y_train), "Test size:", len(y_test))

    # ---------------------------
    # Feature selection step
    # ---------------------------
    cog_features_present = [c for c in cog_features if c in X_train_df.columns]
    non_cog_features = [c for c in X_train_df.columns if c not in cog_features_present]

    selected_non_cog, scores = select_top_k_non_cog(
        X_train_df, y_train, non_cog_features, k=TOP_K_NONCOG
    )

    selected_features = cog_features_present + selected_non_cog
    selected_features = list(dict.fromkeys(selected_features))  # dedupe, keep order

    print(f"  Cognitive features kept: {len(cog_features_present)}")
    print(f"  Non-cognitive features considered: {len(non_cog_features)}")
    print(f"  Non-cognitive features selected: {len(selected_non_cog)}")
    print(f"  Total features used in EBM: {len(selected_features)}")

    # Build numpy arrays for selected features
    X_train = X_train_df[selected_features].values
    X_test = X_test_df[selected_features].values

    # ---------------------------
    # EBM setup + class weighting
    # ---------------------------

    base_model = ExplainableBoostingClassifier(random_state=42)

    # Heavier regularization
    param_grid = {
        "max_leaves": [2, 3],
        "min_samples_leaf": [50, 100],
        "learning_rate": [0.01],
        "outer_bags": [16],
        "max_bins": [32],
        "interactions": [0],  # pure additive model
    }

    # Class weights (inverse frequency on training split)
    class_counts = np.bincount(y_train)
    if len(class_counts) < 2 or class_counts[1] == 0:
        raise ValueError(f"No positive examples in training set for {sex_label}, {variant_label}.")

    n_total = len(y_train)
    w_neg = n_total / (2.0 * class_counts[0])
    w_pos = n_total / (2.0 * class_counts[1])

    sample_weight_train = np.where(y_train == 1, w_pos, w_neg)

    print(f"Class counts (train): 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")
    print(f"Using class weights: w_neg={w_neg:.3f}, w_pos={w_pos:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train, sample_weight=sample_weight_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    cv_best_score = grid.best_score_

    print("Best params:", best_params)
    print("Best CV AUC:", cv_best_score)

    # Predictions
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_proba  = best_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc  = roc_auc_score(y_test,  y_test_proba)

    # Threshold 0.5 for other metrics
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred  = (y_test_proba  >= 0.5).astype(int)

    metrics = {
        "sex": sex_label,
        "variant": variant_label,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_used": len(selected_features),
        "best_params": json.dumps(best_params),
        "cv_best_auc": float(cv_best_score),
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "train_bal_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
        "test_bal_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
        "train_precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "train_recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "train_f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    }

    # Save model
    model_path = os.path.join(OUT_DIR, f"ebm_{sex_label}_{variant_label}.pkl")
    joblib.dump({"model": best_model, "features": selected_features}, model_path)
    print("Saved model (with feature list) to", model_path)

    # ROC curve (test)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{variant_label} (AUC={test_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – sex={sex_label}, variant={variant_label}")
    plt.legend(loc="lower right")
    fig_path = os.path.join(OUT_DIR, f"roc_{sex_label}_{variant_label}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ROC curve to", fig_path)

    # Save ROC points (optional)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_csv_path = os.path.join(OUT_DIR, f"roc_{sex_label}_{variant_label}.csv")
    roc_df.to_csv(roc_csv_path, index=False)

    return metrics


# ---------------------------------------------------------
# 2. Loop over variants and sexes
# ---------------------------------------------------------

all_results = []

for variant_label, filepath in variant_files.items():
    if not os.path.exists(filepath):
        print(f"WARNING: file for variant '{variant_label}' not found at {filepath}, skipping.")
        continue

    print(f"\n========== Variant: {variant_label} ==========")
    df_var = pd.read_csv(filepath)

    # Basic sanity checks
    for col in ["alc_y", "Gender"]:
        if col not in df_var.columns:
            raise ValueError(f"Column '{col}' missing in {filepath}")

    # Build cognitive feature list for this variant
    age_col = None
    if "Age_in_Yrs" in df_var.columns:
        age_col = "Age_in_Yrs"
    elif "Age" in df_var.columns:
        age_col = "Age"

    if age_col is not None:
        cog_candidates = [age_col] + BASE_COG_CANDIDATES
    else:
        cog_candidates = BASE_COG_CANDIDATES

    cog_features = [c for c in cog_candidates if c in df_var.columns]

    # All predictors: everything except Subject/Gender/alc_y
    drop_cols = {"Subject", "Gender", "alc_y"}
    feature_names_all = [c for c in df_var.columns if c not in drop_cols]

    print(f"Total predictors before selection ({variant_label}): {len(feature_names_all)}")
    print(f"Cognitive features present ({variant_label}): {len(cog_features)}")

    for sex in sex_values:
        df_sex = df_var[df_var["Gender"] == sex].copy()
        if df_sex.empty:
            print(f"  Sex={sex}: no rows, skipping.")
            continue

        pos_rate = df_sex["alc_y"].mean()
        print(f"  Sex={sex}, variant={variant_label}, positive rate (alc_y=1): {pos_rate:.3f}")

        try:
            metrics = train_ebm_variant(df_sex, feature_names_all, cog_features, sex, variant_label)
            all_results.append(metrics)
        except ValueError as e:
            print(f"  Skipping sex={sex}, variant={variant_label} due to error: {e}")


# ---------------------------------------------------------
# 3. Save summary metrics
# ---------------------------------------------------------

if all_results:
    res_df = pd.DataFrame(all_results)
    summary_path = os.path.join(OUT_DIR, "ebm_summary_metrics.csv")
    res_df.to_csv(summary_path, index=False)
    print("\nSaved metrics summary to", summary_path)
else:
    print("\nNo models were successfully trained.")
