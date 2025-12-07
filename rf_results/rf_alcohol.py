import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

import joblib

# NEW: SMOTE + imblearn pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -------------------------------------------------------------------
# 0. Paths / config
# -------------------------------------------------------------------

INPUT_DIR = "ebm_inputs"          # same as EBM inputs
OUT_DIR = "rf_alcohol_models"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "tnpca": os.path.join(INPUT_DIR, "ebm_input_tnpca.csv"),
    "vae":   os.path.join(INPUT_DIR, "ebm_input_vae.csv"),
    "pca":   os.path.join(INPUT_DIR, "ebm_input_pca.csv"),
}

RANDOM_STATE = 42
TOP_K_VARIMP = 20  # number of top features for varimp plots


# -------------------------------------------------------------------
# 1. Cognitive + connectome feature helpers
# -------------------------------------------------------------------

BASE_COG_FEATURES = [
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

    # SES-ish covariates
    "SSAGA_Income",
    "SSAGA_Educ",

    # Executive / attention
    "CardSort_Unadj",
    "CardSort_AgeAdj",
    "Flanker_Unadj",
    "Flanker_AgeAdj",
]


def get_cog_features(df: pd.DataFrame) -> list:
    """Return cognitive feature columns present in df, including age if available."""
    age_col = None
    if "Age_in_Yrs" in df.columns:
        age_col = "Age_in_Yrs"
    elif "Age" in df.columns:
        age_col = "Age"

    if age_col is not None:
        candidates = [age_col] + BASE_COG_FEATURES
    else:
        candidates = BASE_COG_FEATURES

    return [c for c in candidates if c in df.columns]


def get_connectome_features(df: pd.DataFrame, base_variant: str) -> list:
    """Return connectome features for a given representation."""
    if base_variant == "tnpca":
        cols = [c for c in df.columns if c.startswith("Struct_PC") or c.startswith("Func_PC")]
    elif base_variant == "vae":
        cols = [c for c in df.columns if c.startswith("VAE_Struct_LD") or c.startswith("VAE_Func_LD")]
    elif base_variant == "pca":
        cols = [c for c in df.columns if c.startswith("Raw_Struct_PC") or c.startswith("Raw_Func_PC")]
    else:
        cols = []
    return cols


# -------------------------------------------------------------------
# 2. Training function: RF + CV + regularization + SMOTE
# -------------------------------------------------------------------

def train_rf_for_variant(
    df_sex: pd.DataFrame,
    feature_names: list,
    sex_label: str,
    variant_label: str,
    metrics_records: list,
):
    """
    Train a Random Forest model for a given sex + variant.

    Addresses:
      - overfitting: max_depth, min_samples_leaf, max_features regularize tree size.
      - hyper-parameter tuning: GridSearchCV over a small grid.
      - cross-validation: stratified 5-fold CV on training split only.
      - class imbalance: handled by SMOTE oversampling within the CV/pipeline.
    """
    cols = feature_names + ["alc_y"]
    sub = df_sex[cols].dropna().copy()

    if sub.empty:
        raise ValueError("No rows left after dropping NA for this sex/variant.")

    X = sub[feature_names]
    y = sub["alc_y"].astype(int).values

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least one positive and one negative example.")

    # stratified train/test split (same style as EBM)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\n=== RF: sex={sex_label}, variant={variant_label} ===")
    print("  train size:", len(y_train), "test size:", len(y_test))
    print("  positive rate (train):", y_train.mean().round(3))

    # --- pipeline: impute missing -> SMOTE -> RF ---
    rf = RandomForestClassifier(
        n_estimators=500,           # tuned below; this is just default
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # imputer only uses fit/transform; SMOTE uses fit_resample
    imb_pipe = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", rf),
        ]
    )

    # --- hyper-parameter grid (regularization controls) ---
    param_grid = {
        "rf__n_estimators": [300, 500],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_leaf": [1, 5, 10],
        "rf__max_features": ["sqrt", "log2", 0.3],
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=imb_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    # Fit: SMOTE is applied inside each CV fold and on full training during final fit
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    cv_best_auc = grid.best_score_

    print("  best params:", best_params)
    print("  best CV AUC:", round(cv_best_auc, 3))

    # --- evaluate on train/test (no SMOTE at predict time) ---
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    model_id = f"{sex_label}_{variant_label}"

    metrics_records.append(
        {
            "sex": sex_label,
            "variant": variant_label,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "cv_best_auc": float(cv_best_auc),
            "train_auc": float(train_auc),
            "test_auc": float(test_auc),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "best_params": str(best_params),
        }
    )

    # --- save model ---
    model_path = os.path.join(OUT_DIR, f"rf_{model_id}.joblib")
    joblib.dump({"model": best_model, "features": feature_names}, model_path)
    print("  saved model:", model_path)

    # --- ROC curve plot (test) ---
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Random Forest ROC\nsex={sex_label}, variant={variant_label}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, f"roc_{model_id}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print("  saved ROC:", roc_path)

    # --- variable importance plot (top K) ---
    rf_fitted = best_model.named_steps["rf"]
    importances = rf_fitted.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    topk = feat_imp.head(TOP_K_VARIMP)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=topk.values, y=topk.index, orient="h")
    plt.xlabel("Feature importance (MDI)")
    plt.ylabel("")
    plt.title(f"RF Feature Importance\nsex={sex_label}, variant={variant_label}")
    plt.tight_layout()
    varimp_path = os.path.join(OUT_DIR, f"varimp_{model_id}.png")
    plt.savefig(varimp_path, dpi=150)
    plt.close()
    print("  saved varimp:", varimp_path)


# -------------------------------------------------------------------
# 3. Main loop: mirror EBM model variants
# -------------------------------------------------------------------

def main():
    metrics_records = []

    for base_variant, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"[WARN] missing {csv_path}, skipping base_variant={base_variant}")
            continue

        print(f"\n========== Base dataset: {base_variant} ==========")
        df_var = pd.read_csv(csv_path)

        # basic checks
        if "Gender" not in df_var.columns or "alc_y" not in df_var.columns:
            raise ValueError(f"'Gender' or 'alc_y' missing in {csv_path}")

        # cognitive features
        cog_features = get_cog_features(df_var)

        # connectome features for this representation
        conn_features = get_connectome_features(df_var, base_variant)

        # build variant-specific feature sets (to mirror EBM: full, cog_only, *_only)
        feature_sets = {}

        # full cognitive + this connectome representation
        feature_sets[base_variant] = [c for c in df_var.columns
                                      if c not in {"Subject", "Gender", "alc_y"}]

        # cog_only (only once off tnpca dataset in EBM; we follow same convention)
        if base_variant == "tnpca":
            feature_sets["cog_only"] = cog_features

        # connectome-only variants
        if base_variant == "tnpca":
            feature_sets["tnpca_only"] = conn_features
        elif base_variant == "vae":
            feature_sets["vae_only"] = conn_features
        elif base_variant == "pca":
            feature_sets["pca_only"] = conn_features

        print(f"  cognitive features: {len(cog_features)}")
        print(f"  connectome features ({base_variant}): {len(conn_features)}")

        for sex in ["M", "F"]:
            df_sex = df_var[df_var["Gender"] == sex].copy()
            if df_sex.empty:
                print(f"  [INFO] no rows for sex={sex}, skipping.")
                continue

            pos_rate = df_sex["alc_y"].mean()
            print(f"\n  sex={sex}, positive rate (alc_y=1): {pos_rate:.3f}")

            for variant_label, feats in feature_sets.items():
                if len(feats) == 0:
                    print(f"    [WARN] variant={variant_label} has no features, skipping.")
                    continue

                try:
                    train_rf_for_variant(
                        df_sex=df_sex,
                        feature_names=feats,
                        sex_label=sex,
                        variant_label=variant_label,
                        metrics_records=metrics_records,
                    )
                except ValueError as e:
                    print(f"    [ERROR] sex={sex}, variant={variant_label}: {e}")

    # save metrics summary
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(OUT_DIR, "rf_summary_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print("\nSaved RF metrics summary to:", metrics_path)
    else:
        print("\nNo RF models were successfully trained.")


if __name__ == "__main__":
    main()
