import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 0. Paths / output dir
# ---------------------------------------------------------
DATA_PATH = "full_data.csv"      # change to "Data/full_data.csv" if needed
OUT_DIR = "ebm_inputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load data and construct alc_y from SSAGA_Alc_D4_Ab_Dx
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

if "SSAGA_Alc_D4_Ab_Dx" not in df.columns:
    raise ValueError("Column 'SSAGA_Alc_D4_Ab_Dx' not found in full_data.csv")

# HCP coding: 1 = No diagnosis, 5 = Yes diagnosis
label_raw = df["SSAGA_Alc_D4_Ab_Dx"]

# Map 5 -> 1 (positive), everything else (1 or NA or weird) -> 0
df["alc_y"] = np.where(label_raw == 5, 1, 0).astype(int)

print("alc_y distribution (0 = no/NA, 1 = abuse dx):")
print(df["alc_y"].value_counts(dropna=False))

# ---------------------------------------------------------
# 2. Define cognitive features and common columns
# ---------------------------------------------------------

# Choose age column if present
age_col = None
if "Age_in_Yrs" in df.columns:
    age_col = "Age_in_Yrs"
elif "Age" in df.columns:
    age_col = "Age"

# Broad cognitive set (we'll keep only those that exist)
base_cog_candidates = [
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

    # Socioeconomic covariates (not cognition per se, but useful covariates)
    "SSAGA_Income",
    "SSAGA_Educ",

    # Executive function & attention
    "CardSort_Unadj",
    "CardSort_AgeAdj",
    "Flanker_Unadj",
    "Flanker_AgeAdj",
]

# Prepend age if available
if age_col is not None:
    cog_candidates = [age_col] + base_cog_candidates
else:
    cog_candidates = base_cog_candidates

# Keep only columns that actually exist in the dataframe
cog_features = [c for c in cog_candidates if c in df.columns]

print("\nCognitive features found (after existence check):")
print(cog_features)
print("Total cognitive features:", len(cog_features))

# Common columns to keep in every dataset
common_cols = ["Subject", "Gender", "alc_y"]
common_cols = [c for c in common_cols if c in df.columns]

# ---------------------------------------------------------
# 3. Define connectome feature sets
# ---------------------------------------------------------

# TN-PCA
struct_tnpca = [f"Struct_PC{i}" for i in range(1, 61) if f"Struct_PC{i}" in df.columns]
func_tnpca   = [f"Func_PC{i}"   for i in range(1, 61) if f"Func_PC{i}"   in df.columns]

# VAE
struct_vae = [f"VAE_Struct_LD{i}" for i in range(1, 61) if f"VAE_Struct_LD{i}" in df.columns]
func_vae   = [f"VAE_Func_LD{i}"   for i in range(1, 61) if f"VAE_Func_LD{i}"   in df.columns]

# Raw vector PCA
struct_pca = [f"Raw_Struct_PC{i}" for i in range(1, 61) if f"Raw_Struct_PC{i}" in df.columns]
func_pca   = [f"Raw_Func_PC{i}"   for i in range(1, 61) if f"Raw_Func_PC{i}"   in df.columns]

print("\nConnectome feature groups:")
print("  TN-PCA   :", len(struct_tnpca) + len(func_tnpca), "(Struct:", len(struct_tnpca), ", Func:", len(func_tnpca), ")")
print("  VAE      :", len(struct_vae)   + len(func_vae),   "(Struct:", len(struct_vae),   ", Func:", len(func_vae),   ")")
print("  Raw PCA  :", len(struct_pca)   + len(func_pca),   "(Struct:", len(struct_pca),   ", Func:", len(func_pca),   ")")

# ---------------------------------------------------------
# 4. Build three datasets and save
# ---------------------------------------------------------

datasets = {}

# 4.1 Cognitive + TN-PCA
feats_tnpca = common_cols + cog_features + struct_tnpca + func_tnpca
feats_tnpca = list(dict.fromkeys(feats_tnpca))  # drop duplicates, keep order
datasets["tnpca"] = df[feats_tnpca].copy()

# 4.2 Cognitive + VAE
feats_vae = common_cols + cog_features + struct_vae + func_vae
feats_vae = list(dict.fromkeys(feats_vae))
datasets["vae"] = df[feats_vae].copy()

# 4.3 Cognitive + Raw PCA
feats_pca = common_cols + cog_features + struct_pca + func_pca
feats_pca = list(dict.fromkeys(feats_pca))
datasets["pca"] = df[feats_pca].copy()

# Save each dataset and print basic info
for name, dsub in datasets.items():
    out_path = os.path.join(OUT_DIR, f"ebm_input_{name}.csv")
    dsub.to_csv(out_path, index=False)
    print(f"\nSaved {name} dataset to {out_path}")
    print("  shape:", dsub.shape)
    print("  first 10 columns:", list(dsub.columns[:10]))
