#!/usr/bin/env python
"""Sex-Stratified Mediation Analysis: Cognitive-Brain-Alcohol Relationships.

Research Question: How do structural and functional brain networks mediate
the relationship between cognitive traits and alcohol dependence differently
across sexes?

Model:
    Cognitive Trait (X) → Brain Network (M) → Alcohol Outcome (Y)
                                ↑
                         Sex (moderator)

Data Sources:
    - Structural Connectome: TNPCA coefficients from HCP
    - Functional Connectome: TNPCA coefficients from HCP
    - Cognitive Traits: NIH Toolbox measures from table1_hcp.csv
    - Alcohol Measures: SSAGA DSM-IV from table2_hcp.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Key variables
COGNITIVE_VARS = [
    "PicSeq_Unadj",  # Episodic Memory
    "CardSort_Unadj",  # Executive Function
    "Flanker_Unadj",  # Attention/Inhibition
    "PMAT24_A_CR",  # Fluid Intelligence
    "ReadEng_Unadj",  # Language/Reading
    "PicVocab_Unadj",  # Language/Vocabulary
    "ProcSpeed_Unadj",  # Processing Speed
    "ListSort_Unadj",  # Working Memory
]

ALCOHOL_VARS = [
    "SSAGA_Alc_D4_Dp_Dx",  # Alcohol Dependence Diagnosis (DSM-IV)
    "SSAGA_Alc_D4_Dp_Sx",  # Alcohol Dependence Symptoms count
    "SSAGA_Alc_D4_Ab_Dx",  # Alcohol Abuse Diagnosis
    "Total_Drinks_7days",  # Recent drinking
]


class MediationAnalysis:
    """Mediation analysis with bootstrap confidence intervals."""

    def __init__(self, n_bootstrap: int = 5000, confidence: float = 0.95, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.random_state = random_state

    def fit(self, X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> dict:
        """Run mediation analysis.

        Args:
            X: Predictor (cognitive trait)
            M: Mediator (brain network)
            Y: Outcome (alcohol)

        Returns
        -------
            Dictionary with mediation results
        """
        X = np.array(X).reshape(-1, 1)
        M = np.array(M).reshape(-1, 1)
        Y = np.array(Y).ravel()

        # Path a: X → M
        model_a = LinearRegression().fit(X, M.ravel())
        a = model_a.coef_[0]

        # Path b and c': [X, M] → Y
        XM = np.hstack([X, M])
        model_b = LinearRegression().fit(XM, Y)
        c_prime = model_b.coef_[0]  # Direct effect
        b = model_b.coef_[1]  # Path b

        # Path c: X → Y (total effect)
        model_c = LinearRegression().fit(X, Y)
        c = model_c.coef_[0]

        # Indirect effect
        ab = a * b

        # Bootstrap CI
        ab_boots = self._bootstrap(X.ravel(), M.ravel(), Y)
        ci_low = np.percentile(ab_boots, (1 - self.confidence) / 2 * 100)
        ci_high = np.percentile(ab_boots, (1 + self.confidence) / 2 * 100)

        # Proportion mediated
        prop_med = ab / c if c != 0 else np.nan

        return {
            "a": a,
            "b": b,
            "c": c,
            "c_prime": c_prime,
            "indirect": ab,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "prop_mediated": prop_med,
            "significant": (ci_low > 0) or (ci_high < 0),
        }

    def _bootstrap(self, X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        ab_boots = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            X_b, M_b, Y_b = X[idx], M[idx], Y[idx]

            a = LinearRegression().fit(X_b.reshape(-1, 1), M_b).coef_[0]
            XM_b = np.column_stack([X_b, M_b])
            b = LinearRegression().fit(XM_b, Y_b).coef_[1]
            ab_boots.append(a * b)

        result: np.ndarray = np.array(ab_boots)
        return result


def load_connectome_pca(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PCA coefficients from MAT file."""
    mat = loadmat(str(mat_path))

    # Debug: print all keys and shapes
    print(f"    MAT file keys: {[k for k in mat if not k.startswith('_')]}")
    for key in mat:
        if not key.startswith("_"):
            arr = np.array(mat[key])
            print(f"      {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Find coefficient and subject ID keys
    coeff_key = None
    id_key = None
    for key in mat:
        if not key.startswith("_"):
            if "coeff" in key.lower():
                coeff_key = key
            if "subj" in key.lower() or "id" in key.lower():
                id_key = key

    pca_coeff = np.array(mat[coeff_key]) if coeff_key else None
    subject_ids = np.array(mat[id_key]).flatten() if id_key else None

    if pca_coeff is not None:
        print(f"    Original PCA shape: {pca_coeff.shape}")

        # Handle various shapes - squeeze extra dimensions
        pca_coeff = np.squeeze(pca_coeff)
        print(f"    After squeeze: {pca_coeff.shape}")

        # If still 3D, we need to reshape
        if pca_coeff.ndim == 3:
            # Try different interpretations
            # (n_components, n_subjects, 1) -> (n_subjects, n_components)
            if pca_coeff.shape[2] == 1:
                pca_coeff = pca_coeff[:, :, 0].T
            # (1, n_subjects, n_components) -> (n_subjects, n_components)
            elif pca_coeff.shape[0] == 1:
                pca_coeff = pca_coeff[0, :, :]
            # (n_subjects, 1, n_components) -> (n_subjects, n_components)
            elif pca_coeff.shape[1] == 1:
                pca_coeff = pca_coeff[:, 0, :]
            print(f"    After 3D reshape: {pca_coeff.shape}")

        # Ensure shape is (n_subjects, n_components)
        # If more columns than rows, transpose
        if pca_coeff.ndim == 2 and pca_coeff.shape[0] < pca_coeff.shape[1]:
            pca_coeff = pca_coeff.T
            print(f"    After transpose: {pca_coeff.shape}")

    return subject_ids, pca_coeff


def load_hcp_data() -> pd.DataFrame:
    """Load and merge all HCP data."""
    print("Loading HCP data...")

    # Load connectome PCA
    print("  Loading structural connectome PCA...")
    sc_ids, sc_pca = load_connectome_pca(
        RAW_DATA / "TNPCA_Result" / "TNPCA_Coeff_HCP_Structural_Connectome.mat"
    )

    print("  Loading functional connectome PCA...")
    fc_ids, fc_pca = load_connectome_pca(
        RAW_DATA / "TNPCA_Result" / "TNPCA_Coeff_HCP_Functional_Connectome.mat"
    )

    # Load trait tables
    print("  Loading cognitive traits (table1_hcp.csv)...")
    table1 = pd.read_csv(RAW_DATA / "traits" / "table1_hcp.csv")

    print("  Loading alcohol measures (table2_hcp.csv)...")
    table2 = pd.read_csv(RAW_DATA / "traits" / "table2_hcp.csv")

    # Create DataFrames for PCA
    n_sc = min(10, sc_pca.shape[1]) if sc_pca is not None else 0
    n_fc = min(10, fc_pca.shape[1]) if fc_pca is not None else 0

    sc_df = pd.DataFrame(sc_pca[:, :n_sc], columns=[f"SC_PC{i+1}" for i in range(n_sc)])
    sc_df["Subject"] = sc_ids

    fc_df = pd.DataFrame(fc_pca[:, :n_fc], columns=[f"FC_PC{i+1}" for i in range(n_fc)])
    fc_df["Subject"] = fc_ids

    # Merge all data
    cog_cols = [v for v in COGNITIVE_VARS if v in table1.columns]
    alc_cols = [v for v in ALCOHOL_VARS if v in table2.columns]

    data = table1[["Subject", "Gender"] + cog_cols].copy()
    data = data.merge(table2[["Subject"] + alc_cols], on="Subject", how="inner")
    data = data.merge(sc_df, on="Subject", how="inner")
    data = data.merge(fc_df, on="Subject", how="inner")

    print(f"  Final dataset: {len(data)} subjects")
    print(f"  Males: {(data['Gender'] == 'M').sum()}")
    print(f"  Females: {(data['Gender'] == 'F').sum()}")

    return pd.DataFrame(data)


def run_mediation_analysis(
    data: pd.DataFrame,
    n_brain_pcs: int = 10,
    test_all_alcohol: bool = True,
) -> pd.DataFrame:
    """Run comprehensive sex-stratified mediation analysis."""
    print("\n" + "=" * 70)
    print("RUNNING SEX-STRATIFIED MEDIATION ANALYSIS")
    print("=" * 70)

    # Get available variables
    cog_vars = [v for v in COGNITIVE_VARS if v in data.columns]
    alc_vars = [v for v in ALCOHOL_VARS if v in data.columns]
    sc_pcs = [c for c in data.columns if c.startswith("SC_PC")][:n_brain_pcs]
    fc_pcs = [c for c in data.columns if c.startswith("FC_PC")][:n_brain_pcs]
    brain_vars = sc_pcs + fc_pcs

    # Alcohol outcomes to test
    if test_all_alcohol:
        alcohol_outcomes = alc_vars
    else:
        alcohol_outcomes = [
            "SSAGA_Alc_D4_Dp_Sx" if "SSAGA_Alc_D4_Dp_Sx" in data.columns else alc_vars[0]
        ]

    print(f"\nAlcohol outcomes: {alcohol_outcomes}")
    print(f"Cognitive variables: {cog_vars}")
    print(f"Brain network PCs: {len(brain_vars)} ({len(sc_pcs)} SC + {len(fc_pcs)} FC)")

    # Split by sex
    males = data[data["Gender"] == "M"]
    females = data[data["Gender"] == "F"]

    results = []
    total_tests = len(cog_vars) * len(brain_vars) * len(alcohol_outcomes) * 2
    test_num = 0

    for alc_var in alcohol_outcomes:
        print(f"\n  Testing alcohol outcome: {alc_var}")
        for cog_var in cog_vars:
            for brain_var in brain_vars:
                for sex, sex_data in [("Male", males), ("Female", females)]:
                    test_num += 1
                    if test_num % 50 == 0:
                        print(f"    Progress: {test_num}/{total_tests}")

                    subset = sex_data[[cog_var, brain_var, alc_var]].dropna()

                    if len(subset) < 50:
                        continue

                    X = subset[cog_var].values
                    M = subset[brain_var].values
                    Y = subset[alc_var].values

                    # Standardize
                    X = (X - X.mean()) / X.std() if X.std() > 0 else X
                    M = (M - M.mean()) / M.std() if M.std() > 0 else M
                    Y = (Y - Y.mean()) / Y.std() if Y.std() > 0 else Y

                    med = MediationAnalysis(n_bootstrap=2000, random_state=42)
                    res = med.fit(X, M, Y)

                    results.append(
                        {
                            "cognitive": cog_var,
                            "brain": brain_var,
                            "brain_type": "SC" if brain_var.startswith("SC") else "FC",
                            "alcohol_outcome": alc_var,
                            "sex": sex,
                            "n": len(subset),
                            "path_a": res["a"],
                            "path_b": res["b"],
                            "total_effect_c": res["c"],
                            "direct_effect_c_prime": res["c_prime"],
                            "indirect_effect": res["indirect"],
                            "ci_low": res["ci_low"],
                            "ci_high": res["ci_high"],
                            "significant": res["significant"],
                            "prop_mediated": res["prop_mediated"],
                        }
                    )

    return pd.DataFrame(results)


def analyze_sex_differences(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze differences in mediation between sexes."""
    print("\n" + "=" * 70)
    print("SEX DIFFERENCE ANALYSIS")
    print("=" * 70)

    # Pivot to compare male vs female
    comparison = []

    for (cog, brain), group in results_df.groupby(["cognitive", "brain"]):
        male = group[group["sex"] == "Male"]
        female = group[group["sex"] == "Female"]

        if len(male) == 0 or len(female) == 0:
            continue

        m_indirect = male["indirect_effect"].values[0]
        f_indirect = female["indirect_effect"].values[0]
        diff = m_indirect - f_indirect

        comparison.append(
            {
                "cognitive": cog,
                "brain": brain,
                "brain_type": male["brain_type"].values[0],
                "male_indirect": m_indirect,
                "male_sig": male["significant"].values[0],
                "female_indirect": f_indirect,
                "female_sig": female["significant"].values[0],
                "sex_difference": diff,
                "abs_sex_diff": abs(diff),
            }
        )

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values("abs_sex_diff", ascending=False)

    return comparison_df


def create_visualizations(results_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    """Create publication-quality visualizations."""
    print("\nCreating visualizations...")

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Sex comparison bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, brain_type in enumerate(["SC", "FC"]):
        ax = axes[i]
        type_data = results_df[results_df["brain_type"] == brain_type]

        if len(type_data) == 0:
            continue

        pivot = type_data.pivot_table(
            index="cognitive", columns="sex", values="indirect_effect", aggfunc="mean"
        )

        if not pivot.empty:
            x = np.arange(len(pivot.index))
            width = 0.35

            if "Male" in pivot.columns:
                ax.bar(
                    x - width / 2,
                    pivot["Male"],
                    width,
                    label="Male",
                    color="#3498db",
                    edgecolor="black",
                )
            if "Female" in pivot.columns:
                ax.bar(
                    x + width / 2,
                    pivot["Female"],
                    width,
                    label="Female",
                    color="#e74c3c",
                    edgecolor="black",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [c.replace("_Unadj", "").replace("_", "\n") for c in pivot.index],
                rotation=45,
                ha="right",
            )
            ax.set_ylabel("Mean Indirect Effect")
            ax.set_title(
                f"{'Structural' if brain_type == 'SC' else 'Functional'} Connectivity\n"
                "Mediation Effects by Sex"
            )
            ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mediation_sex_comparison.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'mediation_sex_comparison.png'}")

    # 2. Heatmap of effects
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for i, sex in enumerate(["Male", "Female"]):
        ax = axes[i]
        sex_data = results_df[results_df["sex"] == sex]

        pivot = sex_data.pivot_table(
            index="cognitive", columns="brain", values="indirect_effect", aggfunc="mean"
        )

        if not pivot.empty:
            # Shorten labels
            pivot.index = [c.replace("_Unadj", "").replace("_A_CR", "") for c in pivot.index]

            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                annot=True,
                fmt=".3f",
                cbar_kws={"label": "Indirect Effect"},
            )
            ax.set_title(f"{sex}s: Cognitive → Brain → Alcohol\nMediation Effects")
            ax.set_xlabel("Brain Network PC")
            ax.set_ylabel("Cognitive Trait")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mediation_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'mediation_heatmap.png'}")

    # 3. Top sex differences
    fig, ax = plt.subplots(figsize=(12, 6))

    top_diff = comparison_df.head(15)
    colors = ["#3498db" if d > 0 else "#e74c3c" for d in top_diff["sex_difference"]]

    ax.barh(
        range(len(top_diff)),
        top_diff["sex_difference"],
        color=colors,
        edgecolor="black",
    )

    ax.set_yticks(range(len(top_diff)))
    ax.set_yticklabels(
        [
            f"{row['cognitive'].replace('_Unadj', '')} → {row['brain']}"
            for _, row in top_diff.iterrows()
        ]
    )
    ax.set_xlabel("Sex Difference in Indirect Effect (Male - Female)")
    ax.set_title("Top 15 Pathways with Largest Sex Differences in Mediation")
    ax.axvline(0, color="black", linestyle="-", linewidth=1)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="Stronger in Males"),
        Patch(facecolor="#e74c3c", label="Stronger in Females"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mediation_top_sex_differences.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'mediation_top_sex_differences.png'}")

    plt.close("all")


def print_summary(results_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    """Print analysis summary."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    sig_results = results_df[results_df["significant"]]

    print(f"\n📊 Total mediation tests: {len(results_df)}")
    print(f"   Significant (95% CI excludes 0): {len(sig_results)}")
    print(f"   Significance rate: {len(sig_results)/len(results_df)*100:.1f}%")

    # Results by alcohol outcome
    print("\n📈 Results by Alcohol Outcome:")
    for alc_var in results_df["alcohol_outcome"].unique():
        alc_data = results_df[results_df["alcohol_outcome"] == alc_var]
        alc_sig = alc_data[alc_data["significant"]]
        print(f"   {alc_var}: {len(alc_sig)}/{len(alc_data)} significant")

    print("\n👨 Male-specific:")
    male_sig = sig_results[sig_results["sex"] == "Male"]
    print(f"   Significant mediations: {len(male_sig)}")
    if len(male_sig) > 0:
        print(f"   Mean |indirect effect|: {male_sig['indirect_effect'].abs().mean():.4f}")

    print("\n👩 Female-specific:")
    female_sig = sig_results[sig_results["sex"] == "Female"]
    print(f"   Significant mediations: {len(female_sig)}")
    if len(female_sig) > 0:
        print(f"   Mean |indirect effect|: {female_sig['indirect_effect'].abs().mean():.4f}")

    print("\n" + "-" * 70)
    print("TOP SIGNIFICANT MEDIATIONS (by effect size):")
    print("-" * 70)

    if len(sig_results) > 0:
        top_sig = sig_results.nlargest(10, "indirect_effect", keep="first")
        for _, row in top_sig.iterrows():
            print(f"\n{row['cognitive']} → {row['brain']} → Alcohol")
            print(f"  Sex: {row['sex']} (n={row['n']})")
            print(
                f"  Indirect: {row['indirect_effect']:.4f} "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
            print(f"  Prop mediated: {row['prop_mediated']:.1%}")
    else:
        print("No significant mediation effects found.")

    print("\n" + "-" * 70)
    print("LARGEST SEX DIFFERENCES:")
    print("-" * 70)

    for _, row in comparison_df.head(5).iterrows():
        print(f"\n{row['cognitive']} → {row['brain']}")
        m_sig = "sig" if row["male_sig"] else "ns"
        f_sig = "sig" if row["female_sig"] else "ns"
        print(f"  Male indirect:   {row['male_indirect']:.4f} ({m_sig})")
        print(f"  Female indirect: {row['female_indirect']:.4f} ({f_sig})")
        print(f"  Difference (M-F): {row['sex_difference']:.4f}")


def main():
    """Run the complete mediation analysis."""
    print("=" * 70)
    print("SEX-STRATIFIED MEDIATION ANALYSIS")
    print("Cognitive → Brain Network → Alcohol Dependence")
    print("=" * 70)

    # Load data
    data = load_hcp_data()

    # Run mediation with expanded parameters:
    # - Test all 4 alcohol outcomes
    # - Use top 10 PCs from each (SC and FC)
    results_df = run_mediation_analysis(
        data,
        n_brain_pcs=10,
        test_all_alcohol=True,
    )

    # Analyze sex differences
    comparison_df = analyze_sex_differences(results_df)

    # Create visualizations
    create_visualizations(results_df, comparison_df)

    # Print summary
    print_summary(results_df, comparison_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "mediation_results_full.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "mediation_sex_comparison.csv", index=False)

    sig_results = results_df[results_df["significant"]]
    if len(sig_results) > 0:
        sig_results.to_csv(OUTPUT_DIR / "mediation_results_significant.csv", index=False)

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print(f"  {OUTPUT_DIR / 'mediation_results_full.csv'}")
    print(f"  {OUTPUT_DIR / 'mediation_sex_comparison.csv'}")
    print(f"  {OUTPUT_DIR / 'mediation_results_significant.csv'}")
    print(f"  {OUTPUT_DIR / 'mediation_sex_comparison.png'}")
    print(f"  {OUTPUT_DIR / 'mediation_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'mediation_top_sex_differences.png'}")

    return results_df, comparison_df


if __name__ == "__main__":
    results, comparison = main()
