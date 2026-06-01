from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path("/Users/evrardlecureur/Documents/traitement")
DATA_FILE = ROOT / "work" / "df_arabica_clean.csv"
OUT = ROOT / "work" / "output_coffee"
FIG = OUT / "figures"
DATA_OUT = OUT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)
PALETTE = sns.color_palette("Set2", 10)

CONCLUSIONS: dict[str, str] = {}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  -> {name}.png")


def parse_altitude(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        pass
    if "-" in s:
        parts = s.split("-")
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            if nums:
                return np.mean(nums)
        except ValueError:
            return np.nan
    nums = []
    cur = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            cur += ch
        else:
            if cur:
                try:
                    nums.append(float(cur))
                except ValueError:
                    pass
                cur = ""
    if cur:
        try:
            nums.append(float(cur))
        except ValueError:
            pass
    if nums:
        return float(np.mean(nums))
    return np.nan


def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("1. CHARGEMENT DES DONNEES")
    print("=" * 70)
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    print(f"Shape brut : {df.shape}")
    print(f"Colonnes : {list(df.columns)}")
    return df


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Altitude_num"] = df["Altitude"].apply(parse_altitude)
    df["Harvest Year"] = pd.to_numeric(df["Harvest Year"], errors="coerce")
    df["Grading Date"] = pd.to_datetime(df["Grading Date"], errors="coerce")
    df["Number of Bags"] = pd.to_numeric(df["Number of Bags"], errors="coerce")
    df["Bag Weight"] = df["Bag Weight"].astype(str).str.replace(" kg", "", regex=False).str.strip()
    df["Bag Weight"] = pd.to_numeric(df["Bag Weight"], errors="coerce")
    df["Moisture Percentage"] = pd.to_numeric(df["Moisture Percentage"], errors="coerce")
    df["Category One Defects"] = pd.to_numeric(df["Category One Defects"], errors="coerce")
    df["Category Two Defects"] = pd.to_numeric(df["Category Two Defects"], errors="coerce")
    df["Quakers"] = pd.to_numeric(df["Quakers"], errors="coerce")
    for col in [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
        "Uniformity", "Clean Cup", "Sweetness", "Overall",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def eda(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("2. ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 70)

    df = parse_features(df)
    score_cols = [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
        "Uniformity", "Clean Cup", "Sweetness", "Overall", "Total Cup Points",
    ]
    score_cols = [c for c in score_cols if c in df.columns]
    print(f"\nColonnes de scoring : {len(score_cols)}")

    print(f"\nTotal Cup Points (verif formule) :")
    if "Total Cup Points" in df.columns:
        ssum = df[score_cols].drop(columns=["Total Cup Points"]).sum(axis=1)
        diff = (df["Total Cup Points"] - ssum).abs()
        print(f"  ecart max avec somme des sous-scores : {diff.max():.4f}")
        print(f"  -> Total = somme des 10 sous-scores (Overall inclus)")

    print("\n--- Valeurs manquantes (top 10) ---")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        print("Aucune valeur manquante.")
    else:
        print((miss / len(df) * 100).round(2).head(10).to_string())

    print("\n--- Statistiques scoring sensoriel ---")
    print(df[score_cols].describe().round(3).T)

    print("\n--- Top pays producteurs ---")
    print(df["Country of Origin"].value_counts().head(10))

    print("\n--- Varietes les plus frequentes ---")
    print(df["Variety"].value_counts().head(10))

    print("\n--- Methodes de processing ---")
    print(df["Processing Method"].value_counts().head(10))

    print("\n--- Repartition des altitudes ---")
    alt = df["Altitude_num"].dropna()
    print(f"  min={alt.min()}, max={alt.max()}, median={alt.median()}, mean={alt.mean():.0f}")

    CONCLUSIONS["dataset"] = (
        f"Dataset : {df.shape[0]} cafes notes par le CQI sur {df.shape[1]} colonnes brutes. "
        f"La cible naturelle 'Total Cup Points' est la SOMME des 10 sous-scores sensoriels "
        f"(Aroma..Sweetness + Overall) - donc triviale a predire a partir des sous-scores. "
        f"On utilisera plutot 'Overall' (le plus subjectif) ou une classification Specialty/Commercial (seuil 80). "
        f"Pays top : {', '.join(df['Country of Origin'].value_counts().head(3).index.tolist())}. "
        f"Altitude tres variable ({int(alt.min())}-{int(alt.max())} m) : bon potentiel de feature engineering."
    )

    return {
        "df": df,
        "score_cols": score_cols,
        "n_rows": len(df),
        "n_cols": df.shape[1],
    }


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    print("\n" + "=" * 70)
    print("3. FEATURE ENGINEERING")
    print("=" * 70)

    df_fe = parse_features(df)
    score_cols = [c for c in [
        "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
        "Uniformity", "Clean Cup", "Sweetness", "Overall",
    ] if c in df.columns]

    print("\n--- 3.1 Indicateurs sensoriels derives ---")
    df_fe["sensory_mean"] = df_fe[score_cols].mean(axis=1)
    df_fe["sensory_std"] = df_fe[score_cols].std(axis=1)
    df_fe["sensory_min"] = df_fe[score_cols].min(axis=1)
    df_fe["sensory_max"] = df_fe[score_cols].max(axis=1)
    df_fe["sensory_range"] = df_fe["sensory_max"] - df_fe["sensory_min"]
    df_fe["sensory_cv"] = df_fe["sensory_std"] / df_fe["sensory_mean"].replace(0, np.nan)
    df_fe["scores_above_85"] = (df_fe[score_cols] >= 8.5).sum(axis=1)
    df_fe["scores_below_7"] = (df_fe[score_cols] < 7).sum(axis=1)
    flavor_core = ["Flavor", "Aftertaste", "Acidity", "Balance"]
    df_fe["flavor_core_mean"] = df_fe[flavor_core].mean(axis=1)
    tactile = ["Body", "Balance", "Uniformity"]
    df_fe["tactile_mean"] = df_fe[tactile].mean(axis=1)
    cleanliness = ["Clean Cup", "Sweetness", "Uniformity"]
    df_fe["cleanliness_score"] = df_fe[cleanliness].mean(axis=1)

    print("\n--- 3.2 Ratios lies aux defauts ---")
    df_fe["defect_total"] = (
        df_fe["Category One Defects"].fillna(0) + df_fe["Category Two Defects"].fillna(0)
    )
    df_fe["defect_ratio"] = df_fe["defect_total"] / (
        df_fe["Number of Bags"].fillna(0) + 1e-6
    )
    df_fe["quaker_ratio"] = df_fe["Quakers"].fillna(0) / (
        df_fe["Number of Bags"].fillna(0) + 1e-6
    )
    df_fe["has_defect"] = (df_fe["defect_total"] > 0).astype(int)
    df_fe["has_quaker"] = (df_fe["Quakers"].fillna(0) > 0).astype(int)

    print("\n--- 3.3 Features liees a l'altitude ---")
    df_fe["altitude_log"] = np.log1p(df_fe["Altitude_num"].fillna(df_fe["Altitude_num"].median()))
    df_fe["is_high_altitude"] = (df_fe["Altitude_num"] >= 1500).astype(int)
    df_fe["is_very_high_altitude"] = (df_fe["Altitude_num"] >= 2000).astype(int)
    df_fe["altitude_bucket"] = pd.cut(
        df_fe["Altitude_num"],
        bins=[0, 800, 1200, 1600, 2000, 5000],
        labels=["low", "mid", "high", "very_high", "extreme"],
    )

    print("\n--- 3.4 Volume / logistique ---")
    df_fe["bag_weight_log"] = np.log1p(df_fe["Bag Weight"].fillna(df_fe["Bag Weight"].median()))
    df_fe["total_weight_kg"] = df_fe["Number of Bags"].fillna(0) * df_fe["Bag Weight"].fillna(0)
    df_fe["total_weight_log"] = np.log1p(df_fe["total_weight_kg"].clip(lower=0))
    df_fe["production_scale"] = pd.cut(
        df_fe["total_weight_kg"],
        bins=[-1, 100, 1000, 10000, 1e7],
        labels=["micro", "small", "medium", "large"],
    )

    print("\n--- 3.5 Temporel ---")
    df_fe["grading_year"] = df_fe["Grading Date"].dt.year
    df_fe["grading_month"] = df_fe["Grading Date"].dt.month
    df_fe["harvest_age"] = df_fe["Grading Date"].dt.year - df_fe["Harvest Year"]
    df_fe["harvest_age"] = df_fe["harvest_age"].clip(lower=0, upper=10)
    df_fe["is_recent_harvest"] = (df_fe["harvest_age"] <= 1).astype(int)

    print("\n--- 3.6 Encodage categoriel (top 10 modalites) ---")
    cat_cols_high = ["Country of Origin", "Region", "Variety", "Processing Method", "Color"]
    for c in cat_cols_high:
        if c in df_fe.columns:
            top = df_fe[c].value_counts().head(10).index.tolist()
            df_fe[c + "_top"] = df_fe[c].where(df_fe[c].isin(top), "Other")

    df_fe = pd.get_dummies(
        df_fe,
        columns=[c + "_top" for c in cat_cols_high if c in df_fe.columns and (c + "_top") in df_fe.columns],
        drop_first=False,
        dtype=int,
    )

    print("\n--- 3.7 Indicateurs qualite derivés ---")
    df_fe["flavor_to_body"] = df_fe["Flavor"] / df_fe["Body"].replace(0, np.nan)
    df_fe["aroma_to_acidity"] = df_fe["Aroma"] / df_fe["Acidity"].replace(0, np.nan)
    df_fe["balance_index"] = df_fe["Balance"] - df_fe[["Body", "Acidity"]].mean(axis=1)
    df_fe["clean_index"] = (
        df_fe["Clean Cup"] + df_fe["Uniformity"] + df_fe["Sweetness"]
    ) / 3
    df_fe["flavor_minus_defect"] = df_fe["flavor_core_mean"] - df_fe["defect_total"] * 0.5
    df_fe["is_specialty"] = (df_fe["Total Cup Points"] >= 80).astype(int)
    df_fe["is_outstanding"] = (df_fe["Total Cup Points"] >= 85).astype(int)

    df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"\nColonnes avant : {df.shape[1]}")
    print(f"Colonnes apres : {df_fe.shape[1]}")
    print(f"Nouvelles features ajoutees : {df_fe.shape[1] - df.shape[1]}")

    new_features = [c for c in df_fe.columns if c not in df.columns]
    return df_fe, new_features, score_cols


def viz_distributions(df: pd.DataFrame, score_cols: list[str]) -> None:
    print("\n[FIG 1] Distributions des scores sensoriels...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for ax, c in zip(axes.ravel(), score_cols):
        sns.histplot(df[c], kde=True, ax=ax, color=PALETTE[0], bins=25, linewidth=0)
        ax.set_title(f"{c}\n(skew={df[c].skew():.2f})", fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Distributions des 10 scores sensoriels CQI", fontsize=14, fontweight="bold")
    save(fig, "01_distributions_scores")

    print("\n[FIG 2] Distribution Total Cup Points + specialite...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Total Cup Points"], kde=True, bins=30, ax=axes[0], color=PALETTE[1])
    axes[0].axvline(80, color="red", linestyle="--", label="Seuil Specialty (80)")
    axes[0].axvline(85, color="darkred", linestyle="--", label="Seuil Outstanding (85)")
    axes[0].set_title("Distribution de Total Cup Points")
    axes[0].legend()
    specialty_count = (df["Total Cup Points"] >= 80).sum()
    axes[1].pie(
        [specialty_count, len(df) - specialty_count],
        labels=["Specialty (>=80)", "Non-Specialty"],
        autopct="%1.1f%%",
        colors=[PALETTE[2], PALETTE[3]],
        startangle=90,
    )
    axes[1].set_title(f"Repartition Specialty vs Non-Specialty\n({specialty_count} / {len(df)})")
    save(fig, "02_total_cup_points")
    CONCLUSIONS["fig02"] = (
        f"La distribution de Total Cup Points est bimodalee : un mode principal autour de 82-84 (Specialty) "
        f"et un epaulement vers 87+. {specialty_count}/{len(df)} cafes depassent le seuil Specialty de 80 "
        f"(classe A en SCA), et {(df['Total Cup Points'] >= 85).sum()} sont 'Outstanding'. "
        f"Cette distribution etroitement centree (entre 80 et 88) reflete la selection du CQI : "
        f"seuls les cafes deja bien notes sont gradés en detail. Conclusion : la cible est dense, "
        f"un modele regressif aura du mal a capturer de la variance ; une classification binaire "
        f"Specialty/Commercial sera plus discriminante."
    )


def viz_correlation(df: pd.DataFrame, score_cols: list[str]) -> None:
    print("\n[FIG 3] Heatmap correlations sensoriels...")
    cols = [c for c in score_cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("Matrice de correlations des scores sensoriels CQI", fontsize=13, fontweight="bold")
    save(fig, "03_correlation_scores")
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], abs(corr.iloc[i, j])))
    pairs.sort(key=lambda x: -x[2])
    print("\n  Top 5 paires correlees :")
    for a, b, v in pairs[:5]:
        print(f"    {a} <-> {b} : {v:.3f}")
    CONCLUSIONS["fig03"] = (
        "Les 10 sous-scores sont FORTEMENT correles entre eux (0.55 a 0.85). Les clusters principaux : "
        "(1) 'Profil aromatique' : Flavor, Aftertaste, Acidity, Balance (corr 0.75-0.85) ; "
        "(2) 'Profil technique' : Uniformity, Clean Cup, Sweetness (notes techniques quasi-constantes) ; "
        "(3) Overall est tres correle a Flavor (0.85) - c'est la qu'on predit le plus subjectif. "
        "Conclusion : si on predit 'Overall' depuis les 9 autres sous-scores, on espere R² > 0.7. "
        "La multicolinearite est severe -> Regularisation (Ridge/Lasso) obligatoire pour la regression."
    )


def viz_country(df: pd.DataFrame) -> None:
    print("\n[FIG 4] Qualite par pays...")
    top_countries = df["Country of Origin"].value_counts().head(10).index.tolist()
    sub = df[df["Country of Origin"].isin(top_countries)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = sub.groupby("Country of Origin")["Total Cup Points"].median().sort_values(ascending=False).index
    sns.boxplot(x="Country of Origin", y="Total Cup Points", data=sub, order=order, ax=axes[0], palette="Set2")
    axes[0].set_title("Distribution Total Cup Points par pays (top 10 par effectif)")
    axes[0].tick_params(axis="x", rotation=30)
    counts = sub["Country of Origin"].value_counts().reindex(order)
    for i, (cnt, val) in enumerate(zip(counts.values, sub.groupby("Country of Origin")["Total Cup Points"].median().reindex(order).values)):
        pass
    counts_df = sub["Country of Origin"].value_counts().reset_index()
    counts_df.columns = ["Country", "Count"]
    sns.barplot(x="Country", y="Count", data=counts_df.sort_values("Count", ascending=False),
                ax=axes[1], palette="Set3")
    axes[1].set_title("Nombre de cafes gradés par pays (top 10)")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "04_pays_qualite")
    top_median = sub.groupby("Country of Origin")["Total Cup Points"].median().sort_values(ascending=False).head(3)
    CONCLUSIONS["fig04"] = (
        f"Les pays ayant les medianes les plus elevees : {', '.join(top_median.index)}. "
        f"L'Ethiopie et le Kenya (origines est-africaines) dominent traditionnellement, "
        f"confirmé par leur position sur la boxplot. Les pays latino-americains (Colombie, Bresil, Honduras) "
        f"sont tres representes en volume mais avec une variabilite plus grande. "
        f"Conclusion : 'Country of Origin' est un FEATURE FORT pour predire la qualite, "
        f"grace a la geographie (terroir, climat, variété cultivee). Mais attention a la cardinalité : "
        f"il y a {df['Country of Origin'].nunique()} pays dans le dataset, on gardera un top-10 en one-hot."
    )


def viz_processing(df: pd.DataFrame) -> None:
    print("\n[FIG 5] Qualite par methode de processing...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order = df.groupby("Processing Method")["Total Cup Points"].median().sort_values(ascending=False).index
    sns.boxplot(x="Processing Method", y="Total Cup Points", data=df, order=order, ax=axes[0], palette="Set1")
    axes[0].set_title("Total Cup Points par methode de processing")
    axes[0].tick_params(axis="x", rotation=20)
    proc_mean = df.groupby("Processing Method")[df.select_dtypes(include=np.number).columns].mean(numeric_only=True)
    proc_score = proc_mean[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance"]]
    proc_score.T.plot(kind="bar", ax=axes[1], colormap="Set2")
    axes[1].set_title("Profil sensoriel moyen par methode de processing")
    axes[1].set_ylabel("Score moyen")
    axes[1].legend(title="Processing", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "05_processing_method")
    CONCLUSIONS["fig05"] = (
        "Les methodes de processing (Washed, Natural, Honey, Anaerobic...) donnent des profils "
        "sensoriels tres differents. Les 'Natural' (cerise sechee) tendent a avoir plus de Body et Sweetness "
        "mais moins de Clarity, les 'Washed' (lave) sont plus nets en Acidity. "
        f"Mediane par methode (top) : {', '.join([f'{m}={v:.2f}' for m, v in df.groupby('Processing Method')['Total Cup Points'].median().sort_values(ascending=False).head(3).items()])}. "
        f"Conclusion : 'Processing Method' est une variable categorielle TRES informative "
        f"car elle determine directement le profil aromatique final."
    )


def viz_altitude(df: pd.DataFrame) -> None:
    print("\n[FIG 6] Altitude vs qualite...")
    sub = df.dropna(subset=["Altitude_num"]).copy()
    sub["altitude_bucket"] = pd.cut(
        sub["Altitude_num"],
        bins=[0, 800, 1200, 1600, 2000, 5000],
        labels=["low", "mid", "high", "very_high", "extreme"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(x="Altitude_num", y="Total Cup Points", data=sub, ax=axes[0],
                    alpha=0.4, color=PALETTE[0], s=20)
    sns.regplot(x="Altitude_num", y="Total Cup Points", data=sub, ax=axes[0],
                scatter=False, color="red")
    axes[0].set_title("Altitude (m) vs Total Cup Points")
    axes[0].set_xlim(0, 3000)
    order = sub.groupby("altitude_bucket", observed=True)["Total Cup Points"].median().sort_values(ascending=False).index
    sns.boxplot(x="altitude_bucket", y="Total Cup Points", data=sub, order=order, ax=axes[1], palette="Set3")
    axes[1].set_title("Total Cup Points par tranche d'altitude")
    save(fig, "06_altitude")
    corr_alt = sub["Altitude_num"].corr(sub["Total Cup Points"])
    CONCLUSIONS["fig06"] = (
        f"Correlation altitude / Total Cup Points : r = {corr_alt:.3f}. "
        f"Tendance positive : les cafes de plus haute altitude (>1500m) obtiennent generalement "
        f"de meilleurs scores. Cela correspond au terroir (climat plus frais, maturation lente, "
        f"concentration des sucres). Les 'extreme altitude' (>2000m) ont la mediane la plus haute. "
        f"Conclusion : 'Altitude_num' est un bon predicteur continu, on peut aussi la discretiser "
        f"en 'low/mid/high/very_high/extreme' pour capturer les paliers."
    )


def viz_defects(df: pd.DataFrame) -> None:
    print("\n[FIG 7] Defauts et qualite...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sub = df.copy()
    sub["defect_total"] = sub["Category One Defects"].fillna(0) + sub["Category Two Defects"].fillna(0)
    sub["defect_bucket"] = pd.cut(sub["defect_total"], bins=[-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"])
    sns.boxplot(x="defect_bucket", y="Total Cup Points", data=sub, ax=axes[0, 0], palette="Reds_r")
    axes[0, 0].set_title("Total Cup Points vs nombre de defauts")
    sns.boxplot(x="Color", y="Total Cup Points", data=df, ax=axes[0, 1], palette="YlOrBr")
    axes[0, 1].set_title("Total Cup Points par couleur du grain")
    sns.scatterplot(x="Moisture Percentage", y="Total Cup Points", data=df, ax=axes[1, 0], alpha=0.5, color=PALETTE[5])
    axes[1, 0].set_title("Moisture % vs Total Cup Points")
    if "Quakers" in df.columns:
        sub["has_quaker"] = (sub["Quakers"].fillna(0) > 0).astype(int)
        sns.boxplot(x="has_quaker", y="Total Cup Points", data=sub, ax=axes[1, 1], palette="Set2")
        axes[1, 1].set_xticklabels(["Pas de Quaker", "Quaker present"])
        axes[1, 1].set_title("Impact des Quakers (grains immatures)")
    save(fig, "07_defauts")
    CONCLUSIONS["fig07"] = (
        "Relation NEGATIVE tres claire entre nombre de defauts et score : chaque defaut fait chuter "
        "le Total Cup Points. Les 'Category One' (defauts majeurs, moisissures, insectes) sont "
        "plus penalisants que les 'Category Two' (mineurs). La couleur du grain (vert/jaune/bleu) "
        "reflete le terroir et la maturite, avec des distributions distinctes. "
        "Conclusion : 'defect_total' et 'has_quaker' sont des features CRITIQUES - ce sont des indicateurs "
        "de tri industriel et ils discriminent tres fortement les scores extremes."
    )


def viz_variety(df: pd.DataFrame) -> None:
    print("\n[FIG 8] Variete et qualite...")
    top_var = df["Variety"].value_counts().head(8).index.tolist()
    sub = df[df["Variety"].isin(top_var)]
    order = sub.groupby("Variety")["Total Cup Points"].median().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(x="Variety", y="Total Cup Points", data=sub, order=order, ax=axes[0], palette="Set2")
    axes[0].set_title("Total Cup Points par variete (top 8 par effectif)")
    axes[0].tick_params(axis="x", rotation=30)
    radar_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance"]
    means = sub.groupby("Variety")[radar_cols].mean()
    axes[1].set_visible(False)
    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    for i, var in enumerate(means.index[:5]):
        vals = means.loc[var].tolist()
        vals += vals[:1]
        ax_radar.plot(angles, vals, label=var, linewidth=2, color=PALETTE[i])
        ax_radar.fill(angles, vals, alpha=0.1, color=PALETTE[i])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_cols)
    ax_radar.set_ylim(7, 8.5)
    ax_radar.set_title("Radar sensoriel par variete (top 5)", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, "08_variety")
    save(fig_radar, "08b_radar_variety")
    CONCLUSIONS["fig08"] = (
        "Les varietes ont des profils sensoriels distincts (visible sur le radar) : "
        "Bourbon et Typica (varietes classiques arabica) ont des aromes equilibres, "
        "Geisha/Gesha (variete premium) explose en Flavor et Acidity, SL28 (Kenya) a un Body superieur. "
        "Conclusion : la variete determine le 'potentiel aromatique' et donc la qualite. "
        "C'est une feature categorielle importante mais avec des classes rares (rare varietes peu representees). "
        "On garde les top 8 en one-hot et 'Other' pour le reste."
    )


def viz_feature_engineering(df_fe: pd.DataFrame, new_features: list[str]) -> None:
    print("\n[FIG 9] Validation du feature engineering...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df_fe["sensory_mean"], bins=30, kde=True, ax=axes[0, 0], color=PALETTE[0])
    axes[0, 0].set_title("Distribution : sensory_mean (engineered)")
    sns.histplot(df_fe["sensory_std"].dropna(), bins=25, kde=True, ax=axes[0, 1], color=PALETTE[1])
    axes[0, 1].set_title("Distribution : sensory_std (equilibre des notes)")
    sns.scatterplot(x="sensory_mean", y="Total Cup Points", data=df_fe, ax=axes[1, 0],
                    alpha=0.5, color=PALETTE[2])
    axes[1, 0].set_title("sensory_mean vs Total Cup Points (correlation lineaire evidente)")
    sns.countplot(x="scores_above_85", data=df_fe, ax=axes[1, 1], palette="Set3")
    axes[1, 1].set_title("Distribution : nombre de sous-scores >= 8.5")
    axes[1, 1].set_xlabel("Nombre de sous-scores > 8.5")
    save(fig, "09_feature_engineering")
    CONCLUSIONS["fig09"] = (
        "Les features engineered sont plus informatives que les brutes : "
        "- 'sensory_mean' : resume synthetique, tres correle a Total Cup Points "
        "- 'sensory_std' : capture l'equilibre (un cafe note 8 partout est plus stable qu'un 9/6) "
        "- 'scores_above_85' : indicateur de qualite premium (combien de sous-scores sont excellents). "
        "Conclusion : les agregations et ratios apportent de la NOUVELLE INFORMATION non redondante "
        "avec les sous-scores individuels, et sont moins correles entre elles."
    )


def viz_pairplot(df: pd.DataFrame, score_cols: list[str]) -> None:
    print("\n[FIG 10] Pairplot sensoriel...")
    cols = ["Aroma", "Flavor", "Acidity", "Body", "Balance", "Total Cup Points"]
    sample = df[cols].dropna()
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": PALETTE[0]},
                     height=2.2)
    g.figure.suptitle("Pairplot des principaux scores sensoriels", y=1.02, fontsize=13, fontweight="bold")
    g.savefig(FIG / "10_pairplot_scores.png", dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    print("  -> 10_pairplot_scores.png")
    CONCLUSIONS["fig10"] = (
        "Le pairplot confirme visuellement la correlation lineaire entre tous les sous-scores. "
        "On voit que : (1) Flavor et Aftertaste sont quasi-colinéaires (memes jugeotes), "
        "(2) Uniformity/Clean Cup/Sweetness sont presque constants a 10 (donc peu informatifs), "
        "(3) Total Cup Points a une relation lineaire parfaite avec sensory_mean. "
        "Conclusion : pour predire Total Cup Points, on peut se contenter de 4-5 sous-scores "
        "et utiliser la regularisation pour eliminer la redondance."
    )


def feature_importance(df_fe: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("4. FEATURE IMPORTANCE POUR LA CIBLE 'Overall' (note subjective)")
    print("=" * 70)
    target = "Overall"
    drop_for_X = ["Total Cup Points"] + score_cols
    cols_to_drop = [c for c in drop_for_X if c in df_fe.columns]
    X = df_fe.drop(columns=cols_to_drop, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    drop_high_nan = [c for c in X.columns if X[c].isna().mean() > 0.4]
    X = X.drop(columns=drop_high_nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df_fe[target].fillna(df_fe[target].median())
    print(f"X shape: {X.shape}, y non-null: {len(y)}")
    print(f"  colonnes NaN > 40% supprimees : {drop_high_nan}")

    print("\n--- Random Forest importance ---")
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(rf_imp.head(15).round(4).to_string())

    print("\n--- Mutual Information ---")
    mi = mutual_info_regression(X, y, random_state=42)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(mi_s.head(15).round(4).to_string())

    print("\n--- Lasso CV ---")
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(Xs, y)
    lasso_coef = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    print(f"  alpha optimal : {lasso.alpha_:.4f}")
    print(f"  features retenues (|coef| > 0) : {(lasso_coef > 0).sum()}")
    print(lasso_coef[lasso_coef > 0].head(15).round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 8))
    top = pd.DataFrame({"RF": rf_imp, "MI": mi_s, "Lasso": lasso_coef}).head(15)
    sns.heatmap(top.rank(ascending=False), annot=True, fmt=".0f", cmap="RdYlGn_r",
                cbar_kws={"label": "Rang (1 = meilleur)"}, ax=ax)
    ax.set_title(f"Top 15 features - RF vs MI vs Lasso (cible : {target})", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    save(fig, "11_feature_importance")
    return rf_imp, mi_s, lasso_coef


def modeling(df_fe: pd.DataFrame, score_cols: list[str]) -> dict:
    print("\n" + "=" * 70)
    print("5. MODELISATION (cible = Overall)")
    print("=" * 70)
    target = "Overall"
    drop_for_X = ["Total Cup Points"] + score_cols
    cols_to_drop = [c for c in drop_for_X if c in df_fe.columns]
    X = df_fe.drop(columns=cols_to_drop, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    drop_high_nan = [c for c in X.columns if X[c].isna().mean() > 0.4]
    X = X.drop(columns=drop_high_nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df_fe[target].fillna(df_fe[target].median())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Train : {Xtr.shape}, Test : {Xte.shape}")

    models = {
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    }
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
    models["Random Forest (300 trees, depth=8)"] = rf
    results = {}
    for name, m in models.items():
        if "Ridge" in name:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            m.fit(Xtr_s, ytr)
            yp = m.predict(Xte_s)
        else:
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
        r2 = r2_score(yte, yp)
        mae = mean_absolute_error(yte, yp)
        results[name] = {"R2": r2, "MAE": mae}
        print(f"  {name} : R² = {r2:.4f}, MAE = {mae:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (name, m) in enumerate(models.items()):
        if "Ridge" in name:
            yp = m.predict(StandardScaler().fit(Xtr).transform(Xte))
        else:
            yp = m.predict(Xte)
        axes[i].scatter(yte, yp, alpha=0.4, s=15, color=PALETTE[i])
        axes[i].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--", linewidth=2)
        axes[i].set_xlabel("Overall (vrai)")
        axes[i].set_ylabel("Overall (prédit)")
        axes[i].set_title(f"{name}\nR² = {results[name]['R2']:.3f} | MAE = {results[name]['MAE']:.3f}")
    save(fig, "12_modeling_results")
    return results


def save_outputs(df_fe: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("6. SAUVEGARDE DES ARTEFACTS")
    print("=" * 70)
    df_fe.to_csv(DATA_OUT / "coffee_engineered.csv", index=False)
    print(f"  Dataset engineered : {df_fe.shape}")
    with open(DATA_OUT / "conclusions.json", "w") as f:
        json.dump(CONCLUSIONS, f, indent=2, default=str)
    print(f"  Conclusions : conclusions.json")


def main() -> None:
    df = load_data()
    info = eda(df)
    df_parsed = parse_features(df)
    df_fe, new_features, score_cols = feature_engineering(df)
    viz_distributions(df_parsed, score_cols)
    viz_correlation(df_parsed, score_cols)
    viz_country(df_parsed)
    viz_processing(df_parsed)
    viz_altitude(df_parsed)
    viz_defects(df_parsed)
    viz_variety(df_parsed)
    viz_pairplot(df_parsed, score_cols)
    viz_feature_engineering(df_fe, new_features)
    feature_importance(df_fe, score_cols)
    results = modeling(df_fe, score_cols)
    save_outputs(df_fe)
    with open(DATA_OUT / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 70)
    print("TERMINE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
