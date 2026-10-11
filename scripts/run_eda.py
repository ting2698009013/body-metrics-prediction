"""Generate reproducible exploratory analysis artifacts for NHANES v0.1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nhanes_2011_2018_body_composition.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "eda"
FIGURE_DIR = OUTPUT_DIR / "figures"

FEATURES = [
    "age_years",
    "height_cm",
    "weight_kg",
    "bmi",
    "upper_arm_length_cm",
    "arm_circumference_cm",
    "waist_circumference_cm",
]

TARGETS = [
    "body_fat_percent",
    "total_fat_mass_kg",
    "total_lean_mass_kg",
]

SEX_LABELS = {1.0: "Male", 2.0: "Female"}


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURES + TARGETS:
        values = data[column]
        rows.append(
            {
                "variable": column,
                "non_missing": int(values.notna().sum()),
                "missing_percent": float(values.isna().mean() * 100),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "minimum": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def missingness_analysis(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, group in data.groupby("cycle", sort=True):
        for column in FEATURES + TARGETS + ["hip_circumference_cm"]:
            rows.append(
                {
                    "cycle": cycle,
                    "variable": column,
                    "rows": int(len(group)),
                    "missing_percent": float(group[column].isna().mean() * 100),
                }
            )
    return pd.DataFrame(rows)


def plot_missingness(missingness: pd.DataFrame) -> None:
    pivot = missingness.pivot(index="variable", columns="cycle", values="missing_percent")
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Missingness by NHANES cycle (%)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            color = "white" if value > 55 else "black"
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="Missing (%)")
    save_figure(fig, "01_missingness_by_cycle.png")


def plot_target_distributions(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {1.0: "#2563eb", 2.0: "#db2777"}
    for ax, target in zip(axes, TARGETS, strict=True):
        for sex_code, label in SEX_LABELS.items():
            values = data.loc[data["sex_code"].eq(sex_code), target].dropna()
            ax.hist(
                values,
                bins=35,
                density=True,
                alpha=0.42,
                color=colors[sex_code],
                label=label,
            )
        ax.set_title(target)
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("DXA target distributions by sex")
    save_figure(fig, "02_target_distributions_by_sex.png")


def correlation_tables(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All", data)] + [
        (label, data.loc[data["sex_code"].eq(code)])
        for code, label in SEX_LABELS.items()
    ]
    for group_name, group in groups:
        correlation = group[FEATURES + TARGETS].corr(method="spearman")
        for feature in FEATURES:
            for target in TARGETS:
                rows.append(
                    {
                        "group": group_name,
                        "feature": feature,
                        "target": target,
                        "spearman_correlation": float(correlation.loc[feature, target]),
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_target_correlations(correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
    for ax, group_name in zip(axes, ["All", "Male", "Female"], strict=True):
        subset = correlations.loc[correlations["group"].eq(group_name)]
        matrix = subset.pivot(index="feature", columns="target", values="spearman_correlation")
        matrix = matrix.reindex(index=FEATURES, columns=TARGETS)
        image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(TARGETS)), TARGETS, rotation=35, ha="right")
        ax.set_yticks(range(len(FEATURES)), FEATURES)
        ax.set_title(group_name)
        for row in range(len(FEATURES)):
            for column in range(len(TARGETS)):
                value = matrix.iloc[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Spearman correlation", shrink=0.75)
    fig.suptitle("Feature-target rank correlations")
    save_figure(fig, "03_feature_target_correlations.png")


def binned_medians(data: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    rows = []
    for sex_code, label in SEX_LABELS.items():
        subset = data.loc[data["sex_code"].eq(sex_code), [feature, target]].dropna().copy()
        subset["bin"] = pd.qcut(subset[feature], q=12, duplicates="drop")
        grouped = subset.groupby("bin", observed=True).agg(
            feature_median=(feature, "median"),
            target_median=(target, "median"),
            rows=(target, "size"),
        )
        grouped["sex"] = label
        rows.append(grouped.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def plot_binned_relationships(data: pd.DataFrame) -> None:
    relationships = [
        ("bmi", "body_fat_percent"),
        ("waist_circumference_cm", "body_fat_percent"),
        ("weight_kg", "total_fat_mass_kg"),
        ("arm_circumference_cm", "total_lean_mass_kg"),
    ]
    colors = {"Male": "#2563eb", "Female": "#db2777"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (feature, target) in zip(axes.ravel(), relationships, strict=True):
        binned = binned_medians(data, feature, target)
        for label in ["Male", "Female"]:
            subset = binned.loc[binned["sex"].eq(label)]
            ax.plot(
                subset["feature_median"],
                subset["target_median"],
                marker="o",
                linewidth=2,
                color=colors[label],
                label=label,
            )
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Median target across feature quantiles")
    save_figure(fig, "04_binned_relationships.png")


def plot_cycle_drift(data: pd.DataFrame) -> None:
    variables = [
        "bmi",
        "waist_circumference_cm",
        "body_fat_percent",
        "total_fat_mass_kg",
        "total_lean_mass_kg",
    ]
    medians = data.groupby("cycle")[variables].median()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, variable in zip(axes.ravel(), variables, strict=False):
        ax.plot(medians.index, medians[variable], marker="o", linewidth=2)
        ax.set_title(variable)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Median values by NHANES cycle")
    save_figure(fig, "05_cycle_medians.png")


def plot_feature_redundancy(data: pd.DataFrame) -> None:
    matrix = data[FEATURES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for row in range(len(FEATURES)):
        for column in range(len(FEATURES)):
            ax.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Input feature redundancy")
    save_figure(fig, "06_feature_redundancy.png")


def target_missingness_bias(data: pd.DataFrame) -> pd.DataFrame:
    working = data.copy()
    working["body_fat_label"] = np.where(
        working["body_fat_percent"].isna(), "DXA target missing", "DXA target present"
    )
    return (
        working.groupby("body_fat_label")
        .agg(
            rows=("participant_id", "size"),
            median_bmi=("bmi", "median"),
            median_weight_kg=("weight_kg", "median"),
            median_waist_cm=("waist_circumference_cm", "median"),
            median_age=("age_years", "median"),
        )
        .reset_index()
    )


def plot_target_missingness_bias(bias: pd.DataFrame) -> None:
    variables = [
        ("median_bmi", "Median BMI"),
        ("median_weight_kg", "Median weight (kg)"),
        ("median_waist_cm", "Median waist (cm)"),
        ("median_age", "Median age")
    ]
    colors = ["#10b981", "#f97316"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for ax, (column, title) in zip(axes, variables, strict=True):
        ax.bar(bias["body_fat_label"], bias[column], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        for index, value in enumerate(bias[column]):
            ax.text(index, value, f"{value:.1f}", ha="center", va="bottom")
    fig.suptitle("Participants with missing DXA targets differ systematically")
    save_figure(fig, "07_target_missingness_bias.png")


def make_findings(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> dict[str, object]:
    overall = correlations.loc[correlations["group"].eq("All")].copy()
    top_by_target = {}
    for target in TARGETS:
        ranked = overall.loc[overall["target"].eq(target)].assign(
            absolute=lambda frame: frame["spearman_correlation"].abs()
        )
        ranked = ranked.sort_values("absolute", ascending=False).head(4)
        top_by_target[target] = ranked[
            ["feature", "spearman_correlation"]
        ].to_dict(orient="records")

    sex_target_medians = (
        data.groupby("sex_code")[TARGETS].median().rename(index=SEX_LABELS).to_dict()
    )
    cycle_target_medians = data.groupby("cycle")[TARGETS].median().to_dict()

    return {
        "rows": int(len(data)),
        "cycles": data["cycle"].value_counts().sort_index().astype(int).to_dict(),
        "top_feature_target_correlations": top_by_target,
        "sex_target_medians": sex_target_medians,
        "cycle_target_medians": cycle_target_medians,
        "summary_records": summary.to_dict(orient="records"),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = summary_statistics(data)
    missingness = missingness_analysis(data)
    correlations = correlation_tables(data)
    missingness_bias = target_missingness_bias(data)

    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    missingness.to_csv(OUTPUT_DIR / "missingness_by_cycle.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "feature_target_correlations.csv", index=False)
    missingness_bias.to_csv(OUTPUT_DIR / "target_missingness_bias.csv", index=False)

    plot_missingness(missingness)
    plot_target_distributions(data)
    plot_feature_target_correlations(correlations)
    plot_binned_relationships(data)
    plot_cycle_drift(data)
    plot_feature_redundancy(data)
    plot_target_missingness_bias(missingness_bias)

    findings = make_findings(data, summary, correlations)
    (OUTPUT_DIR / "findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"EDA artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

