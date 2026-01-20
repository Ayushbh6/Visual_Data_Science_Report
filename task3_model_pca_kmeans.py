import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from task3_model import DATA_PATH, WDI_COLS, _iso_like_country_code, load_joined_data


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


OUT_DIR = os.path.join("Docs", "images")
OUT_FIG = os.path.join(OUT_DIR, "task3_model_pca_kmeans.png")
OUT_SUMMARY = os.path.join("Docs", "task3_model_pca_kmeans_summary.csv")


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def build_modeling_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.dropna(subset=["country_code"])
    df = df[_iso_like_country_code(df["country_code"])]

    tourism = (
        df.loc[df["year"].isin([2019, 2020]), ["country_code", "year", "value"]]
        .dropna(subset=["value"])
        .drop_duplicates(subset=["country_code", "year"])
    )
    pivot = tourism.pivot(index="country_code", columns="year", values="value")
    valid = pivot[2019].notna() & pivot[2020].notna() & (pivot[2019] > 0)
    shock = (pivot.loc[valid, 2020] - pivot.loc[valid, 2019]) / pivot.loc[valid, 2019]
    shock.name = "shock_2020"

    predictors_2019 = (
        df.loc[df["year"] == 2019, ["country_code", *WDI_COLS]]
        .drop_duplicates(subset=["country_code"])
        .set_index("country_code")
    )

    modeling = predictors_2019.join(shock, how="inner")
    X = modeling[WDI_COLS].copy()
    y = modeling["shock_2020"].copy()
    return X, y


def build_preprocess(feature_names: list[str]) -> ColumnTransformer:
    log_cols = [
        c
        for c in feature_names
        if c
        in {
            "NY.GDP.MKTP.CD",
            "NY.GDP.PCAP.CD",
            "SH.XPD.CHEX.PC.CD",
            "IS.AIR.PSGR",
            "SP.POP.TOTL",
            "EN.POP.DNST",
            "SH.MED.BEDS.ZS",
        }
    ]
    pass_cols = [c for c in feature_names if c not in set(log_cols)]

    return ColumnTransformer(
        transformers=[
            (
                "log",
                Pipeline(
                    steps=[
                        (
                            "log1p",
                            FunctionTransformer(_log1p_clip, feature_names_out="one-to-one"),
                        ),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                log_cols,
            ),
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                pass_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def choose_k_by_silhouette(emb: np.ndarray, k_values: list[int]) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2:
            scores[k] = float("nan")
            continue
        scores[k] = float(silhouette_score(emb, labels))
    best_k = max(scores, key=lambda kk: scores[kk])
    return best_k, scores


def main() -> None:
    df = load_joined_data(DATA_PATH)
    X, y = build_modeling_table(df)
    if len(X) < 30:
        raise ValueError(f"Too few countries with valid shock data after filtering: N={len(X)}")

    preprocess = build_preprocess(list(X.columns))
    X_proc = preprocess.fit_transform(X)
    feature_out = preprocess.get_feature_names_out()

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_proc)

    best_k, silhouette_scores = choose_k_by_silhouette(emb, k_values=[2, 3, 4, 5])
    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(emb)

    plot_df = pd.DataFrame(
        {
            "PC1": emb[:, 0],
            "PC2": emb[:, 1],
            "cluster": clusters,
            "shock_pct": y.values * 100,
        },
        index=X.index,
    )
    cluster_summary = (
        plot_df.groupby("cluster")["shock_pct"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_out,
        columns=["PC1_loading", "PC2_loading"],
    )
    loadings["abs_PC1"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2"] = loadings["PC2_loading"].abs()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # Panel 1: PCA embedding with clusters
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    sizes = np.interp(plot_df["shock_pct"].abs(), (plot_df["shock_pct"].abs().min(), plot_df["shock_pct"].abs().max()), (20, 180))
    for c in sorted(plot_df["cluster"].unique()):
        mask = plot_df["cluster"] == c
        ax.scatter(
            plot_df.loc[mask, "PC1"],
            plot_df.loc[mask, "PC2"],
            s=sizes[mask],
            alpha=0.85,
            color=cmap(int(c) % 10),
            edgecolors="white",
            linewidths=0.5,
            label=f"Cluster {c}",
        )
    ax.set_title("PCA(2019 indicators) + KMeans clusters")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # Panel 2: Mean shock per cluster
    ax = axes[1]
    ax.bar(cluster_summary["cluster"].astype(str), cluster_summary["mean"], color="#2c7fb8")
    ax.set_title("Tourism shock by structural cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean shock_2020 (%)")
    for _, row in cluster_summary.iterrows():
        ax.text(
            str(int(row["cluster"])),
            row["mean"] + 0.8,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 3: Top PCA loadings (what defines PC1/PC2)
    ax = axes[2]
    top_pc1 = loadings.sort_values("abs_PC1", ascending=False).head(6)[["PC1_loading"]]
    top_pc2 = loadings.sort_values("abs_PC2", ascending=False).head(6)[["PC2_loading"]]
    top = (
        pd.concat(
            [
                top_pc1.rename(columns={"PC1_loading": "loading"}).assign(component="PC1"),
                top_pc2.rename(columns={"PC2_loading": "loading"}).assign(component="PC2"),
            ]
        )
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    # Two-tone horizontal bars: PC1 vs PC2
    top_pc1_plot = top[top["component"] == "PC1"].copy()
    top_pc2_plot = top[top["component"] == "PC2"].copy()
    y_positions = np.arange(len(top_pc1_plot) + len(top_pc2_plot))
    top_plot = pd.concat([top_pc1_plot, top_pc2_plot], ignore_index=True)
    colors = np.where(top_plot["component"] == "PC1", "#74add1", "#f46d43")
    ax.barh(top_plot["feature"], top_plot["loading"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top PCA loadings (interpretation)")
    ax.set_xlabel("Loading (signed)")
    ax.set_ylabel("")
    ax.legend(handles=[], labels=[])

    fig.suptitle(
        f"Task 3 Model (Unsupervised): Structural Country Types and 2020 Tourism Shock (k={best_k}, silhouette={silhouette_scores[best_k]:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)

    # Save summary for write-up
    summary_rows = []
    for k, score in silhouette_scores.items():
        summary_rows.append({"k": k, "silhouette": score})
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    print("Saved:", OUT_FIG)
    print("Saved:", OUT_SUMMARY)
    print("Cluster summary:")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
