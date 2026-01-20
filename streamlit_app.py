import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


DATA_PATH = "data/cleaned_dataset.csv"
RAW_DATA_PATH = "data/joined_dataset.csv"

WDI_COLS = [
    "NY.GDP.MKTP.CD",
    "NY.GDP.PCAP.CD",
    "NY.GDP.MKTP.KD.ZG",
    "EN.POP.DNST",
    "SP.POP.TOTL",
    "SP.DYN.LE00.IN",
    "SH.XPD.CHEX.PC.CD",
    "SH.MED.BEDS.ZS",
    "IT.NET.USER.ZS",
    "IS.AIR.PSGR",
]

FRIENDLY = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "EN.POP.DNST": "Population density",
    "SP.POP.TOTL": "Population",
    "SP.DYN.LE00.IN": "Life expectancy (years)",
    "SH.XPD.CHEX.PC.CD": "Health exp. per capita (US$)",
    "SH.MED.BEDS.ZS": "Hospital beds (per 1k)",
    "IT.NET.USER.ZS": "Internet users (%)",
    "IS.AIR.PSGR": "Air passengers",
}

ALT_GDP_PCAP = r"NY\.GDP\.PCAP\.CD"
ALT_IT_NET = r"IT\.NET\.USER\.ZS"


def _iso_like_country_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.match(r"^[A-Z]{2,3}$", na=False)


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, a_min=0, a_max=None))


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
                        ("log1p", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
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


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["country_code"])
    df = df[_iso_like_country_code(df["country_code"])]
    return df


@st.cache_data(show_spinner=False)
def load_raw_joined() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna(subset=["country_code"])
    df = df[_iso_like_country_code(df["country_code"])]
    return df


@st.cache_data(show_spinner=False)
def build_dashboard_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data()

    df_2019 = df.loc[df["year"] == 2019].copy()
    df_2019 = df_2019.drop_duplicates(subset=["country_code"])
    df_2019 = df_2019.dropna(subset=["shock_2020"])

    X = df_2019[WDI_COLS].copy()
    y = df_2019["shock_2020"].copy()

    preprocess = build_preprocess(WDI_COLS)
    X_proc = preprocess.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_proc)

    best_k, silhouette_scores = choose_k_by_silhouette(emb, k_values=[2, 3, 4, 5])
    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(emb)

    df_model = df_2019[["country_code", "country_name_tourism", "shock_2020", "value", *WDI_COLS]].copy()
    df_model["shock_pct"] = df_model["shock_2020"] * 100.0
    df_model = df_model.rename(columns={"value": "arrivals_2019"})
    df_model["arrivals_per_1000"] = np.where(
        df_model["SP.POP.TOTL"].gt(0),
        (df_model["arrivals_2019"] / df_model["SP.POP.TOTL"]) * 1000.0,
        np.nan,
    )
    df_model["PC1"] = emb[:, 0]
    df_model["PC2"] = emb[:, 1]
    df_model["cluster"] = clusters.astype(int)
    df_model["cluster_display"] = df_model["cluster"] + 1
    df_model["cluster_label"] = df_model["cluster_display"].map(lambda c: f"Cluster {c}")
    df_model["pca_var_pct"] = (pca.explained_variance_ratio_.sum() * 100.0)
    df_model["silhouette_best_k"] = silhouette_scores[best_k]

    feature_out = preprocess.get_feature_names_out()
    loadings = pd.DataFrame(
        {
            "feature": feature_out,
            "PC1_loading": pca.components_[0, :],
            "PC2_loading": pca.components_[1, :],
        }
    )
    loadings["feature_label"] = loadings["feature"].map(lambda c: FRIENDLY.get(c, c))
    loadings["abs_PC1"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2"] = loadings["PC2_loading"].abs()

    df_all = df.merge(
        df_model[["country_code", "cluster", "cluster_display", "cluster_label", "PC1", "PC2", "shock_pct"]],
        on="country_code",
        how="left",
    )
    df_all = df_all[df_all["cluster"].notna()].copy()
    df_all["cluster"] = df_all["cluster"].astype(int)
    df_all["arrivals_m"] = df_all["value"] / 1_000_000.0

    baseline_2019 = (
        df_all.loc[df_all["year"] == 2019, ["country_code", "value"]]
        .rename(columns={"value": "value_2019"})
        .dropna(subset=["value_2019"])
        .drop_duplicates(subset=["country_code"])
    )
    df_all = df_all.merge(baseline_2019, on="country_code", how="left")
    df_all["arrivals_index"] = np.where(
        df_all["value_2019"].gt(0),
        (df_all["value"] / df_all["value_2019"]) * 100.0,
        np.nan,
    )

    df_raw = load_raw_joined()
    coverage_cols = ["value", *WDI_COLS]
    coverage_cols = [c for c in coverage_cols if c in df_raw.columns]
    cov = (
        df_raw.groupby("year")[coverage_cols]
        .apply(lambda frame: frame.notna().mean() * 100.0)
        .reset_index()
        .melt(id_vars=["year"], var_name="variable", value_name="coverage_pct")
    )
    cov["variable_label"] = cov["variable"].map(
        lambda c: "Tourism arrivals (ST.INT.ARVL)" if c == "value" else FRIENDLY.get(c, c)
    )

    return df_model, df_all, loadings, cov


def make_dashboard_chart(
    df_model: pd.DataFrame, df_all: pd.DataFrame, loadings: pd.DataFrame, coverage: pd.DataFrame
) -> alt.Chart:
    alt.data_transformers.disable_max_rows()

    cluster_sel = alt.selection_point(name="cluster_sel", fields=["cluster_label"], on="click", empty="all")
    country_sel = alt.selection_point(name="country_sel", fields=["country_code"], on="click", empty="none")
    pca_brush = alt.selection_interval(name="pca_brush", encodings=["x", "y"], empty="all")

    base_model = alt.Chart(df_model).transform_filter(cluster_sel)

    pca_scatter = (
        base_model.mark_circle(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("PC1:Q", title="PC1 (Principal Component 1 — structural index)"),
            y=alt.Y("PC2:Q", title="PC2 (Principal Component 2 — structural index)"),
            color=alt.Color("cluster_label:N", title="Cluster"),
            size=alt.Size("shock_pct:Q", title="Shock magnitude (%)", legend=None, scale=alt.Scale(range=[30, 250])),
            opacity=alt.condition(pca_brush, alt.value(0.9), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("country_name_tourism:N", title="Country"),
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("shock_pct:Q", title="2020 shock (%)", format=".1f"),
                alt.Tooltip(f"{ALT_GDP_PCAP}:Q", title="GDP per capita (US$)", format=",.0f"),
                alt.Tooltip(f"{ALT_IT_NET}:Q", title="Internet users (%)", format=".1f"),
            ],
        )
        .add_params(country_sel, pca_brush)
        .properties(
            title="1) PCA map (2019 indicators) — click a country",
            width=520,
            height=330,
        )
    )

    shock_by_cluster = (
        alt.Chart(df_model)
        .transform_aggregate(
            mean_shock="mean(shock_pct)",
            count="count()",
            groupby=["cluster_display", "cluster_label"],
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "cluster_label:N",
                title=None,
                sort=alt.SortField("cluster_display", order="ascending"),
            ),
            y=alt.Y("mean_shock:Q", title="Mean tourism shock 2020 (%)"),
            color=alt.condition(cluster_sel, alt.value("#2c7fb8"), alt.value("#d9d9d9")),
            tooltip=[
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("mean_shock:Q", title="Mean shock (%)", format=".1f"),
                alt.Tooltip("count:Q", title="Countries", format=".0f"),
            ],
        )
        .add_params(cluster_sel)
        .properties(
            title="2) Mean shock by cluster — click to filter",
            width=360,
            height=160,
        )
    )

    loadings_chart = (
        alt.Chart(loadings)
        .transform_window(rank="rank(abs_PC1)", sort=[alt.SortField("abs_PC1", order="descending")])
        .transform_filter(alt.datum.rank <= 5)
        .mark_bar(color="#74add1")
        .encode(
            x=alt.X("PC1_loading:Q", title="PC1 loading (signed)"),
            y=alt.Y("feature_label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
            tooltip=[alt.Tooltip("feature_label:N", title="Indicator"), alt.Tooltip("PC1_loading:Q", format=".3f")],
        )
        .properties(title="PC1 drivers (top 5)", width=360, height=120)
    )

    loadings_chart_2 = (
        alt.Chart(loadings)
        .transform_window(rank="rank(abs_PC2)", sort=[alt.SortField("abs_PC2", order="descending")])
        .transform_filter(alt.datum.rank <= 5)
        .mark_bar(color="#f46d43")
        .encode(
            x=alt.X("PC2_loading:Q", title="PC2 loading (signed)"),
            y=alt.Y("feature_label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
            tooltip=[alt.Tooltip("feature_label:N", title="Indicator"), alt.Tooltip("PC2_loading:Q", format=".3f")],
        )
        .properties(title="PC2 drivers (top 5)", width=360, height=120)
    )

    trend_avg = (
        alt.Chart(df_all)
        .transform_filter(cluster_sel)
        .transform_aggregate(avg_index="mean(arrivals_index)", groupby=["year"])
        .mark_line(color="#7a7a7a", strokeDash=[6, 4])
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("avg_index:Q", title="Arrivals index (2019 = 100)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("avg_index:Q", title="Average index", format=".1f")],
        )
    )

    trend_selected = (
        alt.Chart(df_all)
        .transform_filter(cluster_sel)
        .transform_filter(country_sel)
        .mark_line(color="#1f78b4", strokeWidth=3)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("arrivals_index:Q", title="Arrivals index (2019 = 100)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("country_name_tourism:N", title="Country"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("arrivals_index:Q", title="Index", format=".1f"),
                alt.Tooltip("value:Q", title="Arrivals", format=",.0f"),
            ],
        )
    )

    arrivals_trend = (
        alt.layer(trend_avg, trend_selected)
        .properties(
            title="3) Arrivals over time (selected country vs cluster average)",
            width=360,
            height=160,
        )
    )

    gdp_scatter = (
        base_model.mark_circle(stroke="white", strokeWidth=0.5)
        .transform_filter(alt.datum["NY.GDP.PCAP.CD"] > 0)
        .transform_filter(pca_brush)
        .transform_filter(alt.datum.shock_pct <= 0)
        .encode(
            x=alt.X(
                f"{ALT_GDP_PCAP}:Q",
                title="GDP per capita (2019, US$, log)",
                scale=alt.Scale(type="log", clamp=True),
            ),
            y=alt.Y(
                "shock_pct:Q",
                title="Tourism shock 2020 (%)",
                scale=alt.Scale(domain=[-100, 0]),
            ),
            color=alt.Color("cluster_label:N", title="Cluster"),
            opacity=alt.condition(country_sel, alt.value(1.0), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("country_name_tourism:N", title="Country"),
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("shock_pct:Q", title="2020 shock (%)", format=".1f"),
                alt.Tooltip(f"{ALT_GDP_PCAP}:Q", title="GDP per capita (US$)", format=",.0f"),
            ],
        )
        .properties(
            title="4) GDP per capita vs shock (2019 → 2020)",
            width=450,
            height=220,
        )
    )

    dependence_scatter = (
        base_model.mark_circle(stroke="white", strokeWidth=0.5)
        .transform_filter(alt.datum.arrivals_per_1000 > 0)
        .transform_filter(pca_brush)
        .transform_filter(alt.datum.shock_pct <= 0)
        .encode(
            x=alt.X(
                "arrivals_per_1000:Q",
                title="Tourism dependence (2019 arrivals per 1,000 residents, log)",
                scale=alt.Scale(type="log", clamp=True),
            ),
            y=alt.Y(
                "shock_pct:Q",
                title="Tourism shock 2020 (%)",
                scale=alt.Scale(domain=[-100, 0]),
            ),
            color=alt.Color("cluster_label:N", title="Cluster"),
            opacity=alt.condition(country_sel, alt.value(1.0), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("country_name_tourism:N", title="Country"),
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("arrivals_per_1000:Q", title="Arrivals/1k", format=".2f"),
                alt.Tooltip("shock_pct:Q", title="2020 shock (%)", format=".1f"),
            ],
        )
        .properties(
            title="5) Tourism dependence vs shock",
            width=450,
            height=220,
        )
    )

    shock_hist = (
        alt.Chart(df_model)
        .transform_filter(cluster_sel)
        .transform_filter(pca_brush)
        .transform_filter(alt.datum.shock_pct <= 0)
        .mark_bar(color="#fb8072")
        .encode(
            x=alt.X(
                "shock_pct:Q",
                bin=alt.Bin(maxbins=28),
                title="Tourism shock 2020 vs 2019 (%)",
                scale=alt.Scale(domain=[-100, 0]),
            ),
            y=alt.Y("count():Q", title="Countries"),
            tooltip=[alt.Tooltip("count():Q", title="Countries", format=".0f")],
        )
        .properties(title="Insight: Global collapse (distribution of shock)", width=520, height=170)
    )

    coverage_focus = coverage[coverage["variable"].isin(["value", "NY.GDP.PCAP.CD", "IT.NET.USER.ZS", "SH.XPD.CHEX.PC.CD"])].copy()
    coverage_line = (
        alt.Chart(coverage_focus)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("coverage_pct:Q", title="Data coverage (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("variable_label:N", title=None),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("variable_label:N", title="Series"), alt.Tooltip("coverage_pct:Q", title="Coverage (%)", format=".1f")],
        )
        .properties(title="Data quality: coverage by year (raw joined data)", width=520, height=170)
    )

    header = alt.Chart(
        pd.DataFrame(
            [
                {
                    "text": (
                        "How to use: (a) Click a bar to filter clusters, (b) click a country in the PCA map to show its 2018–2020 trend, "
                        "(c) drag a rectangle on the PCA map to brush a subset and see linked views update. "
                        "This dashboard uses the PCA + KMeans model from Task 3 and links all views via shared selections."
                    )
                }
            ]
        )
    ).mark_text(align="left", baseline="top").encode(text="text:N").properties(width=900, height=28)

    return alt.vconcat(
        header,
        alt.hconcat(
            pca_scatter,
            alt.vconcat(shock_by_cluster, arrivals_trend),
            spacing=16,
        ),
        alt.hconcat(alt.vconcat(loadings_chart, loadings_chart_2), shock_hist),
        coverage_line,
        alt.hconcat(gdp_scatter, dependence_scatter),
        spacing=14,
    ).resolve_scale(color="shared")


def main() -> None:
    st.set_page_config(page_title="Task 3 Dashboard — Tourism Shock", layout="wide")

    st.title("Task 3 Report Dashboard: Structural Country Types & the 2020 Tourism Shock")
    st.caption(
        "Data: World Bank tourism arrivals (ST.INT.ARVL) + 10 WDI indicators. "
        "Model: PCA (2D) + KMeans clustering on 2019 indicators; outcome shown as 2020 shock vs 2019."
    )

    df_model, df_all, loadings, coverage = build_dashboard_tables()

    best_k = int(df_model["cluster"].nunique())
    var_pct = float(df_model["pca_var_pct"].iloc[0])
    sil = float(df_model["silhouette_best_k"].iloc[0])
    mean_shock = float(df_model["shock_pct"].mean())
    median_shock = float(df_model["shock_pct"].median())
    st.markdown(
        f"**Model snapshot:** k = **{best_k}** clusters (silhouette ≈ **{sil:.2f}**), PCA(2D) captures ≈ **{var_pct:.1f}%** of indicator variance.  "
        f"Across countries, the **mean 2020 shock is {mean_shock:.1f}%** (median {median_shock:.1f}%)."
    )
    st.caption("Interpretation note: this is a descriptive, unsupervised model (country ‘types’) — it shows patterns/associations, not causality.")

    with st.expander("Glossary (what the short forms mean)", expanded=False):
        st.markdown(
            "- **PCA** = Principal Component Analysis: a way to compress many correlated indicators into a few summary dimensions.\n"
            "- **PC1 / PC2** = Principal Component 1 / 2: the first two PCA dimensions. Each is a weighted combination of the 2019 indicators; countries close together on the PCA map have similar 2019 profiles.\n"
            "- **KMeans** = a clustering method that groups countries into k “types” based on similarity in the PCA space.\n"
            "- **Silhouette score** = a clustering quality score (higher is better separation).\n"
            "- **WDI** = World Development Indicators (World Bank).\n"
            "- **GDP** = Gross Domestic Product; **US$** = US dollars.\n"
            "- **shock_2020** = percentage change in tourism arrivals from 2019 to 2020."
        )

    st.markdown(
        "- **Main idea:** Countries with similar 2019 “structure” (economy, connectivity, health, population) are grouped into clusters, then we compare how hard each cluster was hit in 2020.\n"
        "- **Interaction:** Click a **cluster bar** to filter, then click a **country** in the PCA map to see its time trend.\n"
        "- **Extra context from Task 2:** The histogram shows the near-universal collapse in 2020, and the coverage chart explains why analysis focuses on 2018–2020 (tourism data is mostly missing after 2020).\n"
        "- **Readability note:** The shock distribution + the two scatterplots focus on **negative shocks (declines)** to avoid a small number of “increase” outliers dominating the scale."
    )

    chart = make_dashboard_chart(df_model, df_all, loadings, coverage)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Indicators used (2019)"):
        st.write(pd.DataFrame([{"Code": c, "Indicator": FRIENDLY.get(c, c)} for c in WDI_COLS]))


if __name__ == "__main__":
    main()
