import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


DATA_PATH = os.path.join("data", "joined_dataset.csv")
OUT_DIR = os.path.join("Docs", "images")
OUT_FIG = os.path.join(OUT_DIR, "task3_model_elasticnet_results.png")
OUT_METRICS = os.path.join("Docs", "task3_model_metrics.csv")


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


@dataclass(frozen=True)
class ModelResults:
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    r2: float
    best_params: dict


@dataclass(frozen=True)
class FinalModelArtifacts:
    model_name: str
    best_params: dict
    mae: float
    r2: float
    permutation_importance_df: pd.DataFrame


def _iso_like_country_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.match(r"^[A-Z]{2,3}$", na=False)


def load_joined_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "country_code",
        "country_name_tourism",
        "year",
        "value",
        "indicator",
        "country_name_wdi",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {sorted(missing)}")
    return df


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
    if 2019 not in pivot.columns or 2020 not in pivot.columns:
        raise ValueError("Cannot compute shock_2020: missing 2019 and/or 2020 arrivals.")
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


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def _split_feature_groups(feature_names: list[str]) -> tuple[list[str], list[str]]:
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
    return log_cols, pass_cols


def build_elasticnet_pipeline(feature_names: list[str]) -> Pipeline:
    log_cols, pass_cols = _split_feature_groups(feature_names)
    preprocess = ColumnTransformer(
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

    model = ElasticNet(max_iter=50000, random_state=42)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def build_hgbr_pipeline(feature_names: list[str]) -> Pipeline:
    log_cols, pass_cols = _split_feature_groups(feature_names)
    preprocess = ColumnTransformer(
        transformers=[
            (
                "log",
                Pipeline(
                    steps=[
                        ("log1p", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                log_cols,
            ),
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                pass_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(random_state=42)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def nested_cv_predictions(
    *,
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> ModelResults:
    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    best_params_all: list[dict] = []

    for train_idx, test_idx in outer.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner = KFold(n_splits=3, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        best_params_all.append(search.best_params_)

    y_true = np.array(y_true_all, dtype=float)
    y_pred = np.array(y_pred_all, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    params_df = pd.DataFrame(best_params_all)
    best_params = params_df.mode().iloc[0].to_dict()

    return ModelResults(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        mae=mae,
        r2=r2,
        best_params=best_params,
    )


def fit_final_model_and_importance(
    *, model_name: str, pipeline: Pipeline, best_params: dict, X: pd.DataFrame, y: pd.Series
) -> FinalModelArtifacts:
    pipeline = pipeline.set_params(**best_params)
    pipeline.fit(X, y)

    # Use permutation importance so the plot stays comparable across model families.
    perm = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=30,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    # Re-score on training data for a quick reference (not a generalization estimate).
    y_pred = pipeline.predict(X)
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    return FinalModelArtifacts(
        model_name=model_name,
        best_params=best_params,
        mae=mae,
        r2=r2,
        permutation_importance_df=perm_df,
    )

def plot_results(
    *,
    elasticnet_cv: ModelResults,
    hgbr_cv: ModelResults,
    chosen: ModelResults,
    chosen_importance: pd.DataFrame,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Predicted vs actual (nested CV predictions)
    ax = axes[0]
    ax.scatter(chosen.y_true * 100, chosen.y_pred * 100, alpha=0.75, color="#2c7fb8", edgecolor="white", linewidth=0.5)
    lims = np.array(
        [
            min(chosen.y_true.min(), chosen.y_pred.min()),
            max(chosen.y_true.max(), chosen.y_pred.max()),
        ]
    )
    ax.plot(lims * 100, lims * 100, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"{chosen.model_name} (Nested CV): Predicted vs Actual Shock")
    ax.set_xlabel("Actual shock_2020 (%)")
    ax.set_ylabel("Predicted shock_2020 (%)")
    ax.text(
        0.02,
        0.98,
        f"MAE: {chosen.mae*100:.1f} pp\nR²: {chosen.r2:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    # Panel 2: Permutation importance (top features)
    ax = axes[1]
    top = chosen_importance.head(8).copy().iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#fdae61")
    ax.set_title("Permutation Importance (MAE decrease)")
    ax.set_xlabel("Importance (higher = more predictive)")
    ax.set_ylabel("")

    # Panel 3: Model comparison (nested CV)
    ax = axes[2]
    comp = pd.DataFrame(
        [
            {"model": elasticnet_cv.model_name, "mae_pp": elasticnet_cv.mae * 100, "r2": elasticnet_cv.r2},
            {"model": hgbr_cv.model_name, "mae_pp": hgbr_cv.mae * 100, "r2": hgbr_cv.r2},
        ]
    )
    ax.bar(comp["model"], comp["mae_pp"], color=["#74add1", "#f46d43"])
    ax.set_title("Model Comparison (Nested CV)")
    ax.set_ylabel("MAE (percentage points)")
    for i, row in comp.iterrows():
        ax.text(i, row["mae_pp"] + 0.2, f"R²={row['r2']:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Task 3 Model: Explaining 2020 Tourism Shock from 2019 Country Indicators",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    df = load_joined_data(DATA_PATH)
    X, y = build_modeling_table(df)
    if len(X) < 30:
        raise ValueError(f"Too few countries with valid shock data after filtering: N={len(X)}")

    elasticnet_pipeline = build_elasticnet_pipeline(feature_names=list(X.columns))
    elasticnet_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
    }
    elasticnet_cv = nested_cv_predictions(
        model_name="ElasticNet",
        pipeline=elasticnet_pipeline,
        param_grid=elasticnet_grid,
        X=X,
        y=y,
    )

    hgbr_pipeline = build_hgbr_pipeline(feature_names=list(X.columns))
    hgbr_grid = {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3, None],
        "model__max_leaf_nodes": [15, 31],
        "model__min_samples_leaf": [10, 20],
        "model__l2_regularization": [0.0, 0.1],
    }
    hgbr_cv = nested_cv_predictions(
        model_name="HistGradientBoosting",
        pipeline=hgbr_pipeline,
        param_grid=hgbr_grid,
        X=X,
        y=y,
    )

    # Choose final model: default to ElasticNet for interpretability unless the nonlinear model is clearly better.
    chosen = elasticnet_cv
    chosen_pipeline = elasticnet_pipeline
    if (hgbr_cv.mae <= elasticnet_cv.mae * 0.95) and (hgbr_cv.r2 >= elasticnet_cv.r2):
        chosen = hgbr_cv
        chosen_pipeline = hgbr_pipeline

    chosen_artifacts = fit_final_model_and_importance(
        model_name=chosen.model_name,
        pipeline=chosen_pipeline,
        best_params=chosen.best_params,
        X=X,
        y=y,
    )

    plot_results(
        elasticnet_cv=elasticnet_cv,
        hgbr_cv=hgbr_cv,
        chosen=chosen,
        chosen_importance=chosen_artifacts.permutation_importance_df,
        out_path=OUT_FIG,
    )

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": elasticnet_cv.model_name,
                "nested_cv_mae": elasticnet_cv.mae,
                "nested_cv_r2": elasticnet_cv.r2,
                "best_params": elasticnet_cv.best_params,
            },
            {
                "model": hgbr_cv.model_name,
                "nested_cv_mae": hgbr_cv.mae,
                "nested_cv_r2": hgbr_cv.r2,
                "best_params": hgbr_cv.best_params,
            },
            {
                "model": f"{chosen_artifacts.model_name} (refit)",
                "nested_cv_mae": np.nan,
                "nested_cv_r2": np.nan,
                "best_params": chosen_artifacts.best_params,
            },
        ]
    ).to_csv(OUT_METRICS, index=False)

    print("Saved:", OUT_FIG)
    print("Saved:", OUT_METRICS)
    print(f"N={len(X)} countries")
    print(f"ElasticNet (nested CV) MAE={elasticnet_cv.mae:.4f}, R2={elasticnet_cv.r2:.4f}, params={elasticnet_cv.best_params}")
    print(f"HistGradientBoosting (nested CV) MAE={hgbr_cv.mae:.4f}, R2={hgbr_cv.r2:.4f}, params={hgbr_cv.best_params}")
    print(f"Chosen: {chosen.model_name} (refit training MAE={chosen_artifacts.mae:.4f}, R2={chosen_artifacts.r2:.4f})")
    print("Top permutation importances:")
    print(chosen_artifacts.permutation_importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
