import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
import geopandas as gpd
import matplotlib.cm as cm

# Configuration
DATA_DIR = "data"
IMG_DIR = "Docs/images"
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_dataset.csv")
RAW_INPUT_FILE = os.path.join(DATA_DIR, "joined_dataset.csv")


def load_data(filepath):
    """Load dataset from CSV file with error handling."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        raise
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise


def plot_missingness_clean(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Heatmap (Cleaned Dataset)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "missingness_heatmap.png"))
    plt.close()
    print("Saved missingness_heatmap.png")


def plot_missingness_by_year(filepath):
    """
    Data quality visualization on the raw joined dataset.
    Shows percentage of missing values per variable and year (2018–2024).
    """
    df_raw = load_data(filepath)
    # Focus on target + key predictors for readability
    cols = [
        "value",
        "EN.POP.DNST",
        "IS.AIR.PSGR",
        "IT.NET.USER.ZS",
        "NY.GDP.MKTP.CD",
        "NY.GDP.MKTP.KD.ZG",
        "NY.GDP.PCAP.CD",
        "SH.MED.BEDS.ZS",
        "SH.XPD.CHEX.PC.CD",
        "SP.DYN.LE00.IN",
        "SP.POP.TOTL",
    ]
    available_cols = [c for c in cols if c in df_raw.columns]
    missing_by_year = df_raw.groupby("year")[available_cols].apply(
        lambda x: x.isnull().mean() * 100
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        missing_by_year.sort_index(),
        annot=False,
        cmap="magma_r",
        cbar_kws={"label": "Missing (%)"},
    )
    plt.title("Missing Data by Year and Variable (Joined Dataset)")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "missingness_by_year_heatmap.png"))
    plt.close()
    print("Saved missingness_by_year_heatmap.png")


def plot_imputation_effect(raw_filepath, cleaned_df, column, filename, pretty_name):
    """
    Compare the distribution of a variable before and after imputation (2018–2020 only).
    """
    df_raw = load_data(raw_filepath)
    df_raw = df_raw[df_raw["year"].between(2018, 2020)]
    raw_vals = df_raw[column].dropna()
    clean_vals = cleaned_df[column].dropna()

    if raw_vals.empty or clean_vals.empty:
        print(f"Not enough data to plot imputation effect for {column}")
        return

    plt.figure(figsize=(10, 5))
    sns.kdeplot(raw_vals, label="Raw (2018–2020)", fill=True, alpha=0.4)
    sns.kdeplot(clean_vals, label="Cleaned (after imputation)", fill=True, alpha=0.4)
    plt.title(f"{pretty_name} – Before vs After Imputation")
    plt.xlabel(pretty_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close()
    print(f"Saved {filename} for {column}")


def insight_1_shock_distribution(df):
    """
    Insight 1: The Global Collapse
    Visualize the distribution of the 2020 Shock (Percentage Change).
    """
    # Get unique countries with shock data
    shock_data = df[["country_code", "shock_2020"]].drop_duplicates().dropna()
    # Filter out extreme outliers (e.g. growth > 0 for pure shock analysis, or just crazy > +1.0 values)
    shock_data = shock_data[shock_data["shock_2020"] < 1.0]

    plt.figure(figsize=(10, 6))
    sns.histplot(shock_data["shock_2020"] * 100, kde=True, bins=20, color="salmon")
    plt.title("Distribution of Tourism Shock (2020 vs 2019)")
    plt.xlabel("Tourism shock 2020 vs 2019 (%)")
    plt.ylabel("Number of Countries")
    plt.axvline(x=0, color="black", linestyle="--")

    # Add stats
    mean_shock = shock_data["shock_2020"].mean() * 100
    median_shock = shock_data["shock_2020"].median() * 100
    plt.text(
        0.05,
        0.9,
        f"Mean: {mean_shock:.1f}%\nMedian: {median_shock:.1f}%",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "insight_1_shock_distribution.png"))
    plt.close()
    print(f"Saved insight_1_shock_distribution.png (N={len(shock_data)})")


def plot_scatter_with_correlation(
    df, x_col, y_col, title, filename, x_label=None, color="blue", log_x=False
):
    # Prepare data (2019 predictors vs 2020 shock)
    df_2019 = df[df["year"] == 2019].copy()
    plot_data = df_2019[["country_code", x_col, y_col]].dropna()

    # Remove outliers for shock if necessary (e.g. > 0) to keep analysis focused on the "Crash"
    plot_data = plot_data[plot_data[y_col] < 0]

    if len(plot_data) < 2:
        print(f"Not enough data for {filename}")
        return

    # Work with percentage shock for interpretability
    plot_data = plot_data.copy()
    plot_data["shock_pct"] = plot_data[y_col] * 100

    # Calculate Correlation
    corr, _ = pearsonr(plot_data[x_col], plot_data["shock_pct"])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x=x_col, y="shock_pct", alpha=0.6, color=color)

    if log_x:
        plt.xscale("log")

    # Trend line
    sns.regplot(
        data=plot_data, x=x_col, y="shock_pct", scatter=False, color=color, logx=log_x
    )

    plt.title(f"{title}\nCorrelation (r): {corr:.2f}")
    plt.xlabel(x_label or x_col)
    plt.ylabel("Tourism shock 2020 vs 2019 (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close()
    print(f"Saved {filename} (N={len(plot_data)}, r={corr:.2f})")


def insight_2_gdp_vs_shock(df):
    plot_scatter_with_correlation(
        df,
        "NY.GDP.PCAP.CD",
        "shock_2020",
        "GDP per Capita (2019) vs. Tourism Shock",
        "insight_2_gdp_vs_shock.png",
        x_label="GDP per capita (current US$, 2019)",
        color="red",
        log_x=True,
    )


def insight_3_internet_vs_shock(df):
    plot_scatter_with_correlation(
        df,
        "IT.NET.USER.ZS",
        "shock_2020",
        "Internet Penetration (2019) vs. Tourism Shock",
        "insight_3_internet_vs_shock.png",
        x_label="Internet users (% of population, 2019)",
        color="green",
    )


def insight_4_health_exp_vs_shock(df):
    plot_scatter_with_correlation(
        df,
        "SH.XPD.CHEX.PC.CD",
        "shock_2020",
        "Health Expenditure (2019) vs. Tourism Shock",
        "insight_4_health_vs_shock.png",
        x_label="Health expenditure per capita (US$, 2019)",
        color="purple",
        log_x=True,
    )


def insight_5_tourism_dependence_vs_shock(df):
    """
    Insight: Tourism dependence vs. depth of the 2020 shock.
    Uses 2019 arrivals per 1,000 residents as a proxy for tourism dependence.
    """
    df_2019 = df[df["year"] == 2019].copy()
    required_cols = ["country_code", "value", "SP.POP.TOTL", "shock_2020"]
    if not all(c in df_2019.columns for c in required_cols):
        print("Missing required columns for tourism dependence insight; skipping.")
        return

    plot_data = df_2019[required_cols].dropna()
    # Focus on countries with a clear negative shock and non-zero arrivals
    plot_data = plot_data[(plot_data["shock_2020"] < 0) & (plot_data["value"] > 0)]

    if len(plot_data) < 2:
        print("Not enough data for tourism dependence insight.")
        return

    plot_data = plot_data.copy()
    plot_data["arrivals_per_1000"] = (
        plot_data["value"] / plot_data["SP.POP.TOTL"]
    ) * 1000
    plot_data["shock_pct"] = plot_data["shock_2020"] * 100

    corr, _ = pearsonr(plot_data["arrivals_per_1000"], plot_data["shock_pct"])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_data, x="arrivals_per_1000", y="shock_pct", alpha=0.6, color="orange"
    )
    sns.regplot(
        data=plot_data,
        x="arrivals_per_1000",
        y="shock_pct",
        scatter=False,
        color="orange",
    )
    plt.title(
        f"Tourism Dependence (2019) vs. Tourism Shock\nCorrelation (r): {corr:.2f}"
    )
    plt.xlabel("International tourism arrivals per 1,000 residents (2019)")
    plt.ylabel("Tourism shock 2020 vs 2019 (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "insight_5_tourism_dependence_vs_shock.png"))
    plt.close()
    print(
        f"Saved insight_5_tourism_dependence_vs_shock.png (N={len(plot_data)}, r={corr:.2f})"
    )


def insight_6_top_bottom_countries(df):
    """
    Insight: Which countries were hit hardest and which were most resilient?
    Shows the Top 10 most affected (deepest shock) and Top 10 least affected countries.
    """
    # Get unique countries with shock data (use 2019 row for country names)
    df_2019 = df[df["year"] == 2019].copy()
    shock_data = df_2019[
        ["country_code", "country_name_tourism", "shock_2020"]
    ].dropna()
    # Filter to negative shocks only (countries that experienced decline)
    shock_data = shock_data[shock_data["shock_2020"] < 0].copy()
    shock_data["shock_pct"] = shock_data["shock_2020"] * 100

    if len(shock_data) < 20:
        print(f"Not enough data for top/bottom countries insight (N={len(shock_data)})")
        return

    # Top 10 most affected (most negative shock)
    most_affected = shock_data.nsmallest(10, "shock_pct")
    # Top 10 least affected (least negative shock, closest to 0)
    least_affected = shock_data.nlargest(10, "shock_pct")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Most affected
    ax1 = axes[0]
    colors_most = cm.Reds(np.linspace(0.4, 0.9, 10))
    bars1 = ax1.barh(
        most_affected["country_name_tourism"],
        most_affected["shock_pct"],
        color=colors_most,
    )
    ax1.set_xlabel("Tourism Shock (%)")
    ax1.set_title("Top 10 Most Affected Countries\n(Deepest Tourism Decline)")
    ax1.invert_yaxis()
    # Add value labels
    for bar, val in zip(bars1, most_affected["shock_pct"]):
        ax1.text(
            val - 2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="right",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    # Right: Least affected
    ax2 = axes[1]
    colors_least = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
    bars2 = ax2.barh(
        least_affected["country_name_tourism"],
        least_affected["shock_pct"],
        color=colors_least,
    )
    ax2.set_xlabel("Tourism Shock (%)")
    ax2.set_title("Top 10 Least Affected Countries\n(Smallest Tourism Decline)")
    ax2.invert_yaxis()
    # Add value labels
    for bar, val in zip(bars2, least_affected["shock_pct"]):
        ax2.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
        )

    plt.suptitle(
        "COVID-19 Tourism Impact: Winners and Losers (2020 vs 2019)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "insight_6_top_bottom_countries.png"), dpi=150)
    plt.close()
    print(f"Saved insight_6_top_bottom_countries.png")


def insight_7_choropleth_shock(df):
    """
    Geographic visualization: Choropleth map showing shock_2020 by country.
    Downloads Natural Earth data directly.
    """
    # Get unique countries with shock data
    shock_data = df[["country_code", "shock_2020"]].drop_duplicates().dropna()
    shock_data = shock_data.copy()
    shock_data["shock_pct"] = shock_data["shock_2020"] * 100

    # Download Natural Earth countries shapefile directly
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    try:
        world = gpd.read_file(url)
    except Exception as e:
        print(f"Could not load Natural Earth shapefile ({e}); skipping choropleth.")
        return

    # Our data uses 2-letter ISO codes (e.g., US, FR, JP).
    # Natural Earth provides ISO_A2 for 2-letter codes and ISO_A3 for 3-letter.
    iso_col = None
    for candidate in ["ISO_A2", "ISO_A3", "ADM0_A3"]:
        if candidate in world.columns:
            iso_col = candidate
            break

    if iso_col is None:
        print(
            "Could not find a suitable country code column in Natural Earth shapefile; "
            "skipping choropleth."
        )
        return

    world = world.merge(
        shock_data, left_on=iso_col, right_on="country_code", how="left"
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot all countries first (base layer in light gray)
    world.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.3)

    # Overlay countries with data using a colormap
    has_data = world[world["shock_pct"].notna()]
    if len(has_data) > 0:
        has_data.plot(
            column="shock_pct",
            ax=ax,
            legend=True,
            legend_kwds={
                "label": "Tourism Shock 2020 vs 2019 (%)",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.02,
            },
            cmap="RdYlGn",  # Red (bad) to Green (good)
            edgecolor="white",
            linewidth=0.3,
            vmin=-100,
            vmax=0,
        )

    ax.set_title(
        "Global Tourism Shock (2020 vs 2019)\nPercentage Change in International Arrivals",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "insight_7_choropleth_shock.png"), dpi=150)
    plt.close()
    print(f"Saved insight_7_choropleth_shock.png (N={len(has_data)} countries mapped)")


def insight_8_regional_island_analysis(df):
    """
    Insight: Not all islands are equal — Caribbean vs Pacific vs Asia.
    The real finding: distance to source markets and policy strictness matter more than
    simple "island vs land-border" distinction.
    """
    # Define regional categories using ISO2 country codes to match World Bank data
    caribbean = [
        "DM",
        "GD",
        "AG",
        "DO",
        "LC",
        "JM",
        "BB",
        "BS",
        "TT",
        "KN",
        "VC",
        "CU",
        "HT",
        "PR",
    ]
    pacific = [
        "AS",
        "WS",
        "FJ",
        "TO",
        "VU",
        "PG",
        "SB",
        "TV",
        "KI",
        "MH",
        "FM",
        "PW",
        "NR",
        "NZ",
        "AU",
    ]
    asian_islands = ["JP", "HK", "SG", "PH", "ID", "TW", "MO", "MV", "LK"]
    eu_land = [
        "FR",
        "DE",
        "IT",
        "ES",
        "PT",
        "NL",
        "BE",
        "LU",
        "AT",
        "CH",
        "PL",
        "CZ",
        "SK",
        "HU",
        "SI",
        "HR",
        "BA",
        "RS",
        "ME",
        "AL",
        "MK",
        "BG",
        "RO",
        "AD",
        "LI",
    ]

    df_2019 = df[df["year"] == 2019].copy()
    df_2019 = df_2019[df_2019["shock_2020"].notna()]
    df_2019 = df_2019[df_2019["shock_2020"] < 0]  # Exclude suspicious positive shocks
    df_2019["shock_pct"] = df_2019["shock_2020"] * 100

    # Categorize by region
    def categorize(code):
        if code in caribbean:
            return "Caribbean Islands"
        elif code in pacific:
            return "Pacific Islands"
        elif code in asian_islands:
            return "Asian Islands/Hubs"
        elif code in eu_land:
            return "Europe (Land Borders)"
        else:
            return "Other"

    df_2019["region"] = df_2019["country_code"].apply(categorize)

    # Filter to focus categories (exclude 'Other' for cleaner visualization)
    focus_categories = [
        "Caribbean Islands",
        "Europe (Land Borders)",
        "Pacific Islands",
        "Asian Islands/Hubs",
    ]
    plot_data = df_2019[df_2019["region"].isin(focus_categories)]

    if plot_data.empty:
        print("No data available for regional island analysis; skipping visualization.")
        return

    # Calculate summary statistics
    summary = (
        plot_data.groupby("region")["shock_pct"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Box plot comparing regions
    ax1 = axes[0]
    order = [
        "Caribbean Islands",
        "Europe (Land Borders)",
        "Pacific Islands",
        "Asian Islands/Hubs",
    ]
    colors = {
        "Caribbean Islands": "#27ae60",  # Green - best performers
        "Europe (Land Borders)": "#3498db",  # Blue
        "Pacific Islands": "#e67e22",  # Orange
        "Asian Islands/Hubs": "#c0392b",  # Red - worst performers
    }

    sns.boxplot(
        data=plot_data,
        x="region",
        y="shock_pct",
        order=order,
        hue="region",
        palette=colors,
        legend=False,
        ax=ax1,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Tourism Shock 2020 vs 2019 (%)")
    ax1.set_title("Tourism Shock by Region")
    ax1.tick_params(axis="x", rotation=15)
    ax1.axhline(
        y=-72, color="red", linestyle="--", alpha=0.5, label="Global Mean (-72%)"
    )
    ax1.legend(loc="lower left")

    # Right: Bar chart with annotations explaining WHY
    ax2 = axes[1]
    means = plot_data.groupby("region")["shock_pct"].mean().reindex(order)
    counts = plot_data.groupby("region")["shock_pct"].count().reindex(order)

    bars = ax2.barh(range(len(order)), means, color=[colors[c] for c in order])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order)
    ax2.set_xlabel("Mean Tourism Shock (%)")
    ax2.set_title("Why Not All Islands Are Equal")
    ax2.axvline(x=-72, color="red", linestyle="--", alpha=0.5)
    ax2.invert_yaxis()

    # Add value labels and explanations
    explanations = {
        "Caribbean Islands": "Short-haul from US\nRelaxed policies",
        "Europe (Land Borders)": "Drive-in tourism\nfrom neighbors",
        "Pacific Islands": "Remote from\nall markets",
        "Asian Islands/Hubs": "Zero-COVID policies\nStrict borders",
    }

    for i, (bar, region) in enumerate(zip(bars, order)):
        mean_val = means[region]
        n = counts[region]
        # Value label
        ax2.text(
            mean_val - 2,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.1f}% (n={n})",
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        # Explanation
        ax2.text(
            -5,
            bar.get_y() + bar.get_height() / 2,
            explanations[region],
            va="center",
            ha="left",
            fontsize=8,
            style="italic",
            color="darkgray",
        )

    ax2.set_xlim(-100, 0)

    plt.suptitle(
        "The Real Finding: Distance to Markets + Policy Strictness > Simple Geography",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "insight_8_accessibility_analysis.png"), dpi=150)
    plt.close()

    # Print summary for report
    print("Saved insight_8_accessibility_analysis.png")
    print("\nRegional Island Analysis Summary:")
    print(summary.to_string())

    if {
        "Caribbean Islands",
        "Europe (Land Borders)",
        "Pacific Islands",
        "Asian Islands/Hubs",
    }.issubset(summary.index):
        print(
            f"\nKey Finding: Caribbean islands ({summary.loc['Caribbean Islands', 'mean']:.0f}%) "
            f"performed as well as European land-border countries ({summary.loc['Europe (Land Borders)', 'mean']:.0f}%)."
        )
        print(
            f"Asian islands ({summary.loc['Asian Islands/Hubs', 'mean']:.0f}%) and Pacific islands "
            f"({summary.loc['Pacific Islands', 'mean']:.0f}%) collapsed more deeply."
        )
        diff = (
            summary.loc["Caribbean Islands", "mean"]
            - summary.loc["Asian Islands/Hubs", "mean"]
        )
        print(
            f"Difference between Caribbean and Asian islands: {diff:.1f} percentage points"
        )


if __name__ == "__main__":
    # Set style
    sns.set_theme(style="whitegrid")

    # Data quality: raw joined dataset (2018–2024)
    plot_missingness_by_year(RAW_INPUT_FILE)

    # Profiling on cleaned analysis dataset (2018–2020)
    df = load_data(INPUT_FILE)
    plot_missingness_clean(df)
    # Data-quality: impact of imputation on a highly incomplete indicator
    plot_imputation_effect(
        RAW_INPUT_FILE,
        df,
        column="SH.MED.BEDS.ZS",
        filename="imputation_effect_hospital_beds.png",
        pretty_name="Hospital beds (per 1,000 people)",
    )
    # Profiling insights
    insight_1_shock_distribution(df)
    insight_2_gdp_vs_shock(df)
    insight_3_internet_vs_shock(df)
    insight_4_health_exp_vs_shock(df)
    insight_5_tourism_dependence_vs_shock(df)
    insight_6_top_bottom_countries(df)
    insight_7_choropleth_shock(df)
    insight_8_regional_island_analysis(df)
