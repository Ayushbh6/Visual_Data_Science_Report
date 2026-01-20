import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "joined_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_dataset.csv")


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    return df


def analyze_missingness(df):
    print("\n--- Missingness Analysis ---")
    print(f"Years present in raw joined dataset: {sorted(df['year'].unique())}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    # Check target variable by year
    print("\nTarget Variable (value) missingness by year:")
    print(df.groupby("year")["value"].apply(lambda x: x.isnull().mean()))


def analyze_key_coverage():
    """
    Simple join-key coverage analysis between the raw tourism and WDI indicator files.
    Reports how many country-year combinations exist only in one source.
    """
    tourism_file = os.path.join(DATA_DIR, "tourism_arrivals.csv")
    wdi_file = os.path.join(DATA_DIR, "wdi_indicators.csv")

    if not (os.path.exists(tourism_file) and os.path.exists(wdi_file)):
        print("\n--- Join Key Coverage ---")
        print("Raw tourism and/or WDI files not found; skipping key coverage analysis.")
        return

    tourism = pd.read_csv(tourism_file)
    wdi = pd.read_csv(wdi_file)

    tourism_keys = set(zip(tourism["country_code"], tourism["year"]))
    wdi_keys = set(zip(wdi["country_code"], wdi["year"]))

    only_tourism = tourism_keys - wdi_keys
    only_wdi = wdi_keys - tourism_keys
    both = tourism_keys & wdi_keys

    print("\n--- Join Key Coverage ---")
    print(f"Country-year keys in tourism only: {len(only_tourism)}")
    print(f"Country-year keys in WDI only: {len(only_wdi)}")
    print(f"Country-year keys in both datasets: {len(both)}")
    print(f"Countries in tourism only: {len({c for c, _ in only_tourism})}")
    print(f"Countries in WDI only: {len({c for c, _ in only_wdi})}")


def clean_data(df):
    print("\n--- Cleaning Data ---")
    df_clean = df.copy()

    # 1. Drop rows with missing country identifier
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=["country_code"])
    print(f"Dropped {initial_rows - len(df_clean)} rows with missing country_code")

    # 2. Filter out World Bank regional aggregates (keep only true ISO country codes)
    # Real ISO 3166-1 alpha-2/3 codes are purely alphabetic (e.g., US, USA, DE, DEU)
    # Aggregates contain numbers or are special codes (e.g., 1A, 1W, 4E, 7E, 8S, B8, Z4, etc.)
    initial_rows = len(df_clean)
    # Keep only codes that are purely alphabetic (2 or 3 uppercase letters)
    df_clean = df_clean[df_clean["country_code"].str.match(r"^[A-Z]{2,3}$")]
    dropped_aggregates = initial_rows - len(df_clean)
    print(
        f"Dropped {dropped_aggregates} rows with regional aggregate codes (non-ISO country codes)"
    )

    # 3. Handle Target Variable (value)
    # We found that 2021-2024 is 100% missing. We will filter to 2018-2020 for the main analysis dataset.
    valid_years = [2018, 2019, 2020]
    df_clean = df_clean[df_clean["year"].isin(valid_years)]
    print(
        f"Filtered to years {valid_years} (dropping 2021+ because tourism arrivals are almost entirely missing)."
    )
    print(f"Shape after year filtering: {df_clean.shape}")

    # 4. Impute Predictors
    # Sort by country and year to ensure ffill/bfill works temporally
    df_clean = df_clean.sort_values(["country_code", "year"])

    # List of predictor columns (excluding identifiers and target)
    identifiers = [
        "country_code",
        "country_name_tourism",
        "year",
        "value",
        "indicator",
        "country_name_wdi",
    ]
    predictors = [c for c in df_clean.columns if c not in identifiers]

    print(f"Imputing predictors (excluding identifiers and target): {predictors}")
    total_missing_before = df_clean[predictors].isnull().sum().sum()
    print(f"Total missing predictor values before imputation: {total_missing_before}")

    # Strategy: Group by country and ffill/bfill
    # This handles cases where a country has data in 2018 but not 2019, etc.
    # Rationale: For time-series data within a country, the most recent available value
    # is often a reasonable estimate for missing values
    for col in predictors:
        df_clean[col] = df_clean.groupby("country_code")[col].transform(
            lambda x: x.ffill().bfill()
        )

    # 5. Handle remaining missing values (countries with NO data for a variable)
    # Fill with global median for that year
    # Rationale: When a country has no data for a variable at all, we use the
    # median value across all countries for that year as a reasonable estimate
    for col in predictors:
        remaining_nulls = df_clean[col].isnull().sum()
        if remaining_nulls > 0:
            print(
                f"  Filling {remaining_nulls} remaining nulls in {col} with yearly median"
            )
            df_clean[col] = df_clean.groupby("year")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # If still null (e.g., variable missing for ALL countries in a year), drop the column or fill with 0
    # Let's check
    remaining_nulls_global = df_clean[predictors].isnull().sum().sum()
    if remaining_nulls_global > 0:
        print(
            f"  Warning: {remaining_nulls_global} nulls remain after country-level and yearly imputation."
        )
        # Identify and drop columns with >50% missing values
        # Rationale: Variables with majority missing values are unreliable for analysis
        null_counts = df_clean[predictors].isnull().sum()
        n_rows = len(df_clean)
        to_drop = null_counts[null_counts > n_rows * 0.5].index.tolist()
        if to_drop:
            print(f"  Dropping columns with >50% missing values: {to_drop}")
            df_clean = df_clean.drop(columns=to_drop)
        else:
            print(
                "  No columns exceed 50% missingness; remaining nulls will persist in the cleaned dataset."
            )
    # Recompute missingness after potential column drops
    remaining_predictors = [c for c in df_clean.columns if c not in identifiers]
    total_missing_after = df_clean[remaining_predictors].isnull().sum().sum()
    print(
        f"Total missing predictor values after imputation / column drop: {total_missing_after}"
    )

    return df_clean


def feature_engineering(df):
    print("\n--- Feature Engineering ---")
    # We need to pivot to calculate year-over-year changes efficiently
    # Create a separate DF for this

    # Pivot target
    tourism_pivot = df.pivot(index="country_code", columns="year", values="value")

    # Calculate Shock (2020 vs 2019) only where both years are present and 2019 arrivals are positive
    if 2019 in tourism_pivot.columns and 2020 in tourism_pivot.columns:
        base = tourism_pivot[2019]
        current = tourism_pivot[2020]
        valid_mask = base.notnull() & current.notnull() & (base > 0)
        tourism_pivot["shock_2020"] = np.nan
        tourism_pivot.loc[valid_mask, "shock_2020"] = (
            current[valid_mask] - base[valid_mask]
        ) / base[valid_mask]
        print(
            f"Calculated 'shock_2020' for {valid_mask.sum()} countries with non-missing, positive 2019/2020 arrivals."
        )
    else:
        print(
            "Could not calculate 'shock_2020': 2019 and/or 2020 arrivals not present in pivot."
        )

    # Merge back into main DF? Or just keep it as a country-level metric?
    # For the "cleaned_dataset.csv", we usually keep the panel structure (Long format).
    # But we can add the static 'shock' variable to all rows of a country.

    df = df.merge(tourism_pivot[["shock_2020"]], on="country_code", how="left")

    return df


def save_data(df, filepath):
    print(f"\nSaving cleaned data to {filepath}...")
    df.to_csv(filepath, index=False)
    print("Done.")


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    analyze_missingness(df)
    analyze_key_coverage()
    df_clean = clean_data(df)
    df_final = feature_engineering(df_clean)

    # Final check
    print("\nFinal Data Stats:")
    print(df_final.info())

    save_data(df_final, OUTPUT_FILE)
