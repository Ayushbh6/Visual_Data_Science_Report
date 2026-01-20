"""
Script to download World Bank datasets and verify requirements
Dataset 1: International Tourism Arrivals (ST.INT.ARVL)
Dataset 2: World Development Indicators (WDI)
"""

import pandas as pd
import numpy as np
import requests
import io
from typing import Dict, List, Tuple

# World Bank API base URL
WB_API_BASE = "https://api.worldbank.org/v2"

def download_wb_indicator(indicator_code: str, start_year: int = 2018, end_year: int = 2024) -> pd.DataFrame:
    """
    Download World Bank indicator data
    
    Args:
        indicator_code: World Bank indicator code (e.g., 'ST.INT.ARVL')
        start_year: Start year
        end_year: End year
    
    Returns:
        DataFrame with country, year, and value columns
    """
    url = f"{WB_API_BASE}/country/all/indicator/{indicator_code}"
    params = {
        'date': f'{start_year}:{end_year}',
        'format': 'json',
        'per_page': 20000  # Large number to get all data
    }
    
    print(f"Downloading {indicator_code}...")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            print(f"Warning: No data returned for {indicator_code}")
            return pd.DataFrame()
        
        # Extract data
        records = []
        for item in data[1]:
            records.append({
                'country_code': item.get('country', {}).get('id', ''),
                'country_name': item.get('country', {}).get('value', ''),
                'year': int(item.get('date', 0)),
                'value': item.get('value'),
                'indicator': indicator_code
            })
        
        df = pd.DataFrame(records)
        print(f"  Downloaded {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error downloading {indicator_code}: {e}")
        return pd.DataFrame()

def download_tourism_data() -> pd.DataFrame:
    """Download international tourism arrivals data (primary indicator)"""
    # Using just arrivals to ensure full 2018-2024 coverage
    # When joined with WDI, we'll have 11+ variables total, meeting the >5 requirement
    return download_wb_indicator('ST.INT.ARVL', 2018, 2024)

def download_wdi_indicators() -> pd.DataFrame:
    """Download multiple WDI indicators"""
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'EN.POP.DNST': 'Population density (people per sq. km of land area)',
        'SP.POP.TOTL': 'Population, total',
        'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
        'SH.XPD.CHEX.PC.CD': 'Current health expenditure per capita (current US$)',
        'SH.MED.BEDS.ZS': 'Hospital beds (per 1,000 people)',
        'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)',
        'IS.AIR.PSGR': 'Air transport, passengers carried'
    }
    
    all_data = []
    for code, name in indicators.items():
        df = download_wb_indicator(code, 2018, 2024)
        if not df.empty:
            df['indicator_name'] = name
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return pd.DataFrame()

def pivot_tourism_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tourism data from long to wide format"""
    if df.empty:
        return pd.DataFrame()
    
    # Create pivot table: country_code, year as index, indicators as columns
    pivot_df = df.pivot_table(
        index=['country_code', 'country_name', 'year'],
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns.name = None
    return pivot_df

def pivot_wdi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot WDI data from long to wide format"""
    if df.empty:
        return pd.DataFrame()
    
    # Create pivot table: country_code, year as index, indicators as columns
    pivot_df = df.pivot_table(
        index=['country_code', 'country_name', 'year'],
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns.name = None
    return pivot_df

def analyze_requirements(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """Analyze if datasets meet the 3 requirements"""
    results = {
        'requirement_1_multidimensional': False,
        'requirement_2_sufficient_rows': False,
        'requirement_3_geographic': False,
        'details': {}
    }
    
    # Requirement 1: >5 multidimensional variables
    if not df1.empty:
        # Count variables (exclude identifier columns)
        exclude_cols = ['country_code', 'country_name', 'year', 'country_name_tourism', 'country_name_wdi']
        df1_vars = len([col for col in df1.columns if col not in exclude_cols])
        results['details']['dataset1_variables'] = df1_vars
        results['details']['dataset1_columns'] = list(df1.columns)
        results['details']['dataset1_variable_names'] = [col for col in df1.columns if col not in exclude_cols]
    
    if not df2.empty:
        # Count variables (exclude identifier columns)
        exclude_cols = ['country_code', 'country_name', 'year', 'country_name_tourism', 'country_name_wdi']
        df2_vars = len([col for col in df2.columns if col not in exclude_cols])
        results['details']['dataset2_variables'] = df2_vars
        results['details']['dataset2_columns'] = list(df2.columns)
        results['details']['dataset2_variable_names'] = [col for col in df2.columns if col not in exclude_cols]
    
    total_vars = results['details'].get('dataset1_variables', 0) + results['details'].get('dataset2_variables', 0)
    results['requirement_1_multidimensional'] = total_vars > 5
    results['details']['total_variables'] = total_vars
    
    # Requirement 2: Sufficient data rows
    if not df1.empty:
        results['details']['dataset1_rows'] = len(df1)
        results['details']['dataset1_countries'] = df1['country_code'].nunique()
        results['details']['dataset1_years'] = sorted(df1['year'].unique())
    
    if not df2.empty:
        results['details']['dataset2_rows'] = len(df2)
        results['details']['dataset2_countries'] = df2['country_code'].nunique()
        results['details']['dataset2_years'] = sorted(df2['year'].unique())
    
    min_rows = min(
        results['details'].get('dataset1_rows', 0),
        results['details'].get('dataset2_rows', 0)
    )
    results['requirement_2_sufficient_rows'] = min_rows >= 100  # At least 100 rows
    results['details']['min_rows'] = min_rows
    
    # Requirement 3: Geographic/spatial information
    has_geo = False
    if not df1.empty:
        has_geo = 'country_code' in df1.columns and 'country_name' in df1.columns
    if not df2.empty:
        has_geo = has_geo or ('country_code' in df2.columns and 'country_name' in df2.columns)
    
    results['requirement_3_geographic'] = has_geo
    results['details']['has_country_codes'] = has_geo
    
    return results

def test_join(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Test joining the two datasets"""
    join_results = {
        'can_join': False,
        'join_type': None,
        'matched_countries': 0,
        'matched_years': 0,
        'total_joined_rows': 0,
        'missing_values': {},
        'common_keys': []
    }
    
    if df1.empty or df2.empty:
        return pd.DataFrame(), join_results
    
    # Check common keys
    common_keys = []
    if 'country_code' in df1.columns and 'country_code' in df2.columns:
        common_keys.append('country_code')
    elif 'country_name' in df1.columns and 'country_name' in df2.columns:
        common_keys.append('country_name')
    
    if 'year' in df1.columns and 'year' in df2.columns:
        common_keys.append('year')
    
    join_results['common_keys'] = common_keys
    
    if not common_keys:
        return pd.DataFrame(), join_results
    
    # Perform inner join
    try:
        joined = pd.merge(df1, df2, on=common_keys, how='inner', suffixes=('_tourism', '_wdi'))
        join_results['can_join'] = True
        join_results['join_type'] = 'inner'
        join_results['total_joined_rows'] = len(joined)
        
        if 'country_code' in common_keys:
            join_results['matched_countries'] = joined['country_code'].nunique()
        elif 'country_name' in common_keys:
            join_results['matched_countries'] = joined['country_name'].nunique()
        
        if 'year' in common_keys:
            join_results['matched_years'] = sorted(joined['year'].unique())
        
        # Check for missing values
        join_results['missing_values'] = joined.isnull().sum().to_dict()
        
        return joined, join_results
        
    except Exception as e:
        print(f"Error joining datasets: {e}")
        return pd.DataFrame(), join_results

def main():
    """Main function to download and analyze datasets"""
    print("="*70)
    print("WORLD BANK DATASET DOWNLOAD AND ANALYSIS")
    print("="*70)
    print()
    
    # Download Dataset 1: Tourism Arrivals
    print("DATASET 1: International Tourism Arrivals")
    print("-"*70)
    tourism_df = download_tourism_data()
    
    if not tourism_df.empty:
        print(f"\nTourism Data Shape: {tourism_df.shape}")
        print(f"Columns: {list(tourism_df.columns)}")
        print(f"\nFirst few rows:")
        print(tourism_df.head())
        print(f"\nCountries: {tourism_df['country_code'].nunique()}")
        print(f"Years: {sorted(tourism_df['year'].unique())}")
        print(f"\nNote: This dataset has 1 variable (arrivals).")
        print(f"      When joined with WDI dataset, total will be 11+ variables.")
    else:
        print("ERROR: Could not download tourism data")
        return
    
    print("\n" + "="*70)
    
    # Download Dataset 2: WDI Indicators
    print("\nDATASET 2: World Development Indicators")
    print("-"*70)
    wdi_df_long = download_wdi_indicators()
    
    if not wdi_df_long.empty:
        print(f"\nWDI Data (long format) Shape: {wdi_df_long.shape}")
        wdi_df = pivot_wdi_data(wdi_df_long)
        print(f"\nWDI Data (pivoted) Shape: {wdi_df.shape}")
        print(f"Columns: {list(wdi_df.columns)}")
        print(f"\nFirst few rows:")
        print(wdi_df.head())
        print(f"\nCountries: {wdi_df['country_code'].nunique()}")
        print(f"Years: {sorted(wdi_df['year'].unique())}")
    else:
        print("ERROR: Could not download WDI data")
        return
    
    print("\n" + "="*70)
    
    # Analyze Requirements
    print("\nREQUIREMENTS ANALYSIS")
    print("-"*70)
    requirements = analyze_requirements(tourism_df, wdi_df)
    
    print(f"\nRequirement 1 (>5 multidimensional variables): {requirements['requirement_1_multidimensional']}")
    print(f"  Total variables: {requirements['details']['total_variables']}")
    print(f"  Dataset 1 variables: {requirements['details'].get('dataset1_variables', 0)}")
    if 'dataset1_variable_names' in requirements['details']:
        print(f"    Variables: {requirements['details']['dataset1_variable_names']}")
    print(f"  Dataset 2 variables: {requirements['details'].get('dataset2_variables', 0)}")
    if 'dataset2_variable_names' in requirements['details']:
        print(f"    Variables: {requirements['details']['dataset2_variable_names']}")
    
    print(f"\nRequirement 2 (Sufficient data rows): {requirements['requirement_2_sufficient_rows']}")
    print(f"  Minimum rows: {requirements['details']['min_rows']}")
    print(f"  Dataset 1 rows: {requirements['details'].get('dataset1_rows', 0)}")
    print(f"  Dataset 2 rows: {requirements['details'].get('dataset2_rows', 0)}")
    
    print(f"\nRequirement 3 (Geographic/spatial information): {requirements['requirement_3_geographic']}")
    print(f"  Has country codes: {requirements['details']['has_country_codes']}")
    
    print("\n" + "="*70)
    
    # Test Join
    print("\nJOIN TEST")
    print("-"*70)
    joined_df, join_info = test_join(tourism_df, wdi_df)
    
    if join_info['can_join']:
        print(f"✅ Datasets can be joined!")
        print(f"  Join keys: {join_info['common_keys']}")
        print(f"  Joined rows: {join_info['total_joined_rows']}")
        print(f"  Matched countries: {join_info['matched_countries']}")
        print(f"  Matched years: {join_info['matched_years']}")
        print(f"\nJoined dataset shape: {joined_df.shape}")
        print(f"\nFirst few rows of joined data:")
        print(joined_df.head())
        
        print(f"\nMissing values in joined dataset:")
        missing = join_info['missing_values']
        for col, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count > 0:
                print(f"  {col}: {count} ({count/len(joined_df)*100:.1f}%)")
    else:
        print("❌ Datasets cannot be joined with current keys")
        print(f"  Common keys found: {join_info['common_keys']}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Save datasets
    import os
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    if not tourism_df.empty:
        tourism_df.to_csv(os.path.join(data_dir, 'tourism_arrivals.csv'), index=False)
        print(f"\n✅ Saved tourism data to: {os.path.join(data_dir, 'tourism_arrivals.csv')}")
    
    if not wdi_df.empty:
        wdi_df.to_csv(os.path.join(data_dir, 'wdi_indicators.csv'), index=False)
        print(f"✅ Saved WDI data to: {os.path.join(data_dir, 'wdi_indicators.csv')}")
    
    if not joined_df.empty:
        joined_df.to_csv(os.path.join(data_dir, 'joined_dataset.csv'), index=False)
        print(f"✅ Saved joined dataset to: {os.path.join(data_dir, 'joined_dataset.csv')}")

if __name__ == "__main__":
    main()

