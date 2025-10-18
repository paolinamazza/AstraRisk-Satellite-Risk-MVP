"""
STEP 3: ENVIRONMENT RISK SCORE CALCULATION
===========================================
Calculate environment risk scores (0-100) for all LEO satellites based on
space weather conditions during their operational lifetime.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 3: ENVIRONMENT RISK SCORE CALCULATION")
print("="*70)

# ============================================================================
# STEP 3.1: Load Data
# ============================================================================
print("\n[1/7] Loading data...")

# Load LEO satellites
leo_sats = pd.read_csv('leo_satellites_enriched.csv')

# Deduplicate by NORAD_CAT_ID (same as step 2)
if leo_sats.duplicated('NORAD_CAT_ID').any():
    n_dups = leo_sats.duplicated('NORAD_CAT_ID').sum()
    print(f"Warning: Found {n_dups} duplicate NORAD_CAT_IDs, keeping first occurrence")
    leo_sats = leo_sats.drop_duplicates('NORAD_CAT_ID', keep='first').reset_index(drop=True)

print(f"Loaded {len(leo_sats):,} LEO satellites")

# Load space weather data
weather_data = pd.read_csv('omni2_LEO_risk_finalfinal.csv')
print(f"Loaded {len(weather_data):,} hours of space weather data")

# Convert date column
weather_data['date'] = pd.to_datetime(weather_data['date'], errors='coerce')

# Check data range
print(f"\nWeather data period:")
print(f"  Start: {weather_data['date'].min().date()}")
print(f"  End: {weather_data['date'].max().date()}")
print(f"  Duration: {(weather_data['date'].max() - weather_data['date'].min()).days} days")

# ============================================================================
# STEP 3.2: Understanding the Weather Data
# ============================================================================
print("\n[2/7] Understanding weather data structure...")

print("\nWeather data columns:")
print(weather_data.columns.tolist())

print("\nLEO_RiskScore_0_100 statistics:")
print(weather_data['LEO_RiskScore_0_100'].describe())

print("\nSample weather data:")
print(weather_data[['date', 'LEO_RiskScore_0_100', 'R_drag', 'R_SEU', 'R_ctrl']].head(10))

# ============================================================================
# STEP 3.3: Calculate Environment Score for Each Satellite
# ============================================================================
print("\n[3/7] Calculating environment scores for all satellites...")

def calculate_environment_score(satellite, weather_data):
    """
    Calculate environment risk score for a satellite

    Method:
    1. Filter weather data for satellite's operational period
    2. Calculate P95 (95th percentile) of LEO_RiskScore
    3. Apply altitude adjustment factor

    Args:
        satellite: Row from leo_satellites dataframe
        weather_data: DataFrame with hourly space weather scores

    Returns:
        dict with environment metrics
    """

    sat_id = satellite['NORAD_CAT_ID']
    launch_date = pd.to_datetime(satellite['LAUNCH_DATE'], errors='coerce')
    age_years = satellite['AGE_YEARS']
    altitude_km = satellite['ALTITUDE_MEAN_KM']

    # Handle invalid launch dates
    if pd.isna(launch_date):
        # Use age to estimate launch date
        launch_date = datetime.now() - timedelta(days=age_years*365.25)

    # Check if launch is before weather data starts
    weather_start = weather_data['date'].min()
    if launch_date < weather_start:
        launch_date = weather_start  # Use earliest available data

    # Calculate end date (current date or end of weather data, whichever is earlier)
    current_date = datetime.now()
    weather_end = weather_data['date'].max()
    end_date = min(current_date, weather_end)

    # Filter weather data for operational period
    sat_weather = weather_data[
        (weather_data['date'] >= launch_date) &
        (weather_data['date'] <= end_date)
    ]

    # Check if we have sufficient data
    if len(sat_weather) == 0:
        # Satellite launched after our weather data ends, use recent baseline
        recent_weather = weather_data.tail(365*24)  # Last year
        env_score_p95 = recent_weather['LEO_RiskScore_0_100'].quantile(0.95)
        env_score_mean = recent_weather['LEO_RiskScore_0_100'].mean()
        max_event = recent_weather['LEO_RiskScore_0_100'].max()
        n_hours = len(recent_weather)
        data_source = 'RECENT_BASELINE'
    elif len(sat_weather) < 24:  # Less than 1 day of data
        # Very new satellite, use recent data
        recent_weather = weather_data.tail(30*24)  # Last 30 days
        env_score_p95 = recent_weather['LEO_RiskScore_0_100'].quantile(0.95)
        env_score_mean = recent_weather['LEO_RiskScore_0_100'].mean()
        max_event = recent_weather['LEO_RiskScore_0_100'].max()
        n_hours = len(sat_weather)
        data_source = 'RECENT_PARTIAL'
    else:
        # Normal case: use satellite's actual operational period
        env_score_p95 = sat_weather['LEO_RiskScore_0_100'].quantile(0.95)
        env_score_mean = sat_weather['LEO_RiskScore_0_100'].mean()
        max_event = sat_weather['LEO_RiskScore_0_100'].max()
        n_hours = len(sat_weather)
        data_source = 'ACTUAL_PERIOD'

    # Count severe events (score > 60)
    if len(sat_weather) > 0:
        severe_events = len(sat_weather[sat_weather['LEO_RiskScore_0_100'] > 60])
        severe_events_per_year = severe_events / (n_hours / 8760) if n_hours > 0 else 0
    else:
        severe_events = 0
        severe_events_per_year = 0

    # Apply altitude adjustment factor
    # Lower altitudes experience more drag during space weather events
    if altitude_km < 400:
        altitude_factor = 1.20  # LEO very low, +20% risk
    elif altitude_km < 600:
        altitude_factor = 1.00  # LEO typical, baseline
    elif altitude_km < 1000:
        altitude_factor = 0.90  # LEO high, -10% risk
    else:
        altitude_factor = 0.75  # LEO very high, -25% risk

    # Apply adjustment (affects mainly the drag component which is 40% of total)
    # Adjusted score = score × (0.60 + 0.40 × altitude_factor)
    env_score_adjusted = env_score_p95 * (0.60 + 0.40 * altitude_factor)
    env_score_adjusted = max(0, min(100, env_score_adjusted))  # Clamp to 0-100

    return {
        'satellite_id': int(sat_id),
        'environment_score': round(env_score_adjusted, 1),
        'env_score_p95_raw': round(env_score_p95, 1),
        'env_score_mean': round(env_score_mean, 1),
        'max_event_score': round(max_event, 1),
        'severe_events_count': int(severe_events),
        'severe_events_per_year': round(severe_events_per_year, 1),
        'altitude_adjustment_factor': round(altitude_factor, 2),
        'n_hours_data': int(n_hours),
        'operational_days': int(n_hours / 24),
        'data_source': data_source,
        'launch_date_used': launch_date.date().isoformat(),
        'data_period_start': sat_weather['date'].min().date().isoformat() if len(sat_weather) > 0 else None,
        'data_period_end': sat_weather['date'].max().date().isoformat() if len(sat_weather) > 0 else None
    }

# Calculate environment scores for all satellites
print(f"\nCalculating environment scores for {len(leo_sats):,} satellites...")

env_results = []
for idx, (_, sat) in enumerate(leo_sats.iterrows()):
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx+1:,}/{len(leo_sats):,} satellites...")

    result = calculate_environment_score(sat, weather_data)
    env_results.append(result)

env_scores_df = pd.DataFrame(env_results)

print(f"\n✓ Calculated environment scores for {len(env_scores_df):,} satellites")

# ============================================================================
# STEP 3.4: Add Satellite Metadata
# ============================================================================
print("\n[4/7] Adding satellite metadata...")

# Merge with LEO satellites to add metadata
env_scores_full = env_scores_df.merge(
    leo_sats[['NORAD_CAT_ID', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
              'ALTITUDE_MEAN_KM', 'INCLINATION', 'ORBIT_CLASS', 'AGE_YEARS']],
    left_on='satellite_id',
    right_on='NORAD_CAT_ID',
    how='left'
)

# Drop duplicate column
env_scores_full = env_scores_full.drop('NORAD_CAT_ID', axis=1)

print(f"✓ Added metadata for {len(env_scores_full):,} satellites")

# ============================================================================
# STEP 3.5: Save Output
# ============================================================================
print("\n[5/7] Saving environment scores...")

# Reorder columns
output_cols = [
    'satellite_id', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
    'ALTITUDE_MEAN_KM', 'INCLINATION', 'ORBIT_CLASS', 'AGE_YEARS',
    'environment_score', 'env_score_p95_raw', 'env_score_mean', 'max_event_score',
    'severe_events_count', 'severe_events_per_year',
    'altitude_adjustment_factor', 'n_hours_data', 'operational_days',
    'data_source', 'launch_date_used', 'data_period_start', 'data_period_end'
]

env_scores_final = env_scores_full[output_cols].copy()

# Save to CSV
env_scores_final.to_csv('environment_scores_leo.csv', index=False)

print(f"\n✅ Saved {len(env_scores_final):,} environment scores to environment_scores_leo.csv")

# ============================================================================
# STEP 3.6: Print Summary Statistics
# ============================================================================
print("\n[6/7] Generating summary statistics...")
print("\n" + "="*70)
print("STEP 3 SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal LEO satellites with environment scores: {len(env_scores_final):,}")

print("\n--- ENVIRONMENT SCORE DISTRIBUTION ---")
print(env_scores_final['environment_score'].describe())

print("\n--- SCORE BY DATA SOURCE ---")
for source in env_scores_final['data_source'].unique():
    subset = env_scores_final[env_scores_final['data_source'] == source]
    print(f"\n{source} (n={len(subset):,}):")
    print(f"  Mean score: {subset['environment_score'].mean():.1f}")
    print(f"  Median score: {subset['environment_score'].median():.1f}")
    print(f"  Min: {subset['environment_score'].min():.1f}")
    print(f"  Max: {subset['environment_score'].max():.1f}")

print("\n--- RISK CATEGORIES ---")
risk_categories = {
    'Very Low (0-20)': (env_scores_final['environment_score'] < 20).sum(),
    'Low (20-40)': ((env_scores_final['environment_score'] >= 20) &
                    (env_scores_final['environment_score'] < 40)).sum(),
    'Medium (40-60)': ((env_scores_final['environment_score'] >= 40) &
                       (env_scores_final['environment_score'] < 60)).sum(),
    'High (60-80)': ((env_scores_final['environment_score'] >= 60) &
                     (env_scores_final['environment_score'] < 80)).sum(),
    'Very High (80-100)': (env_scores_final['environment_score'] >= 80).sum()
}

for category, count in risk_categories.items():
    pct = count / len(env_scores_final) * 100
    print(f"  {category}: {count:,} satellites ({pct:.1f}%)")

print("\n--- ALTITUDE ADJUSTMENT IMPACT ---")
print(f"Mean altitude factor: {env_scores_final['altitude_adjustment_factor'].mean():.2f}")
print(f"Range: {env_scores_final['altitude_adjustment_factor'].min():.2f} - "
      f"{env_scores_final['altitude_adjustment_factor'].max():.2f}")

altitude_bins = [0, 400, 600, 1000, 2000]
altitude_labels = ['<400km', '400-600km', '600-1000km', '1000-2000km']
env_scores_final['altitude_bin'] = pd.cut(
    env_scores_final['ALTITUDE_MEAN_KM'],
    bins=altitude_bins,
    labels=altitude_labels
)

print("\nMean environment score by altitude:")
for label in altitude_labels:
    subset = env_scores_final[env_scores_final['altitude_bin'] == label]
    if len(subset) > 0:
        print(f"  {label}: {subset['environment_score'].mean():.1f} (n={len(subset):,})")

print("\n--- SEVERE EVENTS ANALYSIS ---")
print(f"Total severe events (score >60): {env_scores_final['severe_events_count'].sum():,}")
print(f"Satellites with ≥1 severe event: {(env_scores_final['severe_events_count'] > 0).sum():,}")
print(f"Max severe events for one satellite: {env_scores_final['severe_events_count'].max():,}")

print("\n--- TOP 10 HIGHEST RISK SATELLITES ---")
top10 = env_scores_final.nlargest(10, 'environment_score')
for idx, row in top10.iterrows():
    print(f"  {row['OBJECT_NAME'][:40]:40s} | Score: {row['environment_score']:5.1f} | "
          f"Alt: {row['ALTITUDE_MEAN_KM']:6.1f}km | Source: {row['data_source']}")

print("\n--- DATA COVERAGE ---")
print(f"Satellites with actual operational data: {(env_scores_final['data_source'] == 'ACTUAL_PERIOD').sum():,}")
print(f"Satellites using baseline/recent data: {(env_scores_final['data_source'] != 'ACTUAL_PERIOD').sum():,}")

# ============================================================================
# STEP 3.7: Validation Checks
# ============================================================================
print("\n[7/7] Running validation checks...")

try:
    # Check 1: All LEO satellites have scores
    assert len(env_scores_final) == len(leo_sats), \
        f"ERROR: Mismatch! LEO sats: {len(leo_sats)}, Env scores: {len(env_scores_final)}"

    # Check 2: Scores in valid range
    assert env_scores_final['environment_score'].min() >= 0, "ERROR: Negative scores!"
    assert env_scores_final['environment_score'].max() <= 100, "ERROR: Scores > 100!"

    # Check 3: No nulls in environment_score
    assert env_scores_final['environment_score'].notna().all(), "ERROR: Null environment scores!"

    # Check 4: Satellite IDs are unique
    assert env_scores_final['satellite_id'].nunique() == len(env_scores_final), \
        "ERROR: Duplicate satellite IDs!"

    # Check 5: Altitude factors are reasonable
    assert env_scores_final['altitude_adjustment_factor'].between(0.5, 1.5).all(), \
        "ERROR: Altitude adjustment factors out of reasonable range!"

    print("\n✅ All validation checks passed!")

except AssertionError as e:
    print(f"\n❌ VALIDATION FAILED: {e}")
    raise

print("\n" + "="*70)
print("STEP 3 COMPLETE")
print("="*70)
