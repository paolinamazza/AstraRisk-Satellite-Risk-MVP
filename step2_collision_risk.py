"""
STEP 2: COLLISION RISK SCORE CALCULATION
=========================================
Calculate collision risk scores (0-100) for all LEO satellites based on
Conjunction Data Messages (CDM). Impute scores for satellites without CDM data
using peer group analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 2: COLLISION RISK SCORE CALCULATION")
print("="*70)

# ============================================================================
# STEP 2.1: Load Data
# ============================================================================
print("\n[1/10] Loading data...")

# Load LEO satellites
leo_sats = pd.read_csv('leo_satellites_enriched.csv')

# Deduplicate by NORAD_CAT_ID (keep first occurrence)
if leo_sats.duplicated('NORAD_CAT_ID').any():
    n_dups = leo_sats.duplicated('NORAD_CAT_ID').sum()
    print(f"Warning: Found {n_dups} duplicate NORAD_CAT_IDs, keeping first occurrence")
    leo_sats = leo_sats.drop_duplicates('NORAD_CAT_ID', keep='first').reset_index(drop=True)

print(f"Loaded {len(leo_sats):,} LEO satellites")

# Load CDM data
cdm_data = pd.read_csv('spacetrack_conjunctions_all_20251007.csv')
print(f"Loaded {len(cdm_data):,} CDM events")

# Convert dates
cdm_data['TCA'] = pd.to_datetime(cdm_data['TCA'], errors='coerce')
cdm_data['CREATED'] = pd.to_datetime(cdm_data['CREATED'], errors='coerce')

# ============================================================================
# STEP 2.2: Calculate Data Period
# ============================================================================
print("\n[2/10] Calculating CDM data period...")

min_date = cdm_data['TCA'].min()
max_date = cdm_data['TCA'].max()
data_period_days = (max_date - min_date).days

print(f"\nCDM data period:")
print(f"  Start: {min_date.date()}")
print(f"  End: {max_date.date()}")
print(f"  Duration: {data_period_days} days")

# ============================================================================
# STEP 2.3: Filter CDM for Valid Events
# ============================================================================
print("\n[3/10] Filtering CDM for valid events...")

# Keep only events with valid PC (probability of collision)
cdm_valid = cdm_data[cdm_data['PC'].notna() & (cdm_data['PC'] > 0)].copy()

print(f"\nCDM events with valid PC: {len(cdm_valid):,} ({len(cdm_valid)/len(cdm_data)*100:.1f}%)")

# ============================================================================
# STEP 2.4: Calculate Collision Score for Satellites with CDM Data
# ============================================================================
print("\n[4/10] Calculating collision scores from actual CDM data...")

def calculate_collision_score_from_cdm(satellite_id, cdm_data, data_period_days):
    """
    Calculate collision score for a satellite with CDM data

    Args:
        satellite_id: NORAD_CAT_ID
        cdm_data: DataFrame with CDM events
        data_period_days: Number of days in dataset

    Returns:
        dict with collision metrics
    """

    # Filter events for this satellite (can be SAT_1 or SAT_2)
    sat_events = cdm_data[
        (cdm_data['SAT_1_ID'] == satellite_id) |
        (cdm_data['SAT_2_ID'] == satellite_id)
    ]

    if len(sat_events) == 0:
        return None

    # Calculate PC_annual (aggregate probability)
    # Sum of all PCs (valid for small probabilities, events assumed independent)
    pc_period = sat_events['PC'].sum()

    # Extrapolate to annual (365 days)
    if data_period_days > 0:
        pc_annual = pc_period * (365.0 / data_period_days)
    else:
        pc_annual = pc_period

    # Convert PC_annual to score 0-100 (logarithmic mapping)
    if pc_annual < 1e-4:  # < 0.01%
        collision_score = 0.0
    elif pc_annual >= 0.05:  # >= 5%
        collision_score = 100.0
    else:
        # Logarithmic scale: score = (log(pc) - log(min)) / (log(max) - log(min)) * 100
        log_pc = np.log10(pc_annual)
        log_min = np.log10(1e-4)  # -4
        log_max = np.log10(0.05)   # -1.301
        collision_score = ((log_pc - log_min) / (log_max - log_min)) * 100.0
        collision_score = max(0.0, min(100.0, collision_score))

    # Calculate breakdown by risk level
    n_high_risk = len(sat_events[sat_events['PC'] >= 0.01])      # >= 1%
    n_medium_risk = len(sat_events[(sat_events['PC'] >= 0.001) &
                                    (sat_events['PC'] < 0.01)])   # 0.1% - 1%
    n_low_risk = len(sat_events[sat_events['PC'] < 0.001])       # < 0.1%

    # Other metrics
    n_emergency = len(sat_events[sat_events['EMERGENCY_REPORTABLE'] == 'Y'])

    miss_distances = sat_events['MIN_RNG'].dropna()
    avg_miss_distance = miss_distances.mean() if len(miss_distances) > 0 else None
    min_miss_distance = miss_distances.min() if len(miss_distances) > 0 else None

    max_pc_event = sat_events['PC'].max()

    events_per_month = len(sat_events) * (365.0 / data_period_days) / 12.0

    # Data quality assessment
    if len(sat_events) >= 10:
        data_quality = 'GOOD'
    elif len(sat_events) >= 5:
        data_quality = 'FAIR'
    else:
        data_quality = 'LIMITED'

    return {
        'satellite_id': int(satellite_id),
        'collision_score': round(collision_score, 1),
        'pc_annual': pc_annual,
        'pc_annual_pct': f"{pc_annual*100:.3f}%",
        'n_events_total': len(sat_events),
        'n_high_risk': int(n_high_risk),
        'n_medium_risk': int(n_medium_risk),
        'n_low_risk': int(n_low_risk),
        'n_emergency': int(n_emergency),
        'avg_miss_distance_m': round(avg_miss_distance, 0) if avg_miss_distance else None,
        'min_miss_distance_m': round(min_miss_distance, 0) if min_miss_distance else None,
        'max_pc_event': float(max_pc_event),
        'max_pc_event_pct': f"{max_pc_event*100:.4f}%",
        'events_per_month_estimated': round(events_per_month, 1),
        'data_quality': data_quality,
        'data_source': 'CDM_ACTUAL',
        'extrapolated': True,
        'data_period_days': int(data_period_days)
    }

# Get unique satellite IDs from CDM
sat_ids_1 = cdm_valid['SAT_1_ID'].dropna().unique()
sat_ids_2 = cdm_valid['SAT_2_ID'].dropna().unique()
all_cdm_sat_ids = sorted(set(list(sat_ids_1) + list(sat_ids_2)))

print(f"\nCalculating collision scores for {len(all_cdm_sat_ids):,} satellites with CDM data...")

# Calculate scores
collision_results = []
for i, sat_id in enumerate(all_cdm_sat_ids):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1:,}/{len(all_cdm_sat_ids):,} satellites...")

    result = calculate_collision_score_from_cdm(sat_id, cdm_valid, data_period_days)
    if result:
        collision_results.append(result)

collision_scores_df = pd.DataFrame(collision_results)

print(f"\n✓ Calculated scores for {len(collision_scores_df):,} satellites")
print(f"\nScore distribution:")
print(collision_scores_df['collision_score'].describe())

# ============================================================================
# STEP 2.5: Filter for LEO Satellites Only
# ============================================================================
print("\n[5/10] Filtering for LEO satellites...")

# Keep only satellites that are in our LEO enriched dataset
leo_sat_ids = set(leo_sats['NORAD_CAT_ID'].values)
collision_scores_leo_actual = collision_scores_df[
    collision_scores_df['satellite_id'].isin(leo_sat_ids)
].copy()

print(f"\nLEO satellites with actual CDM data: {len(collision_scores_leo_actual):,}")
print(f"LEO satellites without CDM data: {len(leo_sats) - len(collision_scores_leo_actual):,}")

# ============================================================================
# STEP 2.6: Impute Missing Collision Scores from Peer Groups
# ============================================================================
print("\n[6/10] Imputing missing scores from peer groups...")

def find_peer_group(target_sat, all_sats, min_peers=5):
    """
    Find similar satellites based on orbital characteristics

    Similarity criteria:
    - Altitude ±100 km
    - Inclination ±10 degrees
    - Same orbit class
    """

    altitude = target_sat['ALTITUDE_MEAN_KM']
    inclination = target_sat['INCLINATION']
    orbit_class = target_sat['ORBIT_CLASS']

    # Find peers with similar characteristics
    peers = all_sats[
        (all_sats['ALTITUDE_MEAN_KM'].between(altitude - 100, altitude + 100)) &
        (all_sats['INCLINATION'].between(inclination - 10, inclination + 10)) &
        (all_sats['ORBIT_CLASS'] == orbit_class)
    ]

    # If not enough peers, relax criteria
    if len(peers) < min_peers:
        peers = all_sats[
            (all_sats['ALTITUDE_MEAN_KM'].between(altitude - 200, altitude + 200)) &
            (all_sats['ORBIT_CLASS'] == orbit_class)
        ]

    # If still not enough, use orbit class only
    if len(peers) < min_peers:
        peers = all_sats[all_sats['ORBIT_CLASS'] == orbit_class]

    return peers

def impute_collision_score(target_sat, leo_sats_with_scores, all_leo_sats):
    """
    Impute collision score from peer group
    """

    # Find peer group
    peers = find_peer_group(target_sat, all_leo_sats)

    # Get peers that have collision scores
    peer_ids = set(peers['NORAD_CAT_ID'].values)
    peers_with_scores = leo_sats_with_scores[
        leo_sats_with_scores['satellite_id'].isin(peer_ids)
    ]

    if len(peers_with_scores) >= 5:
        # Use median of peer group (robust to outliers)
        imputed_score = peers_with_scores['collision_score'].median()
        confidence = 'MEDIUM'
        n_peers = len(peers_with_scores)
    elif len(peers_with_scores) > 0:
        # Few peers, use mean but lower confidence
        imputed_score = peers_with_scores['collision_score'].mean()
        confidence = 'LOW'
        n_peers = len(peers_with_scores)
    else:
        # No peers with scores, use baseline for LEO
        # LEO baseline varies by altitude
        altitude = target_sat['ALTITUDE_MEAN_KM']
        if altitude < 400:
            imputed_score = 45  # Very low LEO, high congestion
        elif altitude < 600:
            imputed_score = 35  # Typical LEO, moderate congestion
        elif altitude < 1000:
            imputed_score = 25  # High LEO, lower congestion
        else:
            imputed_score = 15  # Very high LEO, sparse
        confidence = 'BASELINE'
        n_peers = 0

    return {
        'satellite_id': int(target_sat['NORAD_CAT_ID']),
        'collision_score': round(imputed_score, 1),
        'pc_annual': None,
        'pc_annual_pct': None,
        'n_events_total': 0,
        'n_high_risk': 0,
        'n_medium_risk': 0,
        'n_low_risk': 0,
        'n_emergency': 0,
        'avg_miss_distance_m': None,
        'min_miss_distance_m': None,
        'max_pc_event': None,
        'max_pc_event_pct': None,
        'events_per_month_estimated': None,
        'data_quality': None,
        'data_source': f'IMPUTED_PEER_{n_peers}',
        'imputation_confidence': confidence,
        'n_peers_used': n_peers,
        'extrapolated': False,
        'data_period_days': 0
    }

# Find satellites without CDM data
sats_with_cdm = set(collision_scores_leo_actual['satellite_id'].values)
sats_without_cdm = leo_sats[~leo_sats['NORAD_CAT_ID'].isin(sats_with_cdm)]

print(f"\nImputing collision scores for {len(sats_without_cdm):,} satellites...")

imputed_results = []
for idx, (_, sat) in enumerate(sats_without_cdm.iterrows()):
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx+1:,}/{len(sats_without_cdm):,} satellites...")

    imputed = impute_collision_score(sat, collision_scores_leo_actual, leo_sats)
    imputed_results.append(imputed)

collision_scores_imputed = pd.DataFrame(imputed_results)

print(f"\n✓ Imputed scores for {len(collision_scores_imputed):,} satellites")
print(f"\nImputation confidence distribution:")
print(collision_scores_imputed['imputation_confidence'].value_counts())

# ============================================================================
# STEP 2.7: Combine Actual and Imputed Scores
# ============================================================================
print("\n[7/10] Combining actual and imputed scores...")

# Combine both datasets
collision_scores_combined = pd.concat([
    collision_scores_leo_actual,
    collision_scores_imputed
], ignore_index=True)

# Sort by satellite_id
collision_scores_combined = collision_scores_combined.sort_values('satellite_id').reset_index(drop=True)

print(f"\nTotal collision scores: {len(collision_scores_combined):,}")
print(f"  Actual (from CDM): {len(collision_scores_leo_actual):,} ({len(collision_scores_leo_actual)/len(collision_scores_combined)*100:.1f}%)")
print(f"  Imputed (from peers): {len(collision_scores_imputed):,} ({len(collision_scores_imputed)/len(collision_scores_combined)*100:.1f}%)")

# ============================================================================
# STEP 2.8: Add Satellite Metadata
# ============================================================================
print("\n[8/10] Adding satellite metadata...")

# Merge with LEO satellites to add metadata
# Use inner join to ensure we only keep LEO satellites
collision_scores_full = collision_scores_combined.merge(
    leo_sats[['NORAD_CAT_ID', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
              'ALTITUDE_MEAN_KM', 'INCLINATION', 'ORBIT_CLASS']],
    left_on='satellite_id',
    right_on='NORAD_CAT_ID',
    how='inner'  # Only keep matches
)

# Drop duplicate NORAD_CAT_ID column
collision_scores_full = collision_scores_full.drop('NORAD_CAT_ID', axis=1)

print(f"✓ Added metadata for {len(collision_scores_full):,} satellites")

# ============================================================================
# STEP 2.9: Save Output
# ============================================================================
print("\n[9/10] Saving collision scores...")

# Reorder columns
output_cols = [
    'satellite_id', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
    'ALTITUDE_MEAN_KM', 'INCLINATION', 'ORBIT_CLASS',
    'collision_score', 'pc_annual', 'pc_annual_pct',
    'n_events_total', 'n_high_risk', 'n_medium_risk', 'n_low_risk',
    'n_emergency', 'avg_miss_distance_m', 'min_miss_distance_m',
    'max_pc_event', 'max_pc_event_pct', 'events_per_month_estimated',
    'data_quality', 'data_source', 'extrapolated', 'data_period_days'
]

# Add imputation columns if they exist
if 'imputation_confidence' in collision_scores_full.columns:
    output_cols.extend(['imputation_confidence', 'n_peers_used'])

collision_scores_final = collision_scores_full[output_cols].copy()

# Save to CSV
collision_scores_final.to_csv('collision_scores_leo.csv', index=False)

print(f"\n✅ Saved {len(collision_scores_final):,} collision scores to collision_scores_leo.csv")

# ============================================================================
# STEP 2.10: Print Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("STEP 2 SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal LEO satellites with collision scores: {len(collision_scores_final):,}")

print("\n--- COLLISION SCORE DISTRIBUTION ---")
print(collision_scores_final['collision_score'].describe())

print("\n--- SCORE BY DATA SOURCE ---")
for source_type in ['CDM_ACTUAL', 'IMPUTED_PEER']:
    subset = collision_scores_final[collision_scores_final['data_source'].str.contains(source_type, na=False)]
    if len(subset) > 0:
        print(f"\n{source_type} (n={len(subset):,}):")
        print(f"  Mean score: {subset['collision_score'].mean():.1f}")
        print(f"  Median score: {subset['collision_score'].median():.1f}")
        print(f"  Min: {subset['collision_score'].min():.1f}")
        print(f"  Max: {subset['collision_score'].max():.1f}")

print("\n--- RISK CATEGORIES ---")
risk_categories = {
    'Very Low (0-20)': (collision_scores_final['collision_score'] < 20).sum(),
    'Low (20-40)': ((collision_scores_final['collision_score'] >= 20) &
                    (collision_scores_final['collision_score'] < 40)).sum(),
    'Medium (40-60)': ((collision_scores_final['collision_score'] >= 40) &
                       (collision_scores_final['collision_score'] < 60)).sum(),
    'High (60-80)': ((collision_scores_final['collision_score'] >= 60) &
                     (collision_scores_final['collision_score'] < 80)).sum(),
    'Very High (80-100)': (collision_scores_final['collision_score'] >= 80).sum()
}

for category, count in risk_categories.items():
    pct = count / len(collision_scores_final) * 100
    print(f"  {category}: {count:,} satellites ({pct:.1f}%)")

print("\n--- TOP 10 HIGHEST RISK SATELLITES ---")
top10 = collision_scores_final.nlargest(10, 'collision_score')
for idx, row in top10.iterrows():
    print(f"  {row['OBJECT_NAME'][:40]:40s} | Score: {row['collision_score']:5.1f} | "
          f"Alt: {row['ALTITUDE_MEAN_KM']:6.1f}km | Source: {row['data_source']}")

if 'pc_annual' in collision_scores_final.columns:
    actual_only = collision_scores_final[collision_scores_final['data_source'] == 'CDM_ACTUAL']
    if len(actual_only) > 0:
        print("\n--- PC_ANNUAL STATISTICS (CDM Actual Only) ---")
        pc_stats = actual_only['pc_annual'].dropna()
        if len(pc_stats) > 0:
            print(f"  Mean: {pc_stats.mean()*100:.4f}%")
            print(f"  Median: {pc_stats.median()*100:.4f}%")
            print(f"  Max: {pc_stats.max()*100:.3f}%")

# ============================================================================
# VALIDATION CHECKS
# ============================================================================
print("\n[10/10] Running validation checks...")

try:
    # Check 1: All LEO satellites have scores
    assert len(collision_scores_final) == len(leo_sats), \
        f"ERROR: Mismatch in counts! LEO sats: {len(leo_sats)}, Scores: {len(collision_scores_final)}"

    # Check 2: Scores in valid range
    assert collision_scores_final['collision_score'].min() >= 0, "ERROR: Negative scores!"
    assert collision_scores_final['collision_score'].max() <= 100, "ERROR: Scores > 100!"

    # Check 3: No nulls in collision_score
    assert collision_scores_final['collision_score'].notna().all(), "ERROR: Null collision scores!"

    # Check 4: Satellite IDs are unique
    assert collision_scores_final['satellite_id'].nunique() == len(collision_scores_final), \
        "ERROR: Duplicate satellite IDs!"

    print("\n✅ All validation checks passed!")

except AssertionError as e:
    print(f"\n❌ VALIDATION FAILED: {e}")
    raise

print("\n" + "="*70)
print("STEP 2 COMPLETE")
print("="*70)
