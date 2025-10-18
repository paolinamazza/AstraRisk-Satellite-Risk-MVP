"""
STEP 5: INTEGRATION & TOTAL RISK CALCULATION
=============================================
Merge all three risk driver scores (Collision, Environment, BORS),
calculate Total Risk Score, compute insurance premiums, analyze risk
contributions, and perform peer benchmarking.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 5: INTEGRATION & TOTAL RISK CALCULATION")
print("="*80)

# ============================================================================
# STEP 5.1: Load All Data
# ============================================================================
print("\n[1/12] Loading all risk score datasets...")

# Load all risk scores
leo_sats = pd.read_csv('leo_satellites_enriched.csv')
collision_scores = pd.read_csv('collision_scores_leo.csv')
environment_scores = pd.read_csv('environment_scores_leo.csv')
bors_scores = pd.read_csv('bors_scores_leo.csv')

print(f"Loaded data:")
print(f"  LEO satellites: {len(leo_sats):,}")
print(f"  Collision scores: {len(collision_scores):,}")
print(f"  Environment scores: {len(environment_scores):,}")
print(f"  BORS scores: {len(bors_scores):,}")

# Deduplicate LEO satellites (same as previous steps)
if leo_sats.duplicated('NORAD_CAT_ID').any():
    n_dups = leo_sats.duplicated('NORAD_CAT_ID').sum()
    print(f"Warning: Found {n_dups} duplicate NORAD_CAT_IDs in base, deduplicating...")
    leo_sats = leo_sats.drop_duplicates('NORAD_CAT_ID', keep='first').reset_index(drop=True)

# ============================================================================
# STEP 5.2: Merge All Scores
# ============================================================================
print("\n[2/12] Merging all risk scores...")

# Start with LEO satellites base
risk_assessment = leo_sats.copy()

# Merge collision scores
risk_assessment = risk_assessment.merge(
    collision_scores[['satellite_id', 'collision_score', 'pc_annual',
                      'n_events_total', 'data_source', 'data_quality']],
    left_on='NORAD_CAT_ID',
    right_on='satellite_id',
    how='left',
    suffixes=('', '_collision')
)

# Merge environment scores
risk_assessment = risk_assessment.merge(
    environment_scores[['satellite_id', 'environment_score', 'env_score_p95_raw',
                        'env_score_mean', 'severe_events_per_year',
                        'altitude_adjustment_factor', 'data_source']],
    left_on='NORAD_CAT_ID',
    right_on='satellite_id',
    how='left',
    suffixes=('', '_env')
)

# Merge BORS
risk_assessment = risk_assessment.merge(
    bors_scores[['satellite_id', 'bors_score', 'age_factor', 'tri',
                 'operator_heritage_index', 'infant_mortality_flag']],
    left_on='NORAD_CAT_ID',
    right_on='satellite_id',
    how='left',
    suffixes=('', '_bors')
)

# Clean up duplicate satellite_id columns
cols_to_drop = [col for col in risk_assessment.columns if col.startswith('satellite_id')]
risk_assessment = risk_assessment.drop(columns=cols_to_drop)

print(f"\n✓ Merged all scores: {len(risk_assessment):,} satellites")

# Check for missing scores
print(f"\nMissing scores check:")
print(f"  Collision score nulls: {risk_assessment['collision_score'].isna().sum()}")
print(f"  Environment score nulls: {risk_assessment['environment_score'].isna().sum()}")
print(f"  BORS nulls: {risk_assessment['bors_score'].isna().sum()}")

# ============================================================================
# STEP 5.3: Calculate Total Risk Score
# ============================================================================
print("\n[3/12] Calculating Total Risk Score...")

# Define weights for total risk calculation
WEIGHT_COLLISION = 0.40
WEIGHT_ENVIRONMENT = 0.30
WEIGHT_BORS = 0.30

def calculate_total_risk_score(row):
    """
    Calculate weighted combination of three risk drivers

    Total Risk Score = 0.40 × Collision + 0.30 × Environment + 0.30 × BORS
    """
    collision = row['collision_score']
    environment = row['environment_score']
    bors = row['bors_score']

    # Handle any missing values (shouldn't happen but safe)
    if pd.isna(collision) or pd.isna(environment) or pd.isna(bors):
        return np.nan

    total = (
        WEIGHT_COLLISION * collision +
        WEIGHT_ENVIRONMENT * environment +
        WEIGHT_BORS * bors
    )

    return round(total, 1)

risk_assessment['total_risk_score'] = risk_assessment.apply(
    calculate_total_risk_score,
    axis=1
)

print(f"\n✓ Calculated Total Risk Scores")
print(f"\nTotal Risk Score distribution:")
print(risk_assessment['total_risk_score'].describe())

# ============================================================================
# STEP 5.4: Categorize Risk Levels
# ============================================================================
print("\n[4/12] Categorizing risk levels...")

def categorize_risk(score):
    """
    Categorize total risk score into bins
    """
    if pd.isna(score):
        return 'Unknown'
    elif score < 30:
        return 'Low'
    elif score < 50:
        return 'Medium-Low'
    elif score < 70:
        return 'Medium-High'
    elif score < 85:
        return 'High'
    else:
        return 'Critical'

risk_assessment['risk_category'] = risk_assessment['total_risk_score'].apply(categorize_risk)

print("\nRisk category distribution:")
print(risk_assessment['risk_category'].value_counts().sort_index())

# ============================================================================
# STEP 5.5: Identify Dominant Risk Driver
# ============================================================================
print("\n[5/12] Identifying dominant risk driver...")

def identify_dominant_driver(row):
    """
    Identify which risk driver contributes most to total risk
    """
    collision_contrib = WEIGHT_COLLISION * row['collision_score']
    env_contrib = WEIGHT_ENVIRONMENT * row['environment_score']
    bors_contrib = WEIGHT_BORS * row['bors_score']

    max_contrib = max(collision_contrib, env_contrib, bors_contrib)

    if collision_contrib == max_contrib:
        return 'Collision'
    elif env_contrib == max_contrib:
        return 'Environment'
    else:
        return 'BORS'

risk_assessment['dominant_driver'] = risk_assessment.apply(
    identify_dominant_driver,
    axis=1
)

print("\nDominant driver distribution:")
print(risk_assessment['dominant_driver'].value_counts())

# ============================================================================
# STEP 5.6: Calculate Risk Contributions (Percentage)
# ============================================================================
print("\n[6/12] Calculating risk contributions (%)...")

def calculate_contributions(row):
    """
    Calculate percentage contribution of each driver to total risk
    """
    total = row['total_risk_score']

    if pd.isna(total) or total == 0:
        return 0, 0, 0

    collision_contrib = (WEIGHT_COLLISION * row['collision_score'] / total) * 100
    env_contrib = (WEIGHT_ENVIRONMENT * row['environment_score'] / total) * 100
    bors_contrib = (WEIGHT_BORS * row['bors_score'] / total) * 100

    return (round(collision_contrib, 1),
            round(env_contrib, 1),
            round(bors_contrib, 1))

risk_assessment[['collision_contribution_pct',
                 'environment_contribution_pct',
                 'bors_contribution_pct']] = risk_assessment.apply(
    lambda row: pd.Series(calculate_contributions(row)),
    axis=1
)

print("✓ Calculated risk contributions")

# ============================================================================
# STEP 5.7: Calculate Insurance Premiums
# ============================================================================
print("\n[7/12] Calculating insurance premiums...")

def calculate_insurance_premium(total_risk_score, sum_insured):
    """
    Convert total risk score to insurance premium

    Uses empirical mapping from risk score to annual loss probability
    Then applies load factor for expenses, profit, and buffer
    """

    # Step 1: Map score to annual loss probability (empirical)
    if total_risk_score < 30:
        p_loss_annual = 0.005  # 0.5%
    elif total_risk_score < 50:
        p_loss_annual = 0.015  # 1.5%
    elif total_risk_score < 70:
        p_loss_annual = 0.035  # 3.5%
    elif total_risk_score < 85:
        p_loss_annual = 0.070  # 7.0%
    else:
        p_loss_annual = 0.150  # 15.0% (nearly uninsurable)

    # Step 2: Calculate base premium (expected loss)
    base_premium = sum_insured * p_loss_annual

    # Step 3: Apply load factor
    expense_ratio = 0.30  # 30% for operational costs
    profit_margin = 0.15  # 15% profit target
    uncertainty_buffer = 0.10  # 10% for model uncertainty

    load_factor = 1 + expense_ratio + profit_margin + uncertainty_buffer
    # Total load factor = 1.55

    # Step 4: Final premium
    premium_annual = base_premium * load_factor

    # Step 5: Rate on line (% of sum insured)
    rate_on_line_pct = (premium_annual / sum_insured) * 100 if sum_insured > 0 else 0

    return {
        'premium_annual': round(premium_annual, 0),
        'rate_on_line_pct': round(rate_on_line_pct, 2),
        'p_loss_annual': p_loss_annual,
        'expected_annual_loss': round(base_premium, 0)
    }

# Assume sum insured = replacement value
# Estimate based on mass: €50k per kg for small sats, €30k per kg for large
def estimate_sum_insured(mass_kg):
    """
    Estimate satellite replacement value
    """
    if pd.isna(mass_kg) or mass_kg == 0:
        return 50_000_000  # Default €50M

    if mass_kg < 100:
        # CubeSat/SmallSat: higher cost per kg
        return mass_kg * 50_000
    elif mass_kg < 500:
        return mass_kg * 40_000
    else:
        # LargeSat: economies of scale
        return mass_kg * 30_000

risk_assessment['sum_insured'] = risk_assessment['LAUNCH_MASS_KG'].apply(estimate_sum_insured)

# Calculate premiums
premium_results = risk_assessment.apply(
    lambda row: pd.Series(calculate_insurance_premium(
        row['total_risk_score'],
        row['sum_insured']
    )),
    axis=1
)

risk_assessment = pd.concat([risk_assessment, premium_results], axis=1)

print("\n✓ Calculated insurance premiums")
print(f"\nPremium statistics:")
print(risk_assessment['premium_annual'].describe())
print(f"\nTotal insured value: €{risk_assessment['sum_insured'].sum():,.0f}")
print(f"Total premiums: €{risk_assessment['premium_annual'].sum():,.0f}")
print(f"Average rate on line: {risk_assessment['rate_on_line_pct'].mean():.2f}%")

# ============================================================================
# STEP 5.8: Peer Group Benchmarking
# ============================================================================
print("\n[8/12] Performing peer group benchmarking...")

def benchmark_satellite(target_sat, all_sats):
    """
    Compare satellite to peer group

    Peers defined as satellites with:
    - Similar altitude (±150 km)
    - Similar age (±2 years)
    - Same orbit class
    """

    altitude = target_sat['ALTITUDE_MEAN_KM']
    age = target_sat['AGE_YEARS']
    orbit_class = target_sat['ORBIT_CLASS']

    # Find peers
    peers = all_sats[
        (all_sats['ALTITUDE_MEAN_KM'].between(altitude - 150, altitude + 150)) &
        (all_sats['AGE_YEARS'].between(age - 2, age + 2)) &
        (all_sats['ORBIT_CLASS'] == orbit_class)
    ]

    # Need at least 10 peers for meaningful comparison
    if len(peers) < 10:
        # Relax criteria: use orbit class only
        peers = all_sats[all_sats['ORBIT_CLASS'] == orbit_class]

    if len(peers) < 5:
        # Not enough peers, return None
        return {
            'peer_group_size': 0,
            'peer_avg_risk': None,
            'peer_median_risk': None,
            'percentile_ranking': None
        }

    # Calculate statistics
    peer_avg = peers['total_risk_score'].mean()
    peer_median = peers['total_risk_score'].median()

    # Calculate percentile
    percentile = stats.percentileofscore(peers['total_risk_score'], target_sat['total_risk_score'])

    return {
        'peer_group_size': int(len(peers)),
        'peer_avg_risk': round(peer_avg, 1),
        'peer_median_risk': round(peer_median, 1),
        'percentile_ranking': round(percentile, 1)
    }

print("Calculating peer benchmarks (this may take a moment)...")

benchmark_results = risk_assessment.apply(
    lambda row: pd.Series(benchmark_satellite(row, risk_assessment)),
    axis=1
)

risk_assessment = pd.concat([risk_assessment, benchmark_results], axis=1)

print("✓ Completed peer benchmarking")

# ============================================================================
# STEP 5.9: Data Quality Score
# ============================================================================
print("\n[9/12] Calculating data quality scores...")

def calculate_overall_data_quality(row):
    """
    Calculate overall data quality score (0-100)

    Considers quality of all three drivers
    """
    # Collision data quality
    if row.get('data_source') == 'CDM_ACTUAL':
        collision_quality = 100
        if row.get('data_quality') == 'LIMITED':
            collision_quality = 70
        elif row.get('data_quality') == 'FAIR':
            collision_quality = 85
    elif 'IMPUTED' in str(row.get('data_source', '')):
        collision_quality = 50
    else:
        collision_quality = 30

    # Environment data quality
    if row.get('data_source_env') == 'ACTUAL_PERIOD':
        env_quality = 100
    elif 'PARTIAL' in str(row.get('data_source_env', '')):
        env_quality = 70
    else:
        env_quality = 60

    # BORS data quality (based on estimated fields)
    bors_quality = row.get('DATA_COMPLETENESS_SCORE', 70)

    # Weighted average
    overall_quality = (
        0.40 * collision_quality +
        0.30 * env_quality +
        0.30 * bors_quality
    )

    return round(overall_quality, 0)

risk_assessment['overall_data_quality'] = risk_assessment.apply(
    calculate_overall_data_quality,
    axis=1
)

print(f"\n✓ Calculated overall data quality scores")
print(f"Mean data quality: {risk_assessment['overall_data_quality'].mean():.1f}/100")

# ============================================================================
# STEP 5.10: Save Final Output
# ============================================================================
print("\n[10/12] Saving final risk assessment...")

# Select key columns for final output
output_columns = [
    # Identification
    'NORAD_CAT_ID', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE', 'OPS_STATUS_CODE',

    # Orbital parameters
    'ALTITUDE_MEAN_KM', 'INCLINATION', 'APOGEE', 'PERIGEE', 'ORBIT_CLASS',

    # Satellite characteristics
    'LAUNCH_DATE', 'LAUNCH_YEAR', 'AGE_YEARS', 'LAUNCH_MASS_KG',
    'EXPECTED_LIFETIME_YEARS', 'PURPOSE', 'CONTRACTOR_TIER', 'OPERATOR_TIER',

    # Risk scores (main)
    'collision_score', 'environment_score', 'bors_score', 'total_risk_score',

    # Risk analysis
    'risk_category', 'dominant_driver',
    'collision_contribution_pct', 'environment_contribution_pct', 'bors_contribution_pct',

    # Collision details
    'pc_annual', 'n_events_total',

    # Environment details
    'env_score_p95_raw', 'severe_events_per_year', 'altitude_adjustment_factor',

    # BORS details
    'age_factor', 'tri', 'operator_heritage_index', 'infant_mortality_flag',

    # Financial
    'sum_insured', 'premium_annual', 'rate_on_line_pct', 'p_loss_annual', 'expected_annual_loss',

    # Benchmarking
    'peer_group_size', 'peer_avg_risk', 'peer_median_risk', 'percentile_ranking',

    # Data quality
    'overall_data_quality', 'DATA_COMPLETENESS_SCORE'
]

# Select columns that exist
final_columns = [col for col in output_columns if col in risk_assessment.columns]
final_risk_assessment = risk_assessment[final_columns].copy()

# Sort by total risk score (descending)
final_risk_assessment = final_risk_assessment.sort_values(
    'total_risk_score',
    ascending=False
).reset_index(drop=True)

# Save to CSV
final_risk_assessment.to_csv('final_risk_assessment_leo.csv', index=False)

print(f"\n✅ Saved final risk assessment: {len(final_risk_assessment):,} satellites")
print(f"   Output file: final_risk_assessment_leo.csv")

# ============================================================================
# STEP 5.11: Print Comprehensive Summary
# ============================================================================
print("\n[11/12] Generating comprehensive summary...")
print("\n" + "="*80)
print("STEP 5 SUMMARY - FINAL RISK ASSESSMENT")
print("="*80)

print(f"\n{'OVERVIEW':=^80}")
print(f"Total LEO satellites analyzed: {len(final_risk_assessment):,}")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n{'TOTAL RISK SCORE DISTRIBUTION':=^80}")
print(final_risk_assessment['total_risk_score'].describe())

print(f"\n{'RISK CATEGORIES':=^80}")
for category in ['Low', 'Medium-Low', 'Medium-High', 'High', 'Critical']:
    count = (final_risk_assessment['risk_category'] == category).sum()
    pct = count / len(final_risk_assessment) * 100
    print(f"  {category:15s}: {count:5,} satellites ({pct:5.1f}%)")

print(f"\n{'DOMINANT RISK DRIVERS':=^80}")
for driver in ['Collision', 'Environment', 'BORS']:
    count = (final_risk_assessment['dominant_driver'] == driver).sum()
    pct = count / len(final_risk_assessment) * 100
    print(f"  {driver:15s}: {count:5,} satellites ({pct:5.1f}%)")

print(f"\n{'AVERAGE RISK SCORES BY DRIVER':=^80}")
print(f"  Collision:    {final_risk_assessment['collision_score'].mean():5.1f}")
print(f"  Environment:  {final_risk_assessment['environment_score'].mean():5.1f}")
print(f"  BORS:         {final_risk_assessment['bors_score'].mean():5.1f}")
print(f"  Total Risk:   {final_risk_assessment['total_risk_score'].mean():5.1f}")

print(f"\n{'FINANCIAL METRICS':=^80}")
total_insured = final_risk_assessment['sum_insured'].sum()
total_premium = final_risk_assessment['premium_annual'].sum()
total_expected_loss = final_risk_assessment['expected_annual_loss'].sum()

print(f"  Total Insured Value:     €{total_insured:15,.0f}")
print(f"  Total Annual Premiums:   €{total_premium:15,.0f}")
print(f"  Total Expected Loss:     €{total_expected_loss:15,.0f}")
print(f"  Average Rate on Line:    {final_risk_assessment['rate_on_line_pct'].mean():15.2f}%")
print(f"  Loss Ratio (EL/Premium): {(total_expected_loss/total_premium)*100:15.1f}%")

print(f"\n{'TOP 20 HIGHEST RISK SATELLITES':=^80}")
top20 = final_risk_assessment.head(20)
print(f"{'Rank':<5} {'Name':<35} {'Total':>6} {'Coll':>5} {'Env':>5} {'BORS':>5} {'Category':<13} {'Driver':<11}")
print("-" * 95)
for idx, row in top20.iterrows():
    print(f"{idx+1:<5} {row['OBJECT_NAME'][:34]:<35} "
          f"{row['total_risk_score']:>6.1f} "
          f"{row['collision_score']:>5.1f} "
          f"{row['environment_score']:>5.1f} "
          f"{row['bors_score']:>5.1f} "
          f"{row['risk_category']:<13} "
          f"{row['dominant_driver']:<11}")

print(f"\n{'DATA QUALITY METRICS':=^80}")
print(f"  Mean overall data quality:  {final_risk_assessment['overall_data_quality'].mean():.1f}/100")
print(f"  High quality (>80):         {(final_risk_assessment['overall_data_quality'] > 80).sum():,} satellites")
print(f"  Medium quality (60-80):     {((final_risk_assessment['overall_data_quality'] >= 60) & (final_risk_assessment['overall_data_quality'] <= 80)).sum():,} satellites")
print(f"  Low quality (<60):          {(final_risk_assessment['overall_data_quality'] < 60).sum():,} satellites")

print(f"\n{'CORRELATION ANALYSIS':=^80}")
corr_matrix = final_risk_assessment[['collision_score', 'environment_score', 'bors_score', 'total_risk_score']].corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# ============================================================================
# STEP 5.12: Validation Checks
# ============================================================================
print("\n[12/12] Running validation checks...")

try:
    # Check 1: All satellites have total risk scores
    assert final_risk_assessment['total_risk_score'].notna().all(), "ERROR: Null total risk scores!"

    # Check 2: Total risk score in valid range
    assert final_risk_assessment['total_risk_score'].between(0, 100).all(), "ERROR: Total risk out of range!"

    # Check 3: Contributions sum to ~100%
    contribution_sum = (final_risk_assessment['collision_contribution_pct'] +
                       final_risk_assessment['environment_contribution_pct'] +
                       final_risk_assessment['bors_contribution_pct'])
    assert contribution_sum.between(99, 101).all(), "ERROR: Contributions don't sum to 100%!"

    # Check 4: Premium calculations are reasonable
    assert final_risk_assessment['premium_annual'].min() > 0, "ERROR: Negative premiums!"
    assert final_risk_assessment['rate_on_line_pct'].max() < 30, "ERROR: Unreasonably high rates!"

    # Check 5: Sanity check - high risk should have high premiums
    high_risk = final_risk_assessment[final_risk_assessment['total_risk_score'] > 80]
    low_risk = final_risk_assessment[final_risk_assessment['total_risk_score'] < 30]
    if len(high_risk) > 0 and len(low_risk) > 0:
        assert high_risk['rate_on_line_pct'].mean() > low_risk['rate_on_line_pct'].mean(), \
            "ERROR: High risk satellites don't have higher premiums!"

    print("\n✅ All validation checks passed!")

except AssertionError as e:
    print(f"\n❌ VALIDATION FAILED: {e}")
    raise

print("\n" + "="*80)
print("✅ STEP 5 COMPLETE - Final Risk Assessment Ready")
print("="*80)
