"""
STEP 4: BORS (BASE/OPERATIONS RISK) CALCULATION
================================================
Calculate Base/Operations Risk Scores (0-100) for all LEO satellites based on
intrinsic reliability factors: satellite age, technical quality, operator
experience, and infant mortality risk.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 4: BORS (BASE/OPERATIONS RISK) CALCULATION")
print("="*70)

# ============================================================================
# STEP 4.1: Load Data and Extract Mappings from Reference
# ============================================================================
print("\n[1/7] Loading data and extracting BORS mappings...")

# Load LEO satellites
leo_sats = pd.read_csv('leo_satellites_enriched.csv')

# Deduplicate by NORAD_CAT_ID (same as steps 2 & 3)
if leo_sats.duplicated('NORAD_CAT_ID').any():
    n_dups = leo_sats.duplicated('NORAD_CAT_ID').sum()
    print(f"Warning: Found {n_dups} duplicate NORAD_CAT_IDs, keeping first occurrence")
    leo_sats = leo_sats.drop_duplicates('NORAD_CAT_ID', keep='first').reset_index(drop=True)

print(f"Loaded {len(leo_sats):,} LEO satellites")

# Load BORS reference dataset
bors_ref = pd.read_csv('bors_leo_operational_risk_dataset_v2.csv')
print(f"Loaded {len(bors_ref):,} reference satellites")

# Extract weights from reference dataset (these are constants)
w_age = bors_ref['w_age'].iloc[0]  # 0.30
w_tech = bors_ref['w_tech'].iloc[0]  # 0.40
w_ops = bors_ref['w_ops'].iloc[0]  # 0.15
w_err = bors_ref['w_err'].iloc[0]  # 0.15

w_CQI = bors_ref['w_CQI'].iloc[0]  # 0.25
w_HI = bors_ref['w_HI'].iloc[0]   # 0.40
w_MRI = bors_ref['w_MRI'].iloc[0]  # 0.20
w_SVI = bors_ref['w_SVI'].iloc[0]  # 0.15

print(f"\nBORS weights extracted:")
print(f"  Main: Age={w_age}, Tech={w_tech}, Ops={w_ops}, Err={w_err}")
print(f"  TRI sub-weights: CQI={w_CQI}, HI={w_HI}, MRI={w_MRI}, SVI={w_SVI}")

# Examine reference data
print("\n--- Reference Dataset Sample ---")
print(bors_ref[['Satellite_ID', 'Age_current_years', 'Expected_Lifetime_years',
                'Launch_Mass_kg', 'Purpose', 'Contractor_Tier', 'Operator_Tier',
                'AgeFactor', 'CQI', 'HI', 'MRI', 'SVI',
                'Operator_Heritage_Index', 'Infant_Mortality_Flag',
                'Operational_Risk_Score']].head())

# ============================================================================
# STEP 4.2: Define Mapping Functions
# ============================================================================
print("\n[2/7] Defining mapping functions...")

def calculate_age_factor(age_years, expected_lifetime_years):
    """
    AgeFactor = min(Age / Expected_Lifetime, 1.0)

    Returns value between 0 and 1:
    - 0 = brand new satellite
    - 1 = reached or exceeded design lifetime
    """
    if pd.isna(age_years) or pd.isna(expected_lifetime_years) or expected_lifetime_years == 0:
        return 0.5  # Default to middle if data missing

    age_factor = age_years / expected_lifetime_years
    return min(age_factor, 1.0)

def calculate_cqi(mass_kg):
    """
    Component Quality Index based on mass

    Small satellites tend to use COTS (commercial) components
    Large satellites use space-grade components

    Returns:
    - 0.7 for mass < 100 kg (high risk, COTS likely)
    - 0.5 for 100 <= mass < 500 kg
    - 0.3 for mass >= 500 kg (low risk, space-grade)
    """
    if pd.isna(mass_kg):
        return 0.5  # Default to medium

    if mass_kg < 100:
        return 0.7
    elif mass_kg < 500:
        return 0.5
    else:
        return 0.3

def calculate_hi(contractor_tier):
    """
    Heritage Index based on contractor tier

    Returns:
    - 0.2 for Tier 1 (experienced, low risk)
    - 0.5 for Tier 2 (moderate experience)
    - 0.8 for Tier 3 (new entrant, high risk)
    """
    tier_to_hi = {1: 0.2, 2: 0.5, 3: 0.8}
    return tier_to_hi.get(contractor_tier, 0.5)

def calculate_mri(mass_kg):
    """
    Mass Reliability Index - U-shaped curve

    Very small and very large satellites have higher risk
    Medium-sized satellites are in the "sweet spot"

    Returns:
    - 0.7 for mass < 500 kg (small, high infant mortality)
    - 0.3 for 500 <= mass <= 2500 kg (sweet spot, low risk)
    - 0.6 for mass > 2500 kg (large, system complexity risk)
    """
    if pd.isna(mass_kg):
        return 0.5

    if mass_kg < 500:
        return 0.7
    elif mass_kg <= 2500:
        return 0.3
    else:
        return 0.6

def calculate_svi(purpose):
    """
    Subsystem Vulnerability Index based on mission purpose

    Communications and Earth Observation have most vulnerable subsystems

    Returns:
    - 0.8 for Communications (power & comms subsystems critical)
    - 0.6 for Earth Observation (complex payload)
    - 0.4 for others (Navigation, Science, Tech Demo)
    """
    purpose_to_svi = {
        'Communications': 0.8,
        'Earth Observation': 0.6,
        'Navigation': 0.4,
        'Science': 0.4,
        'Technology Demo': 0.4,
        'Military': 0.5
    }
    return purpose_to_svi.get(purpose, 0.4)

def calculate_operator_heritage_index(operator_tier):
    """
    Operator Heritage Index based on operator tier

    Returns:
    - 0.1 for Tier 1 (government agencies, major operators - very low risk)
    - 0.4 for Tier 2 (established commercial)
    - 0.7 for Tier 3 (new operators, high risk)
    """
    tier_to_index = {1: 0.1, 2: 0.4, 3: 0.7}
    return tier_to_index.get(operator_tier, 0.4)

def calculate_infant_mortality_flag(age_years):
    """
    Infant Mortality Flag for first-year risk

    Returns:
    - 0.8 if age < 1 year (high risk period)
    - 0.1 otherwise (normal operational risk)
    """
    if pd.isna(age_years):
        return 0.1

    return 0.8 if age_years < 1.0 else 0.1

print("✓ Defined all mapping functions")

# ============================================================================
# STEP 4.3: Calculate BORS for All Satellites
# ============================================================================
print("\n[3/7] Calculating BORS for all satellites...")

def calculate_bors(satellite):
    """
    Calculate complete BORS for a satellite

    Formula:
    TRI = w_CQI·CQI + w_HI·HI + w_MRI·MRI + w_SVI·SVI
    raw_score = w_age·AgeFactor + w_tech·TRI + w_ops·Operator_Index + w_err·Infant_Flag
    BORS = 100 × raw_score (assuming raw_score naturally falls in 0-1 range)
    """

    # PILLAR 1: Age Factor
    age_factor = calculate_age_factor(
        satellite['AGE_YEARS'],
        satellite['EXPECTED_LIFETIME_YEARS']
    )

    # PILLAR 2: Technical Reliability Index (TRI)
    cqi = calculate_cqi(satellite['LAUNCH_MASS_KG'])
    hi = calculate_hi(satellite['CONTRACTOR_TIER'])
    mri = calculate_mri(satellite['LAUNCH_MASS_KG'])
    svi = calculate_svi(satellite['PURPOSE'])

    tri = w_CQI * cqi + w_HI * hi + w_MRI * mri + w_SVI * svi

    # PILLAR 3: Operator Heritage Index
    operator_index = calculate_operator_heritage_index(satellite['OPERATOR_TIER'])

    # PILLAR 4: Infant Mortality Flag
    infant_flag = calculate_infant_mortality_flag(satellite['AGE_YEARS'])

    # FINAL BORS CALCULATION
    raw_score = (
        w_age * age_factor +
        w_tech * tri +
        w_ops * operator_index +
        w_err * infant_flag
    )

    # Scale to 0-100
    bors_score = raw_score * 100

    # Clamp to valid range
    bors_score = max(0, min(100, bors_score))

    return {
        'satellite_id': int(satellite['NORAD_CAT_ID']),
        'bors_score': round(bors_score, 1),
        'age_factor': round(age_factor, 3),
        'cqi': round(cqi, 2),
        'hi': round(hi, 2),
        'mri': round(mri, 2),
        'svi': round(svi, 2),
        'tri': round(tri, 3),
        'operator_heritage_index': round(operator_index, 2),
        'infant_mortality_flag': round(infant_flag, 2),
        'age_years': round(satellite['AGE_YEARS'], 2),
        'expected_lifetime_years': satellite['EXPECTED_LIFETIME_YEARS'],
        'launch_mass_kg': satellite['LAUNCH_MASS_KG'],
        'contractor_tier': satellite['CONTRACTOR_TIER'],
        'operator_tier': satellite['OPERATOR_TIER'],
        'purpose': satellite['PURPOSE']
    }

# Calculate BORS for all satellites
print(f"\nCalculating BORS for {len(leo_sats):,} satellites...")

bors_results = []
for idx, (_, sat) in enumerate(leo_sats.iterrows()):
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx+1:,}/{len(leo_sats):,} satellites...")

    result = calculate_bors(sat)
    bors_results.append(result)

bors_scores_df = pd.DataFrame(bors_results)

print(f"\n✓ Calculated BORS for {len(bors_scores_df):,} satellites")

# ============================================================================
# STEP 4.4: Add Satellite Metadata
# ============================================================================
print("\n[4/7] Adding satellite metadata...")

# Merge with LEO satellites to add metadata
bors_scores_full = bors_scores_df.merge(
    leo_sats[['NORAD_CAT_ID', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
              'ALTITUDE_MEAN_KM', 'ORBIT_CLASS']],
    left_on='satellite_id',
    right_on='NORAD_CAT_ID',
    how='left'
)

# Drop duplicate column
bors_scores_full = bors_scores_full.drop('NORAD_CAT_ID', axis=1)

print(f"✓ Added metadata for {len(bors_scores_full):,} satellites")

# ============================================================================
# STEP 4.5: Save Output
# ============================================================================
print("\n[5/7] Saving BORS scores...")

# Reorder columns
output_cols = [
    'satellite_id', 'OBJECT_NAME', 'OWNER', 'OBJECT_TYPE',
    'ALTITUDE_MEAN_KM', 'ORBIT_CLASS',
    'bors_score',
    'age_factor', 'tri', 'operator_heritage_index', 'infant_mortality_flag',
    'cqi', 'hi', 'mri', 'svi',
    'age_years', 'expected_lifetime_years', 'launch_mass_kg',
    'contractor_tier', 'operator_tier', 'purpose'
]

bors_scores_final = bors_scores_full[output_cols].copy()

# Save to CSV
bors_scores_final.to_csv('bors_scores_leo.csv', index=False)

print(f"\n✅ Saved {len(bors_scores_final):,} BORS scores to bors_scores_leo.csv")

# ============================================================================
# STEP 4.6: Print Summary Statistics
# ============================================================================
print("\n[6/7] Generating summary statistics...")
print("\n" + "="*70)
print("STEP 4 SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal LEO satellites with BORS: {len(bors_scores_final):,}")

print("\n--- BORS DISTRIBUTION ---")
print(bors_scores_final['bors_score'].describe())

print("\n--- RISK CATEGORIES ---")
risk_categories = {
    'Very Low (0-30)': (bors_scores_final['bors_score'] < 30).sum(),
    'Low (30-40)': ((bors_scores_final['bors_score'] >= 30) &
                    (bors_scores_final['bors_score'] < 40)).sum(),
    'Medium (40-60)': ((bors_scores_final['bors_score'] >= 40) &
                       (bors_scores_final['bors_score'] < 60)).sum(),
    'High (60-70)': ((bors_scores_final['bors_score'] >= 60) &
                     (bors_scores_final['bors_score'] < 70)).sum(),
    'Very High (70-100)': (bors_scores_final['bors_score'] >= 70).sum()
}

for category, count in risk_categories.items():
    pct = count / len(bors_scores_final) * 100
    print(f"  {category}: {count:,} satellites ({pct:.1f}%)")

print("\n--- PILLAR CONTRIBUTIONS (MEAN VALUES) ---")
print(f"AgeFactor: {bors_scores_final['age_factor'].mean():.3f} (weight: {w_age})")
print(f"TRI: {bors_scores_final['tri'].mean():.3f} (weight: {w_tech})")
print(f"Operator Index: {bors_scores_final['operator_heritage_index'].mean():.3f} (weight: {w_ops})")
print(f"Infant Flag: {bors_scores_final['infant_mortality_flag'].mean():.3f} (weight: {w_err})")

print("\n--- SCORE BY CONTRACTOR TIER ---")
for tier in [1, 2, 3]:
    subset = bors_scores_final[bors_scores_final['contractor_tier'] == tier]
    if len(subset) > 0:
        print(f"Tier {tier} (n={len(subset):,}): Mean BORS = {subset['bors_score'].mean():.1f}")

print("\n--- SCORE BY OPERATOR TIER ---")
for tier in [1, 2, 3]:
    subset = bors_scores_final[bors_scores_final['operator_tier'] == tier]
    if len(subset) > 0:
        print(f"Tier {tier} (n={len(subset):,}): Mean BORS = {subset['bors_score'].mean():.1f}")

print("\n--- INFANT MORTALITY ANALYSIS ---")
infant_sats = bors_scores_final[bors_scores_final['infant_mortality_flag'] == 0.8]
print(f"Satellites in infant mortality period (age <1y): {len(infant_sats):,}")
if len(infant_sats) > 0:
    mature_sats = bors_scores_final[bors_scores_final['infant_mortality_flag'] == 0.1]
    print(f"  Mean BORS (infant): {infant_sats['bors_score'].mean():.1f}")
    print(f"  Mean BORS (mature): {mature_sats['bors_score'].mean():.1f}")
    print(f"  Difference: +{infant_sats['bors_score'].mean() - mature_sats['bors_score'].mean():.1f} points")

print("\n--- AGE ANALYSIS ---")
age_bins = [0, 1, 3, 5, 10, 100]
age_labels = ['<1y', '1-3y', '3-5y', '5-10y', '>10y']
bors_scores_final['age_bin'] = pd.cut(
    bors_scores_final['age_years'],
    bins=age_bins,
    labels=age_labels
)

print("\nMean BORS by age group:")
for label in age_labels:
    subset = bors_scores_final[bors_scores_final['age_bin'] == label]
    if len(subset) > 0:
        print(f"  {label}: {subset['bors_score'].mean():.1f} (n={len(subset):,})")

print("\n--- TOP 10 HIGHEST RISK SATELLITES ---")
top10 = bors_scores_final.nlargest(10, 'bors_score')
for idx, row in top10.iterrows():
    print(f"  {row['OBJECT_NAME'][:40]:40s} | BORS: {row['bors_score']:5.1f} | "
          f"Age: {row['age_years']:4.1f}y | Tier: C{row['contractor_tier']}/O{row['operator_tier']}")

print("\n--- COMPARISON WITH REFERENCE DATASET ---")
print(f"Reference dataset mean BORS: {bors_ref['Operational_Risk_Score'].mean():.1f}")
print(f"Reference dataset std: {bors_ref['Operational_Risk_Score'].std():.1f}")
print(f"Our calculated mean BORS: {bors_scores_final['bors_score'].mean():.1f}")
print(f"Our calculated std: {bors_scores_final['bors_score'].std():.1f}")

diff_mean = abs(bors_scores_final['bors_score'].mean() - bors_ref['Operational_Risk_Score'].mean())
if diff_mean < 10:
    print(f"✓ Mean difference: {diff_mean:.1f} points (acceptable, <10)")
else:
    print(f"⚠ Mean difference: {diff_mean:.1f} points (review calibration)")

# ============================================================================
# STEP 4.7: Validation Checks
# ============================================================================
print("\n[7/7] Running validation checks...")

try:
    # Check 1: All LEO satellites have BORS
    assert len(bors_scores_final) == len(leo_sats), \
        f"ERROR: Mismatch! LEO sats: {len(leo_sats)}, BORS: {len(bors_scores_final)}"

    # Check 2: Scores in valid range
    assert bors_scores_final['bors_score'].min() >= 0, "ERROR: Negative BORS!"
    assert bors_scores_final['bors_score'].max() <= 100, "ERROR: BORS > 100!"

    # Check 3: No nulls in bors_score
    assert bors_scores_final['bors_score'].notna().all(), "ERROR: Null BORS scores!"

    # Check 4: Satellite IDs are unique
    assert bors_scores_final['satellite_id'].nunique() == len(bors_scores_final), \
        "ERROR: Duplicate satellite IDs!"

    # Check 5: Component values in expected ranges
    assert bors_scores_final['age_factor'].between(0, 1).all(), "ERROR: AgeFactor out of range!"
    assert bors_scores_final['tri'].between(0, 1).all(), "ERROR: TRI out of range!"
    assert bors_scores_final['operator_heritage_index'].between(0, 1).all(), "ERROR: Operator index out of range!"
    assert bors_scores_final['infant_mortality_flag'].isin([0.1, 0.8]).all(), "ERROR: Invalid infant flag values!"

    # Check 6: Correlation sanity check - older satellites should have higher BORS
    age_bors_corr = bors_scores_final['age_years'].corr(bors_scores_final['bors_score'])
    assert age_bors_corr > 0.2, f"ERROR: Age-BORS correlation too weak: {age_bors_corr:.3f}"
    print(f"\n✓ Age-BORS correlation: {age_bors_corr:.3f} (healthy positive correlation)")

    print("\n✅ All validation checks passed!")

except AssertionError as e:
    print(f"\n❌ VALIDATION FAILED: {e}")
    raise

print("\n" + "="*70)
print("STEP 4 COMPLETE")
print("="*70)
