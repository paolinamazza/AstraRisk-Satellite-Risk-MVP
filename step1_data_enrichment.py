"""
STEP 1: DATA ENRICHMENT & LEO FILTERING
========================================
Filter satellite catalog to LEO only (160-2000 km) and enrich with:
- Satellite age
- Launch mass estimates
- Expected lifetime
- Contractor/Operator tiers
- Mission purpose
- Data quality flags
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

print("="*60)
print("STEP 1: DATA ENRICHMENT & LEO FILTERING")
print("="*60)

# ============================================================================
# STEP 1.1: Load and Filter LEO Satellites
# ============================================================================
print("\n[1/11] Loading and filtering LEO satellites...")

satellites = pd.read_csv('satellites_clean.csv')
print(f"Total satellites in catalog: {len(satellites):,}")

# Filter for LEO only (160-2000 km altitude)
leo_satellites = satellites[
    (satellites['ALTITUDE_MEAN_KM'] >= 160) &
    (satellites['ALTITUDE_MEAN_KM'] <= 2000)
].copy()

print(f"LEO satellites (160-2000 km): {len(leo_satellites):,}")
print(f"Percentage: {len(leo_satellites)/len(satellites)*100:.1f}%")

# ============================================================================
# STEP 1.2: Calculate Satellite Age
# ============================================================================
print("\n[2/11] Calculating satellite age...")

REFERENCE_DATE = datetime.now()
print(f"Reference date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")

# Convert LAUNCH_DATE to datetime
leo_satellites['LAUNCH_DATE_DT'] = pd.to_datetime(
    leo_satellites['LAUNCH_DATE'],
    errors='coerce'
)

# Calculate age in years
leo_satellites['AGE_YEARS'] = (
    (REFERENCE_DATE - leo_satellites['LAUNCH_DATE_DT']).dt.days / 365.25
)

# Handle satellites with invalid launch dates
leo_satellites['AGE_YEARS'].fillna(0, inplace=True)

print(f"Age range: {leo_satellites['AGE_YEARS'].min():.1f} - {leo_satellites['AGE_YEARS'].max():.1f} years")

# ============================================================================
# STEP 1.3: Match with Nanosats Database
# ============================================================================
print("\n[3/11] Matching with nanosats database...")

# Load nanosats database
nanosats = pd.read_csv('nanosats_database_20251007.csv')
print(f"Nanosats database entries: {len(nanosats):,}")

# Prepare for fuzzy matching on mission name
def clean_name(name):
    """Remove common prefixes/suffixes for better matching"""
    if pd.isna(name):
        return ''
    name = str(name).upper()
    # Remove common patterns
    name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parentheses content
    name = re.sub(r'\s+', ' ', name).strip()
    return name

leo_satellites['CLEAN_NAME'] = leo_satellites['OBJECT_NAME'].apply(clean_name)
nanosats['CLEAN_NAME'] = nanosats['Mission name'].apply(clean_name)

# Left join to add nanosats data
leo_satellites = leo_satellites.merge(
    nanosats[['CLEAN_NAME', 'Organisation', 'Type (U/mass)', 'Mission description']],
    on='CLEAN_NAME',
    how='left',
    suffixes=('', '_NANO')
)

matched = leo_satellites['Organisation'].notna().sum()
print(f"Satellites matched: {matched:,} ({matched/len(leo_satellites)*100:.1f}%)")

# ============================================================================
# STEP 1.4: Estimate Launch Mass
# ============================================================================
print("\n[4/11] Estimating launch mass...")

def estimate_launch_mass(row):
    """
    Estimate satellite launch mass using multiple methods

    Priority:
    1. Parse from nanosats 'Type (U/mass)' if available
    2. Estimate from OBJECT_TYPE and LAUNCH_YEAR
    3. Default based on orbit characteristics
    """

    # Method 1: Parse CubeSat U designation (e.g., "3U" = 4kg, "6U" = 8kg)
    if pd.notna(row['Type (U/mass)']):
        type_str = str(row['Type (U/mass)'])
        match = re.search(r'(\d+)U', type_str, re.IGNORECASE)
        if match:
            u_count = int(match.group(1))
            mass = u_count * 1.33  # 1U ≈ 1.33kg standard
            return mass, 'CUBESAT_U'

    # Method 2: Estimate from object type and era
    object_type = row['OBJECT_TYPE']
    launch_year = row['LAUNCH_YEAR']

    if object_type == 'DEBRIS':
        return np.nan, 'DEBRIS_SKIP'

    if object_type == 'ROCKET BODY':
        return 2000, 'ROCKET_BODY_DEFAULT'

    # PAYLOAD estimation based on era
    if launch_year < 2000:
        return 800, 'PRE_2000_ESTIMATE'
    elif launch_year < 2010:
        return 500, 'EARLY_2000s_ESTIMATE'
    elif launch_year < 2015:
        return 200, 'SMALLSAT_ERA_ESTIMATE'
    else:
        return 100, 'CUBESAT_ERA_ESTIMATE'

# Apply estimation
leo_satellites[['LAUNCH_MASS_KG', 'MASS_SOURCE']] = leo_satellites.apply(
    estimate_launch_mass, axis=1, result_type='expand'
)

print("\nMass estimation summary:")
print(leo_satellites['MASS_SOURCE'].value_counts())

# ============================================================================
# STEP 1.5: Estimate Expected Lifetime
# ============================================================================
print("\n[5/11] Estimating expected lifetime...")

def estimate_expected_lifetime(mass_kg, object_type, altitude_km):
    """
    Estimate design lifetime based on satellite characteristics

    Rules:
    - CubeSat 1-3U: 2 years
    - CubeSat 6-12U: 3 years
    - SmallSat: 5 years
    - MediumSat: 7 years
    - LargeSat: 12 years
    - Very large: 15 years
    """

    if pd.isna(mass_kg) or object_type in ['DEBRIS', 'ROCKET BODY']:
        return np.nan

    if mass_kg < 10:  # CubeSat 1-3U
        return 2
    elif mass_kg < 50:  # CubeSat 6-12U
        return 3
    elif mass_kg < 100:  # SmallSat
        return 5
    elif mass_kg < 500:  # MediumSat
        return 7
    elif mass_kg < 2500:  # LargeSat
        return 12
    else:  # Very large
        return 15

leo_satellites['EXPECTED_LIFETIME_YEARS'] = leo_satellites.apply(
    lambda row: estimate_expected_lifetime(
        row['LAUNCH_MASS_KG'],
        row['OBJECT_TYPE'],
        row['ALTITUDE_MEAN_KM']
    ),
    axis=1
)

print("\nExpected lifetime distribution:")
print(leo_satellites['EXPECTED_LIFETIME_YEARS'].value_counts().sort_index())

# ============================================================================
# STEP 1.6: Infer Contractor Tier
# ============================================================================
print("\n[6/11] Inferring contractor tier...")

CONTRACTOR_TIER_1 = [
    'AIRBUS', 'BOEING', 'LOCKHEED MARTIN', 'THALES', 'NORTHROP GRUMMAN',
    'NASA', 'ESA', 'ROSCOSMOS', 'CNSA', 'JAXA', 'ISRO',
    'MAXAR', 'ORBITAL ATK', 'BALL AEROSPACE', 'SSL', 'LORAL'
]

CONTRACTOR_TIER_2 = [
    'PLANET', 'SPIRE', 'ICEYE', 'AXELSPACE', 'ASTROSCALE',
    'CAPELLA', 'TERRAN ORBITAL', 'YORK SPACE', 'LG', 'SAMSUNG',
    'MITSUBISHI', 'AIRBUS DS', 'OHB', 'RUAG', 'SITAEL'
]

def infer_contractor_tier(owner_string):
    """
    Infer contractor tier from OWNER field
    Returns: 1, 2, or 3
    """
    if pd.isna(owner_string):
        return 3

    owner_upper = str(owner_string).upper()

    # Check Tier 1
    for contractor in CONTRACTOR_TIER_1:
        if contractor in owner_upper:
            return 1

    # Check Tier 2
    for contractor in CONTRACTOR_TIER_2:
        if contractor in owner_upper:
            return 2

    # Default Tier 3
    return 3

leo_satellites['CONTRACTOR_TIER'] = leo_satellites['OWNER'].apply(infer_contractor_tier)

print("\nContractor Tier distribution:")
print(leo_satellites['CONTRACTOR_TIER'].value_counts().sort_index())

# ============================================================================
# STEP 1.7: Infer Operator Tier
# ============================================================================
print("\n[7/11] Inferring operator tier...")

OPERATOR_TIER_1 = [
    'NASA', 'ESA', 'ROSCOSMOS', 'CNSA', 'JAXA', 'ISRO',
    'US SPACE FORCE', 'USSF', 'US AIR FORCE', 'USAF', 'US NAVY',
    'NOAA', 'NRO', 'DARPA', 'DLR', 'CNES', 'ASI',
    'INTELSAT', 'SES', 'EUTELSAT', 'TELESAT'
]

OPERATOR_TIER_2 = [
    'SPACECOM', 'ORBCOMM', 'IRIDIUM', 'GLOBALSTAR',
    'PLANET', 'SPIRE', 'ICEYE', 'MAXAR', 'CAPELLA',
    'ONEWEB', 'STARLINK', 'SPACEX', 'BLUE ORIGIN'
]

def infer_operator_tier(owner_string):
    """
    Infer operator tier from OWNER field
    Returns: 1, 2, or 3
    """
    if pd.isna(owner_string):
        return 3

    owner_upper = str(owner_string).upper()

    for operator in OPERATOR_TIER_1:
        if operator in owner_upper:
            return 1

    for operator in OPERATOR_TIER_2:
        if operator in owner_upper:
            return 2

    return 3

leo_satellites['OPERATOR_TIER'] = leo_satellites['OWNER'].apply(infer_operator_tier)

print("\nOperator Tier distribution:")
print(leo_satellites['OPERATOR_TIER'].value_counts().sort_index())

# ============================================================================
# STEP 1.8: Infer Mission Purpose
# ============================================================================
print("\n[8/11] Inferring mission purpose...")

def infer_purpose(object_name, owner_string, mission_description):
    """
    Infer satellite mission purpose from available information

    Categories:
    - Communications
    - Earth Observation
    - Navigation
    - Science
    - Technology Demo
    - Military
    """

    # Combine all text fields
    text = ' '.join([
        str(object_name) if pd.notna(object_name) else '',
        str(owner_string) if pd.notna(owner_string) else '',
        str(mission_description) if pd.notna(mission_description) else ''
    ]).upper()

    # Pattern matching
    if any(kw in text for kw in ['STARLINK', 'ONEWEB', 'IRIDIUM', 'GLOBALSTAR',
                                   'TELESAT', 'ORBCOMM', 'INTELSAT', 'COMMS',
                                   'COMMUNICATIONS', 'TELECOM']):
        return 'Communications'

    if any(kw in text for kw in ['SENTINEL', 'LANDSAT', 'TERRA', 'AQUA',
                                   'WORLDVIEW', 'PLANET', 'ICEYE', 'CAPELLA',
                                   'EARTH OBSERVATION', 'IMAGING', 'REMOTE SENSING']):
        return 'Earth Observation'

    if any(kw in text for kw in ['GPS', 'GLONASS', 'GALILEO', 'BEIDOU',
                                   'NAVSTAR', 'NAVIGATION']):
        return 'Navigation'

    if any(kw in text for kw in ['ISS', 'TIANGONG', 'HUBBLE', 'SCIENCE',
                                   'RESEARCH', 'EXPERIMENT']):
        return 'Science'

    if any(kw in text for kw in ['CUBESAT', 'TEST', 'DEMO', 'EXPERIMENTAL',
                                   'UNIVERSITY', 'STUDENT', 'TECHNOLOGY']):
        return 'Technology Demo'

    if any(kw in text for kw in ['MILITARY', 'NAVY', 'AIR FORCE', 'DEFENSE',
                                   'RECONNAISSANCE', 'CLASSIFIED']):
        return 'Military'

    # Default
    return 'Communications'  # Most common

leo_satellites['PURPOSE'] = leo_satellites.apply(
    lambda row: infer_purpose(
        row['OBJECT_NAME'],
        row['OWNER'],
        row.get('Mission description', '')
    ),
    axis=1
)

print("\nPurpose distribution:")
print(leo_satellites['PURPOSE'].value_counts())

# ============================================================================
# STEP 1.9: Add Quality Flags
# ============================================================================
print("\n[9/11] Adding quality flags and completeness scores...")

# Create data quality flags
leo_satellites['MASS_ESTIMATED'] = leo_satellites['MASS_SOURCE'].str.contains(
    'ESTIMATE|DEFAULT',
    na=False
)

leo_satellites['TIER_INFERRED'] = True  # All tiers are inferred

leo_satellites['PURPOSE_INFERRED'] = ~leo_satellites['Mission description'].notna()

leo_satellites['LIFETIME_ESTIMATED'] = True  # All lifetimes are estimated

# Calculate data completeness score (0-100)
def calculate_data_completeness(row):
    score = 100

    if row['MASS_ESTIMATED']:
        score -= 20
    if row['TIER_INFERRED']:
        score -= 15
    if row['PURPOSE_INFERRED']:
        score -= 10
    if row['LIFETIME_ESTIMATED']:
        score -= 15
    if pd.isna(row['ALTITUDE_MEAN_KM']):
        score -= 30

    # Bonus for nanosats match
    if pd.notna(row.get('Organisation')):
        score += 10

    return max(0, min(100, score))

leo_satellites['DATA_COMPLETENESS_SCORE'] = leo_satellites.apply(
    calculate_data_completeness,
    axis=1
)

print(f"\nData completeness distribution:")
print(leo_satellites['DATA_COMPLETENESS_SCORE'].describe())

# ============================================================================
# STEP 1.10: Save Output
# ============================================================================
print("\n[10/11] Saving enriched dataset...")

# Select and order final columns
output_columns = [
    'NORAD_CAT_ID',
    'OBJECT_NAME',
    'OWNER',
    'OBJECT_TYPE',
    'OPS_STATUS_CODE',
    'LAUNCH_DATE',
    'LAUNCH_YEAR',
    'AGE_YEARS',
    'ALTITUDE_MEAN_KM',
    'INCLINATION',
    'APOGEE',
    'PERIGEE',
    'PERIOD',
    'ORBIT_CLASS',
    'LAUNCH_MASS_KG',
    'MASS_SOURCE',
    'EXPECTED_LIFETIME_YEARS',
    'CONTRACTOR_TIER',
    'OPERATOR_TIER',
    'PURPOSE',
    'DATA_COMPLETENESS_SCORE',
    'MASS_ESTIMATED',
    'TIER_INFERRED',
    'PURPOSE_INFERRED',
    'LIFETIME_ESTIMATED'
]

leo_enriched = leo_satellites[output_columns].copy()

# Save to CSV
leo_enriched.to_csv('leo_satellites_enriched.csv', index=False)

print(f"\n✅ Saved {len(leo_enriched):,} LEO satellites to leo_satellites_enriched.csv")

# ============================================================================
# STEP 1.11: Print Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("STEP 1 SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal LEO satellites: {len(leo_enriched):,}")
print(f"Altitude range: {leo_enriched['ALTITUDE_MEAN_KM'].min():.0f} - {leo_enriched['ALTITUDE_MEAN_KM'].max():.0f} km")
print(f"Age range: {leo_enriched['AGE_YEARS'].min():.1f} - {leo_enriched['AGE_YEARS'].max():.1f} years")

print("\nMass distribution:")
print(leo_enriched['LAUNCH_MASS_KG'].describe())

print("\nContractor Tiers:")
for tier in [1, 2, 3]:
    count = (leo_enriched['CONTRACTOR_TIER'] == tier).sum()
    pct = count / len(leo_enriched) * 100
    print(f"  Tier {tier}: {count:,} satellites ({pct:.1f}%)")

print("\nOperator Tiers:")
for tier in [1, 2, 3]:
    count = (leo_enriched['OPERATOR_TIER'] == tier).sum()
    pct = count / len(leo_enriched) * 100
    print(f"  Tier {tier}: {count:,} satellites ({pct:.1f}%)")

print("\nPurpose breakdown:")
print(leo_enriched['PURPOSE'].value_counts())

print("\nData quality:")
print(f"Average completeness score: {leo_enriched['DATA_COMPLETENESS_SCORE'].mean():.1f}/100")
print(f"High quality (>70): {(leo_enriched['DATA_COMPLETENESS_SCORE'] > 70).sum():,} satellites")

# ============================================================================
# VALIDATION CHECKS
# ============================================================================
print("\n[11/11] Running validation checks...")

try:
    # Check 1: No nulls in critical columns
    critical_cols = ['NORAD_CAT_ID', 'ALTITUDE_MEAN_KM', 'AGE_YEARS',
                     'CONTRACTOR_TIER', 'OPERATOR_TIER', 'PURPOSE']
    for col in critical_cols:
        null_count = leo_enriched[col].isna().sum()
        assert null_count == 0, f"ERROR: {col} has {null_count} null values!"

    # Check 2: Altitude range is correct
    assert leo_enriched['ALTITUDE_MEAN_KM'].min() >= 160, "ERROR: Satellites below 160km!"
    assert leo_enriched['ALTITUDE_MEAN_KM'].max() <= 2000, "ERROR: Satellites above 2000km!"

    # Check 3: Tiers are 1, 2, or 3 only
    assert set(leo_enriched['CONTRACTOR_TIER'].unique()) <= {1, 2, 3}, "ERROR: Invalid contractor tiers!"
    assert set(leo_enriched['OPERATOR_TIER'].unique()) <= {1, 2, 3}, "ERROR: Invalid operator tiers!"

    # Check 4: Age is non-negative
    assert leo_enriched['AGE_YEARS'].min() >= 0, "ERROR: Negative ages found!"

    print("\n✅ All validation checks passed!")

except AssertionError as e:
    print(f"\n❌ VALIDATION FAILED: {e}")
    raise

print("\n" + "="*60)
print("STEP 1 COMPLETE")
print("="*60)
