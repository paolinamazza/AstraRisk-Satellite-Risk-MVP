"""
STEP 6: ANALYTICS & DASHBOARD DATA GENERATION
===============================================
Generate comprehensive analytics from final risk assessment, including fleet
profiles, scenario analysis, time projections, and multi-risk heatmaps.
Output structured JSON files optimized for dashboard consumption.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 6: ANALYTICS & DASHBOARD DATA GENERATION")
print("="*70)

# ============================================================================
# STEP 6.1: Load Data and Setup
# ============================================================================
print("\n[1/10] Loading data and setting up environment...")

# Load final risk assessment
risk_data = pd.read_csv('final_risk_assessment_leo.csv')

print(f"Loaded {len(risk_data):,} satellites for analytics")
print(f"Columns available: {len(risk_data.columns)}")

# Create output directory for JSON files (if it doesn't exist)
import os
if not os.path.exists('dashboard_data'):
    os.makedirs('dashboard_data')
    print("✓ Created dashboard_data directory")
else:
    print("✓ dashboard_data directory exists")

# ============================================================================
# STEP 6.2: Fleet Summary Statistics
# ============================================================================
print("\n[2/10] Generating fleet summary statistics...")

def generate_fleet_summary(data):
    """
    Generate comprehensive fleet-level statistics
    """

    summary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_satellites': int(len(data)),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        },
        'risk_scores': {
            'total_risk': {
                'mean': round(float(data['total_risk_score'].mean()), 1),
                'median': round(float(data['total_risk_score'].median()), 1),
                'std': round(float(data['total_risk_score'].std()), 1),
                'min': round(float(data['total_risk_score'].min()), 1),
                'max': round(float(data['total_risk_score'].max()), 1),
                'percentile_25': round(float(data['total_risk_score'].quantile(0.25)), 1),
                'percentile_75': round(float(data['total_risk_score'].quantile(0.75)), 1)
            },
            'collision': {
                'mean': round(float(data['collision_score'].mean()), 1),
                'median': round(float(data['collision_score'].median()), 1)
            },
            'environment': {
                'mean': round(float(data['environment_score'].mean()), 1),
                'median': round(float(data['environment_score'].median()), 1)
            },
            'bors': {
                'mean': round(float(data['bors_score'].mean()), 1),
                'median': round(float(data['bors_score'].median()), 1)
            }
        },
        'financial': {
            'total_insured_value': int(data['sum_insured'].sum()),
            'total_annual_premium': int(data['premium_annual'].sum()),
            'total_expected_loss': int(data['expected_annual_loss'].sum()),
            'average_rate_on_line': round(float(data['rate_on_line_pct'].mean()), 2),
            'loss_ratio_pct': round(float((data['expected_annual_loss'].sum() / data['premium_annual'].sum()) * 100), 1)
        },
        'risk_concentration': {
            'top_10_pct_value': int(data.nlargest(int(len(data)*0.1), 'total_risk_score')['sum_insured'].sum()),
            'top_10_pct_share': round(float((data.nlargest(int(len(data)*0.1), 'total_risk_score')['sum_insured'].sum() / data['sum_insured'].sum()) * 100), 1)
        },
        'data_quality': {
            'mean_quality_score': round(float(data['overall_data_quality'].mean()), 1),
            'high_quality_count': int((data['overall_data_quality'] > 80).sum()),
            'high_quality_pct': round(float(((data['overall_data_quality'] > 80).sum() / len(data)) * 100), 1)
        }
    }

    return summary

fleet_summary = generate_fleet_summary(risk_data)

# Save to JSON
with open('dashboard_data/fleet_summary.json', 'w') as f:
    json.dump(fleet_summary, f, indent=2)

print("✓ Generated fleet_summary.json")

# ============================================================================
# STEP 6.3: Risk Distribution Analysis
# ============================================================================
print("\n[3/10] Generating risk distribution analysis...")

def generate_risk_distribution(data):
    """
    Generate detailed risk category distributions
    """

    # Risk categories
    categories = ['Low', 'Medium-Low', 'Medium-High', 'High', 'Critical']

    distribution = {
        'risk_categories': [],
        'dominant_drivers': [],
        'score_bins': []
    }

    # Category distribution
    for category in categories:
        count = int((data['risk_category'] == category).sum())
        total_value = int(data[data['risk_category'] == category]['sum_insured'].sum())

        distribution['risk_categories'].append({
            'category': category,
            'count': count,
            'percentage': round(float((count / len(data)) * 100), 1),
            'total_insured_value': total_value,
            'avg_risk_score': round(float(data[data['risk_category'] == category]['total_risk_score'].mean()), 1) if count > 0 else 0
        })

    # Dominant driver distribution
    for driver in ['Collision', 'Environment', 'BORS']:
        count = int((data['dominant_driver'] == driver).sum())

        distribution['dominant_drivers'].append({
            'driver': driver,
            'count': count,
            'percentage': round(float((count / len(data)) * 100), 1),
            'avg_contribution': round(float(data[data['dominant_driver'] == driver][f'{driver.lower()}_contribution_pct'].mean()), 1) if count > 0 else 0
        })

    # Score bins (0-20, 20-40, 40-60, 60-80, 80-100)
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']

    data_copy = data.copy()
    data_copy['score_bin'] = pd.cut(data_copy['total_risk_score'], bins=bins, labels=labels, include_lowest=True)

    for label in labels:
        count = int((data_copy['score_bin'] == label).sum())

        distribution['score_bins'].append({
            'bin': label,
            'count': count,
            'percentage': round(float((count / len(data)) * 100), 1)
        })

    return distribution

risk_distribution = generate_risk_distribution(risk_data)

with open('dashboard_data/risk_distribution.json', 'w') as f:
    json.dump(risk_distribution, f, indent=2)

print("✓ Generated risk_distribution.json")

# ============================================================================
# STEP 6.4: Top Risk Satellites
# ============================================================================
print("\n[4/10] Generating top risk satellites list...")

def generate_top_risks(data, top_n=50):
    """
    Generate list of highest risk satellites with details
    """

    top_risks = data.nlargest(top_n, 'total_risk_score')

    satellites_list = []

    for idx, (_, row) in enumerate(top_risks.iterrows()):
        satellites_list.append({
            'rank': int(idx + 1),
            'norad_id': int(row['NORAD_CAT_ID']),
            'name': str(row['OBJECT_NAME']),
            'owner': str(row['OWNER']),
            'total_risk_score': round(float(row['total_risk_score']), 1),
            'collision_score': round(float(row['collision_score']), 1),
            'environment_score': round(float(row['environment_score']), 1),
            'bors_score': round(float(row['bors_score']), 1),
            'risk_category': str(row['risk_category']),
            'dominant_driver': str(row['dominant_driver']),
            'altitude_km': round(float(row['ALTITUDE_MEAN_KM']), 1),
            'age_years': round(float(row['AGE_YEARS']), 1),
            'premium_annual': int(row['premium_annual']),
            'rate_on_line_pct': round(float(row['rate_on_line_pct']), 2),
            'orbital_params': {
                'altitude': round(float(row['ALTITUDE_MEAN_KM']), 1),
                'inclination': round(float(row['INCLINATION']), 1),
                'orbit_class': str(row['ORBIT_CLASS'])
            }
        })

    return {'top_risks': satellites_list}

top_risks = generate_top_risks(risk_data, top_n=50)

with open('dashboard_data/top_risks.json', 'w') as f:
    json.dump(top_risks, f, indent=2)

print("✓ Generated top_risks.json")

# ============================================================================
# STEP 6.5: Scenario Analysis
# ============================================================================
print("\n[5/10] Generating scenario analysis (what-if scenarios)...")

def generate_scenario_analysis(data):
    """
    Generate what-if scenarios with different weight configurations
    """

    scenarios = [
        {
            'name': 'Baseline',
            'description': 'Current weights',
            'weights': {'collision': 0.40, 'environment': 0.30, 'bors': 0.30}
        },
        {
            'name': 'Mega-Constellations Era',
            'description': 'Higher collision focus',
            'weights': {'collision': 0.60, 'environment': 0.20, 'bors': 0.20}
        },
        {
            'name': 'Solar Maximum Period',
            'description': 'Higher environment focus',
            'weights': {'collision': 0.30, 'environment': 0.50, 'bors': 0.20}
        },
        {
            'name': 'NewSpace Focus',
            'description': 'Higher operations/heritage focus',
            'weights': {'collision': 0.30, 'environment': 0.20, 'bors': 0.50}
        },
        {
            'name': 'Equal Weights',
            'description': 'All drivers equal',
            'weights': {'collision': 0.333, 'environment': 0.333, 'bors': 0.334}
        }
    ]

    results = []

    for scenario in scenarios:
        w_c = scenario['weights']['collision']
        w_e = scenario['weights']['environment']
        w_b = scenario['weights']['bors']

        # Recalculate total risk with new weights
        scenario_scores = (
            w_c * data['collision_score'] +
            w_e * data['environment_score'] +
            w_b * data['bors_score']
        )

        # Calculate new premiums (simplified)
        def calc_premium_scenario(score):
            if score < 30:
                p_loss = 0.005
            elif score < 50:
                p_loss = 0.015
            elif score < 70:
                p_loss = 0.035
            elif score < 85:
                p_loss = 0.070
            else:
                p_loss = 0.150
            return p_loss

        p_loss_scenario = scenario_scores.apply(calc_premium_scenario)
        premium_scenario = data['sum_insured'] * p_loss_scenario * 1.55

        results.append({
            'scenario': scenario['name'],
            'description': scenario['description'],
            'weights': scenario['weights'],
            'results': {
                'mean_total_risk': round(float(scenario_scores.mean()), 1),
                'median_total_risk': round(float(scenario_scores.median()), 1),
                'total_premium': int(premium_scenario.sum()),
                'avg_rate_on_line': round(float(((premium_scenario.sum() / data['sum_insured'].sum()) * 100)), 2),
                'high_risk_count': int((scenario_scores >= 70).sum()),
                'critical_risk_count': int((scenario_scores >= 85).sum())
            },
            'comparison_to_baseline': {
                'risk_change': round(float(scenario_scores.mean() - data['total_risk_score'].mean()), 1),
                'premium_change_pct': round(float(((premium_scenario.sum() / data['premium_annual'].sum()) - 1) * 100), 1)
            }
        })

    return {'scenarios': results}

scenario_analysis = generate_scenario_analysis(risk_data)

with open('dashboard_data/scenario_analysis.json', 'w') as f:
    json.dump(scenario_analysis, f, indent=2)

print("✓ Generated scenario_analysis.json")

# ============================================================================
# STEP 6.6: Correlation Matrix
# ============================================================================
print("\n[6/10] Generating correlation matrix...")

def generate_correlation_matrix(data):
    """
    Generate correlation analysis between risk drivers
    """

    # Calculate correlations
    corr_cols = ['collision_score', 'environment_score', 'bors_score', 'total_risk_score',
                 'AGE_YEARS', 'ALTITUDE_MEAN_KM']

    corr_matrix = data[corr_cols].corr()

    # Convert to JSON-friendly format
    correlation_data = {
        'variables': corr_cols,
        'matrix': corr_matrix.round(3).values.tolist(),
        'key_insights': []
    }

    # Generate insights
    collision_env_corr = corr_matrix.loc['collision_score', 'environment_score']
    collision_bors_corr = corr_matrix.loc['collision_score', 'bors_score']
    env_bors_corr = corr_matrix.loc['environment_score', 'bors_score']
    age_bors_corr = corr_matrix.loc['AGE_YEARS', 'bors_score']

    correlation_data['key_insights'].append({
        'metric': 'Collision vs Environment',
        'correlation': round(float(collision_env_corr), 3),
        'interpretation': 'Nearly independent' if abs(collision_env_corr) < 0.3 else 'Moderately correlated'
    })

    correlation_data['key_insights'].append({
        'metric': 'Collision vs BORS',
        'correlation': round(float(collision_bors_corr), 3),
        'interpretation': 'Weak positive correlation (older satellites in more crowded orbits)'
    })

    correlation_data['key_insights'].append({
        'metric': 'Age vs BORS',
        'correlation': round(float(age_bors_corr), 3),
        'interpretation': 'Strong positive correlation (as expected, age drives BORS)'
    })

    return correlation_data

correlation_matrix = generate_correlation_matrix(risk_data)

with open('dashboard_data/correlation_matrix.json', 'w') as f:
    json.dump(correlation_matrix, f, indent=2)

print("✓ Generated correlation_matrix.json")

# ============================================================================
# STEP 6.7: Time Projections
# ============================================================================
print("\n[7/10] Generating time projections (10-year forward)...")

def generate_time_projections(data, sample_satellites=20):
    """
    Generate 10-year forward projections for sample satellites

    Projects how risk scores and premiums will evolve
    """

    # Select diverse sample: some high risk, some low risk, some medium
    high_risk = data.nlargest(7, 'total_risk_score')
    low_risk = data.nsmallest(7, 'total_risk_score')
    medium_risk = data[(data['total_risk_score'] >= 45) & (data['total_risk_score'] <= 55)].head(6)

    sample = pd.concat([high_risk, medium_risk, low_risk])

    projections_list = []

    for idx, (_, sat) in enumerate(sample.iterrows()):
        sat_projections = {
            'satellite': {
                'norad_id': int(sat['NORAD_CAT_ID']),
                'name': str(sat['OBJECT_NAME']),
                'current_age': round(float(sat['AGE_YEARS']), 1),
                'expected_lifetime': float(sat['EXPECTED_LIFETIME_YEARS'])
            },
            'years': []
        }

        # Project 10 years forward
        for year in range(11):  # 0 to 10
            future_age = sat['AGE_YEARS'] + year

            # BORS increases with age (AgeFactor component)
            future_age_factor = min(future_age / sat['EXPECTED_LIFETIME_YEARS'], 1.0)
            # Recalculate BORS (simplified - age is 30% of BORS)
            bors_increase = (future_age_factor - sat['age_factor']) * 0.30 * 100
            future_bors = min(sat['bors_score'] + bors_increase, 100)

            # Collision: assume 2% annual increase (congestion growth)
            future_collision = min(sat['collision_score'] * (1.02 ** year), 100)

            # Environment: assume cyclical (11-year solar cycle)
            solar_cycle_boost = 10 * np.sin(2 * np.pi * year / 11)
            future_environment = min(max(sat['environment_score'] + solar_cycle_boost, 0), 100)

            # Total risk
            future_total = (
                0.40 * future_collision +
                0.30 * future_environment +
                0.30 * future_bors
            )

            # Premium
            if future_total < 30:
                p_loss = 0.005
            elif future_total < 50:
                p_loss = 0.015
            elif future_total < 70:
                p_loss = 0.035
            elif future_total < 85:
                p_loss = 0.070
            else:
                p_loss = 0.150

            future_premium = sat['sum_insured'] * p_loss * 1.55
            future_rol = (future_premium / sat['sum_insured']) * 100

            sat_projections['years'].append({
                'year': year,
                'age_years': round(future_age, 1),
                'collision_score': round(future_collision, 1),
                'environment_score': round(future_environment, 1),
                'bors_score': round(future_bors, 1),
                'total_risk_score': round(future_total, 1),
                'premium_annual': int(future_premium),
                'rate_on_line_pct': round(future_rol, 2)
            })

        projections_list.append(sat_projections)

    return {'projections': projections_list}

time_projections = generate_time_projections(risk_data, sample_satellites=20)

with open('dashboard_data/time_projections.json', 'w') as f:
    json.dump(time_projections, f, indent=2)

print("✓ Generated time_projections.json")

# ============================================================================
# STEP 6.8: Multi-Risk Heatmap
# ============================================================================
print("\n[8/10] Generating multi-risk heatmap data...")

def generate_heatmap_data(data):
    """
    Generate heatmap showing collision vs BORS risk distribution
    """

    # Create bins
    collision_bins = [0, 30, 60, 100]
    bors_bins = [0, 40, 60, 100]

    collision_labels = ['Low', 'Medium', 'High']
    bors_labels = ['Low', 'Medium', 'High']

    data_copy = data.copy()
    data_copy['collision_bin'] = pd.cut(data_copy['collision_score'], bins=collision_bins, labels=collision_labels, include_lowest=True)
    data_copy['bors_bin'] = pd.cut(data_copy['bors_score'], bins=bors_bins, labels=bors_labels, include_lowest=True)

    heatmap_data = {
        'axes': {
            'x': bors_labels,
            'y': collision_labels
        },
        'cells': []
    }

    for c_label in collision_labels:
        for b_label in bors_labels:
            subset = data_copy[(data_copy['collision_bin'] == c_label) & (data_copy['bors_bin'] == b_label)]

            if len(subset) > 0:
                avg_risk = round(float(subset['total_risk_score'].mean()), 1)
                count = int(len(subset))
                total_value = int(subset['sum_insured'].sum())
            else:
                avg_risk = 0
                count = 0
                total_value = 0

            heatmap_data['cells'].append({
                'collision_bin': c_label,
                'bors_bin': b_label,
                'count': count,
                'avg_total_risk': avg_risk,
                'total_insured_value': total_value,
                'risk_level': 'critical' if avg_risk >= 70 else 'high' if avg_risk >= 50 else 'medium' if avg_risk >= 30 else 'low'
            })

    # Add insights
    critical_zone = [cell for cell in heatmap_data['cells'] if cell['collision_bin'] == 'High' and cell['bors_bin'] == 'High']
    safe_zone = [cell for cell in heatmap_data['cells'] if cell['collision_bin'] == 'Low' and cell['bors_bin'] == 'Low']

    heatmap_data['insights'] = {
        'critical_zone_count': critical_zone[0]['count'] if critical_zone else 0,
        'safe_zone_count': safe_zone[0]['count'] if safe_zone else 0,
        'highest_concentration': max(heatmap_data['cells'], key=lambda x: x['count'])
    }

    return heatmap_data

heatmap_data = generate_heatmap_data(risk_data)

with open('dashboard_data/heatmap_data.json', 'w') as f:
    json.dump(heatmap_data, f, indent=2)

print("✓ Generated heatmap_data.json")

# ============================================================================
# STEP 6.9: Owner/Operator Analysis
# ============================================================================
print("\n[9/10] Generating owner/operator analysis...")

def generate_owner_analysis(data, top_n=20):
    """
    Generate risk analysis by owner/operator
    """

    # Group by owner
    owner_groups = data.groupby('OWNER').agg({
        'NORAD_CAT_ID': 'count',
        'total_risk_score': 'mean',
        'sum_insured': 'sum',
        'premium_annual': 'sum',
        'rate_on_line_pct': 'mean',
        'collision_score': 'mean',
        'environment_score': 'mean',
        'bors_score': 'mean'
    }).reset_index()

    owner_groups.columns = ['owner', 'satellite_count', 'avg_risk', 'total_insured_value',
                            'total_premium', 'avg_rate', 'avg_collision', 'avg_environment', 'avg_bors']

    # Filter owners with at least 3 satellites
    owner_groups = owner_groups[owner_groups['satellite_count'] >= 3]

    # Sort by total insured value
    owner_groups = owner_groups.sort_values('total_insured_value', ascending=False).head(top_n)

    owners_list = []

    for idx, row in owner_groups.iterrows():
        owners_list.append({
            'owner': str(row['owner']),
            'satellite_count': int(row['satellite_count']),
            'avg_total_risk': round(float(row['avg_risk']), 1),
            'avg_collision': round(float(row['avg_collision']), 1),
            'avg_environment': round(float(row['avg_environment']), 1),
            'avg_bors': round(float(row['avg_bors']), 1),
            'total_insured_value': int(row['total_insured_value']),
            'total_annual_premium': int(row['total_premium']),
            'avg_rate_on_line': round(float(row['avg_rate']), 2)
        })

    return {'owners': owners_list}

owner_analysis = generate_owner_analysis(risk_data, top_n=20)

with open('dashboard_data/owner_analysis.json', 'w') as f:
    json.dump(owner_analysis, f, indent=2)

print("✓ Generated owner_analysis.json")

# ============================================================================
# STEP 6.10: Summary and Validation
# ============================================================================
print("\n[10/10] Validating JSON files and generating summary...")

print("\n" + "="*70)
print("STEP 6 SUMMARY - ANALYTICS & DASHBOARD DATA")
print("="*70)

print("\n✅ Generated JSON files:")
print("   1. fleet_summary.json - Overall fleet statistics")
print("   2. risk_distribution.json - Risk category distributions")
print("   3. top_risks.json - Top 50 highest risk satellites")
print("   4. scenario_analysis.json - 5 what-if scenarios")
print("   5. correlation_matrix.json - Driver correlations")
print("   6. time_projections.json - 10-year projections for 20 satellites")
print("   7. heatmap_data.json - Multi-risk heatmap")
print("   8. owner_analysis.json - Risk by top 20 owners")

print("\n📊 Key Analytics Generated:")
print(f"   • Fleet-level aggregates: risk scores, financials, concentrations")
print(f"   • {len(risk_distribution['risk_categories'])} risk categories analyzed")
print(f"   • {len(top_risks['top_risks'])} highest risk satellites profiled")
print(f"   • {len(scenario_analysis['scenarios'])} scenarios modeled")
print(f"   • {len(time_projections['projections'])} satellites projected 10 years forward")
print(f"   • {len(heatmap_data['cells'])} heatmap cells calculated")
print(f"   • {len(owner_analysis['owners'])} major owners analyzed")

# Validate JSON files
print("\n🔍 Validating JSON files...")
json_files = [
    'fleet_summary.json', 'risk_distribution.json', 'top_risks.json',
    'scenario_analysis.json', 'correlation_matrix.json', 'time_projections.json',
    'heatmap_data.json', 'owner_analysis.json'
]

all_valid = True
for filename in json_files:
    filepath = f'dashboard_data/{filename}'
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"   ✓ {filename} is valid JSON")
    except Exception as e:
        print(f"   ✗ {filename} ERROR: {str(e)}")
        all_valid = False

if all_valid:
    print("\n✅ All JSON files validated successfully!")
else:
    print("\n⚠️ Some JSON files have errors - review above")

# Print sample insights
print("\n" + "="*70)
print("KEY INSIGHTS FROM ANALYTICS")
print("="*70)

print("\n--- FLEET SUMMARY ---")
print(f"Total satellites: {fleet_summary['metadata']['total_satellites']:,}")
print(f"Mean total risk: {fleet_summary['risk_scores']['total_risk']['mean']}")
print(f"Total insured value: €{fleet_summary['financial']['total_insured_value']:,}")
print(f"Total annual premium: €{fleet_summary['financial']['total_annual_premium']:,}")
print(f"Average rate on line: {fleet_summary['financial']['average_rate_on_line']}%")

print("\n--- RISK DISTRIBUTION ---")
for cat in risk_distribution['risk_categories']:
    if cat['count'] > 0:
        print(f"{cat['category']:15s}: {cat['count']:5,} satellites ({cat['percentage']:5.1f}%)")

print("\n--- DOMINANT DRIVERS ---")
for driver in risk_distribution['dominant_drivers']:
    print(f"{driver['driver']:15s}: {driver['count']:5,} satellites ({driver['percentage']:5.1f}%)")

print("\n--- TOP 5 HIGHEST RISK SATELLITES ---")
for sat in top_risks['top_risks'][:5]:
    print(f"{sat['rank']:2d}. {sat['name'][:40]:40s} | Risk: {sat['total_risk_score']:5.1f} | Premium: €{sat['premium_annual']:,}")

print("\n--- SCENARIO ANALYSIS HIGHLIGHTS ---")
for scenario in scenario_analysis['scenarios']:
    print(f"\n{scenario['scenario']}:")
    print(f"  Mean risk: {scenario['results']['mean_total_risk']}")
    print(f"  Premium change: {scenario['comparison_to_baseline']['premium_change_pct']:+.1f}%")

print("\n" + "="*70)
print("✅ STEP 6 COMPLETE - Dashboard data ready")
print("="*70)
