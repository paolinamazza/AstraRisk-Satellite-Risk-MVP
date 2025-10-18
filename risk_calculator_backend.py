"""
Interactive Satellite Risk Calculator - Backend Engine
Provides risk calculation API based on learned models from our dataset
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import re
import traceback
from typing import List, Tuple

# Optional OpenAI client
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # New-style SDK (2024+)
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Load trained data
df = pd.read_csv('final_risk_assessment_leo.csv')

# Build lookup models
def build_models():
    """Build statistical models from our dataset for predictions"""

    models = {}

    # 1. Collision score vs altitude (simple binning + interpolation)
    altitude_bins = [160, 400, 600, 1000, 2000]
    models['collision_vs_altitude'] = df.groupby(pd.cut(df['ALTITUDE_MEAN_KM'], bins=altitude_bins))['collision_score'].mean().to_dict()

    # 2. Environment score vs altitude (with adjustment factor)
    models['environment_vs_altitude'] = df.groupby(pd.cut(df['ALTITUDE_MEAN_KM'], bins=altitude_bins))['environment_score'].mean().to_dict()

    # 3. BORS components (age, tier, purpose)
    models['bors_age_factor'] = df.groupby(pd.cut(df['AGE_YEARS'], bins=[0, 1, 3, 5, 10, 100]))['bors_score'].mean().to_dict()
    models['bors_contractor_tier'] = df.groupby('CONTRACTOR_TIER')['bors_score'].mean().to_dict()
    models['bors_purpose'] = df.groupby('PURPOSE')['bors_score'].mean().to_dict()

    # 4. Fleet statistics for comparison
    models['fleet_stats'] = {
        'mean_total_risk': float(df['total_risk_score'].mean()),
        'mean_collision': float(df['collision_score'].mean()),
        'mean_environment': float(df['environment_score'].mean()),
        'mean_bors': float(df['bors_score'].mean()),
        'std_total_risk': float(df['total_risk_score'].std())
    }

    return models

models = build_models()


# -----------------------------
# Chat / Repo context utilities
# -----------------------------
TEXT_EXTS = {'.py', '.md', '.txt', '.html', '.js', '.json', '.css', '.sh'}
CSV_EXTS = {'.csv'}
MAX_TEXT_BYTES = 120_000  # ~120KB per file excerpt
MAX_CSV_ROWS = 50
MAX_TOTAL_CONTEXT_CHARS = 300_000  # Upper bound across all included content

def _safe_read_text(path: str, max_bytes: int = MAX_TEXT_BYTES) -> str:
    try:
        with open(path, 'rb') as f:
            data = f.read(max_bytes)
        # Try to decode; replace errors to avoid exceptions on mixed encodings
        return data.decode('utf-8', errors='replace')
    except Exception:
        return ''

def _safe_read_csv_head(path: str, max_rows: int = MAX_CSV_ROWS) -> str:
    try:
        # Read a small head to avoid loading very large files
        df_head = pd.read_csv(path, nrows=max_rows)
        # Limit number of columns for huge CSVs
        if df_head.shape[1] > 20:
            df_head = df_head.iloc[:, :20]
        return df_head.to_csv(index=False)
    except Exception:
        # Fallback to raw text head
        return _safe_read_text(path, 50_000)

def _iter_repo_files(root: str = '.') -> List[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden/system folders that aren’t relevant
        if any(part.startswith('.') for part in dirpath.split(os.sep)):
            continue
        for name in filenames:
            # Skip very large binaries by extension
            ext = os.path.splitext(name)[1].lower()
            if ext in TEXT_EXTS or ext in CSV_EXTS:
                files.append(os.path.join(dirpath, name))
    return files

def _file_manifest(paths: List[str]) -> str:
    lines = []
    total_size = 0
    for p in sorted(paths):
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        total_size += size
        lines.append(f"- {p} ({size/1024:.1f} KB)")
    lines.append(f"\nTotal files: {len(paths)} | Total size: {total_size/1024/1024:.2f} MB")
    return "\n".join(lines)

def generate_project_overview() -> str:
    parts = []
    parts.append("Project: Satellite Risk Assessment MVP")
    parts.append("="*80)
    parts.append("")
    parts.append("ARCHITECTURE:")
    parts.append("- Calculator backend (Flask) at http://localhost:5555 serving risk_calculator.html and API endpoints")
    parts.append("- Analytics dashboard (React/Recharts) in risk_dashboard.html on port 8000")
    parts.append("- Data pipeline: step1_data_enrichment.py → step2_collision_risk.py → step3_environment_risk.py → step4_bors_calculation.py → step5_integration_total_risk.py → step6_analytics_dashboard_data.py")
    parts.append("- Key data files: final_risk_assessment_leo.csv (18,612 LEO satellites), collision_scores_leo.csv, environment_scores_leo.csv, bors_scores_leo.csv")
    parts.append("- Launch: ./launch_calculator.sh (port 5555), ./launch_dashboard.sh (port 8000)")
    parts.append("")
    parts.append("KEY APIs (port 5555):")
    parts.append("- POST /api/calculate - Calculate risk scores for satellite parameters")
    parts.append("- GET /api/scenarios - Get available scenarios (Baseline, Solar Storm, Mega-Constellation, etc.)")
    parts.append("- GET /api/fleet_stats - Get fleet-wide statistics for comparison")
    parts.append("- POST /api/chat - OpenAI-powered chat about project (repo-aware)")
    parts.append("")
    parts.append("="*80)
    parts.append("RISK CALCULATION METHODOLOGY (DETAILED FORMULAS):")
    parts.append("="*80)
    parts.append("")
    parts.append("1. COLLISION SCORE (0-100) - step2_collision_risk.py")
    parts.append("   Data source: CDM (Conjunction Data Messages) from Space-Track.org")
    parts.append("   ")
    parts.append("   For satellites WITH CDM data:")
    parts.append("   - Sum all PC (Probability of Collision) values for the satellite over data period")
    parts.append("   - Extrapolate to annual: PC_annual = PC_period × (365 / data_period_days)")
    parts.append("   - Map to 0-100 score using logarithmic scale:")
    parts.append("     * PC_annual < 0.0001 (0.01%): score = 0")
    parts.append("     * PC_annual >= 0.05 (5%): score = 100")
    parts.append("     * In between: score = (log10(PC_annual) - log10(0.0001)) / (log10(0.05) - log10(0.0001)) × 100")
    parts.append("   ")
    parts.append("   For satellites WITHOUT CDM data (imputed from peer groups):")
    parts.append("   - Find peers: satellites with similar altitude (±100km), inclination (±10°), same orbit class")
    parts.append("   - Use median collision score of peers (or mean if <5 peers)")
    parts.append("   - If no peers: use altitude-based baseline:")
    parts.append("     * <400km: 45 (very low LEO, high congestion)")
    parts.append("     * 400-600km: 35 (typical LEO)")
    parts.append("     * 600-1000km: 25 (high LEO)")
    parts.append("     * >1000km: 15 (very high LEO, sparse)")
    parts.append("")
    parts.append("2. ENVIRONMENT SCORE (0-100) - step3_environment_risk.py")
    parts.append("   Data source: OMNI2 space weather data (25+ years hourly measurements)")
    parts.append("   ")
    parts.append("   Method:")
    parts.append("   - Filter weather data for satellite's operational period (launch date to present)")
    parts.append("   - Calculate P95 (95th percentile) of LEO_RiskScore_0_100 during operational period")
    parts.append("   - Apply altitude adjustment factor:")
    parts.append("     * <400km: 1.20 (lower orbits experience more drag risk, +20%)")
    parts.append("     * 400-600km: 1.00 (baseline)")
    parts.append("     * 600-1000km: 0.90 (-10%)")
    parts.append("     * >1000km: 0.75 (-25%)")
    parts.append("   - Adjusted score = P95 × (0.60 + 0.40 × altitude_factor)")
    parts.append("   - Clamp to [0, 100]")
    parts.append("   ")
    parts.append("   LEO_RiskScore_0_100 composition:")
    parts.append("   - 40% atmospheric drag risk (R_drag)")
    parts.append("   - 40% single-event upset risk (R_SEU, radiation)")
    parts.append("   - 20% control anomaly risk (R_ctrl)")
    parts.append("")
    parts.append("3. BORS (Base/Operations Risk Score, 0-100) - step4_bors_calculation.py")
    parts.append("   4-Pillar Model for intrinsic satellite reliability")
    parts.append("   ")
    parts.append("   Weights:")
    parts.append("   - w_age = 0.30 (Age Factor)")
    parts.append("   - w_tech = 0.40 (Technical Reliability Index, TRI)")
    parts.append("   - w_ops = 0.15 (Operator Heritage Index)")
    parts.append("   - w_err = 0.15 (Infant Mortality Flag)")
    parts.append("   ")
    parts.append("   PILLAR 1: Age Factor")
    parts.append("   AgeFactor = min(Age_years / Expected_Lifetime_years, 1.0)")
    parts.append("   - 0 = brand new, 1 = reached/exceeded design lifetime")
    parts.append("   ")
    parts.append("   PILLAR 2: Technical Reliability Index (TRI)")
    parts.append("   TRI = 0.25×CQI + 0.40×HI + 0.20×MRI + 0.15×SVI")
    parts.append("   ")
    parts.append("   CQI (Component Quality Index, mass-based):")
    parts.append("   - mass < 100kg: 0.7 (high risk, COTS components likely)")
    parts.append("   - 100kg ≤ mass < 500kg: 0.5")
    parts.append("   - mass ≥ 500kg: 0.3 (low risk, space-grade components)")
    parts.append("   ")
    parts.append("   HI (Heritage Index, contractor tier):")
    parts.append("   - Tier 1 (NASA, Boeing, Lockheed Martin, etc.): 0.2 (experienced, low risk)")
    parts.append("   - Tier 2 (mid-sized contractors): 0.5")
    parts.append("   - Tier 3 (new entrants, startups): 0.8 (high risk)")
    parts.append("   ")
    parts.append("   MRI (Mass Reliability Index, U-shaped curve):")
    parts.append("   - mass < 500kg: 0.7 (small sats, high infant mortality)")
    parts.append("   - 500kg ≤ mass ≤ 2500kg: 0.3 (sweet spot, low risk)")
    parts.append("   - mass > 2500kg: 0.6 (large, system complexity risk)")
    parts.append("   ")
    parts.append("   SVI (Subsystem Vulnerability Index, mission purpose):")
    parts.append("   - Communications: 0.8 (critical power & comms subsystems)")
    parts.append("   - Earth Observation: 0.6 (complex payload)")
    parts.append("   - Technology Demo: 0.5")
    parts.append("   - Science/Navigation: 0.4")
    parts.append("   ")
    parts.append("   PILLAR 3: Operator Heritage Index")
    parts.append("   - Tier 1 (government agencies, major operators): 0.1 (very low risk)")
    parts.append("   - Tier 2 (established commercial): 0.4")
    parts.append("   - Tier 3 (new operators): 0.7 (high risk)")
    parts.append("   ")
    parts.append("   PILLAR 4: Infant Mortality Flag")
    parts.append("   - Age < 1 year: 0.8 (high risk period)")
    parts.append("   - Age ≥ 1 year: 0.1 (normal operations)")
    parts.append("   ")
    parts.append("   FINAL BORS:")
    parts.append("   raw_score = 0.30×AgeFactor + 0.40×TRI + 0.15×Operator_Index + 0.15×Infant_Flag")
    parts.append("   BORS = raw_score × 100")
    parts.append("   Clamp to [0, 100]")
    parts.append("")
    parts.append("4. TOTAL RISK SCORE (0-100) - step5_integration_total_risk.py")
    parts.append("   Weighted combination of three drivers:")
    parts.append("   ")
    parts.append("   Total Risk = 0.40 × Collision + 0.30 × Environment + 0.30 × BORS")
    parts.append("   ")
    parts.append("   Risk Categories:")
    parts.append("   - Low: <30")
    parts.append("   - Medium-Low: 30-50")
    parts.append("   - Medium-High: 50-70")
    parts.append("   - High: 70-85")
    parts.append("   - Critical: 85-100")
    parts.append("")
    parts.append("5. INSURANCE PREMIUM CALCULATION - step5_integration_total_risk.py")
    parts.append("   ")
    parts.append("   Step 1: Map Total Risk Score → Annual Loss Probability")
    parts.append("   - Total Risk < 30: P_loss = 0.5%")
    parts.append("   - 30 ≤ Total Risk < 50: P_loss = 1.5%")
    parts.append("   - 50 ≤ Total Risk < 70: P_loss = 3.5%")
    parts.append("   - 70 ≤ Total Risk < 85: P_loss = 7.0%")
    parts.append("   - Total Risk ≥ 85: P_loss = 15.0% (nearly uninsurable)")
    parts.append("   ")
    parts.append("   Step 2: Calculate Base Premium (Expected Loss)")
    parts.append("   Base_Premium = Sum_Insured × P_loss_annual")
    parts.append("   ")
    parts.append("   Step 3: Apply Load Factor")
    parts.append("   Load_Factor = 1.55 composed of:")
    parts.append("   - 30% expense ratio (operational costs)")
    parts.append("   - 15% profit margin")
    parts.append("   - 10% uncertainty buffer")
    parts.append("   ")
    parts.append("   Step 4: Final Premium")
    parts.append("   Premium_Annual = Base_Premium × 1.55")
    parts.append("   Rate_on_Line = (Premium_Annual / Sum_Insured) × 100%")
    parts.append("   ")
    parts.append("   Sum Insured Estimation (if not provided):")
    parts.append("   - mass < 100kg: €50,000 per kg")
    parts.append("   - 100kg ≤ mass < 500kg: €40,000 per kg")
    parts.append("   - mass ≥ 500kg: €30,000 per kg (economies of scale)")
    parts.append("")
    parts.append("6. SCENARIO ADJUSTMENTS")
    parts.append("   Scenarios modify baseline scores with multipliers:")
    parts.append("   ")
    parts.append("   - Baseline: No adjustments (1.0× all drivers)")
    parts.append("   - Solar Storm: Environment × 1.30 (+30%)")
    parts.append("   - Mega-Constellation Era: Collision × 1.50 (+50%)")
    parts.append("   - Solar Minimum: Environment × 0.75 (-25%)")
    parts.append("   - Active Debris Removal: Collision × 0.70 (-30%)")
    parts.append("")
    parts.append("="*80)
    parts.append("DATA SOURCES & TRAINING:")
    parts.append("="*80)
    parts.append("- 18,612 LEO satellites from UCS Satellite Database + Space-Track.org")
    parts.append("- CDM (Conjunction Data Messages): conjunction events with PC values")
    parts.append("- OMNI2 space weather: 25+ years of hourly measurements (drag, radiation, control risks)")
    parts.append("- BORS reference dataset: bors_leo_operational_risk_dataset_v2.csv with weights and mappings")
    parts.append("- Calculator uses learned statistical models from this training data")
    parts.append("")
    parts.append("="*80)
    parts.append("INSURANCE INDUSTRY CONTEXT:")
    parts.append("="*80)
    parts.append("")
    parts.append("UNDERWRITING GUIDELINES BY RISK CATEGORY:")
    parts.append("")
    parts.append("Risk Score 0-30 (Low Risk):")
    parts.append("- Coverage: Standard terms, competitive pricing, streamlined underwriting")
    parts.append("- Typical satellites: New Tier 1 satellites (Boeing, Lockheed) in proven orbits")
    parts.append("- Authority level: Local underwriter can bind")
    parts.append("- Deductibles: 5-10% of sum insured")
    parts.append("- Rate-on-Line (In-Orbit): 1.0-2.0%")
    parts.append("- Rate-on-Line (Launch+1st year): 4-7% for Tier 1 contractors")
    parts.append("- Policy conditions: Standard telemetry, 30-day notice for orbital changes")
    parts.append("- Portfolio treatment: No special concentration limits")
    parts.append("")
    parts.append("Risk Score 30-50 (Medium-Low Risk):")
    parts.append("- Coverage: Standard with enhanced conditions")
    parts.append("- Typical satellites: Established commercial operators, mid-life satellites")
    parts.append("- Authority level: Regional underwriter approval required")
    parts.append("- Deductibles: 10-15% of sum insured")
    parts.append("- Rate-on-Line (In-Orbit): 2.0-3.5%")
    parts.append("- Rate-on-Line (Launch+1st year): 7-12% for Tier 2 contractors")
    parts.append("- Policy conditions: Must have orbital maneuver capability, collision avoidance system")
    parts.append("- Required: Space-Track.org subscription, 24hr CDM response protocol")
    parts.append("- Portfolio treatment: Monitor concentration in congested orbits (550km band)")
    parts.append("")
    parts.append("Risk Score 50-70 (Medium-High Risk):")
    parts.append("- Coverage: Non-standard, higher pricing, enhanced scrutiny")
    parts.append("- Typical satellites: Aging satellites (>10yrs), Tier 3 contractors, congested orbits")
    parts.append("- Authority level: Headquarters approval required")
    parts.append("- Deductibles: 15-25% of sum insured")
    parts.append("- Rate-on-Line (In-Orbit): 3.5-6.0%")
    parts.append("- Rate-on-Line (Launch+1st year): 12-20% for Tier 3 contractors")
    parts.append("- Policy conditions: Enhanced telemetry (daily reports), active debris tracking")
    parts.append("- Required: Proof of collision avoidance maneuvers, solar weather monitoring")
    parts.append("- Portfolio treatment: Strict concentration limits (max 15 sats in same 100km altitude shell)")
    parts.append("- Co-insurance: May require for sum insured >$100M")
    parts.append("")
    parts.append("Risk Score 70-85 (High Risk):")
    parts.append("- Coverage: Specialized, significant restrictions, capacity constraints")
    parts.append("- Typical satellites: End-of-life, high collision zones, unproven operators")
    parts.append("- Authority level: Executive underwriter + actuarial review")
    parts.append("- Deductibles: 25-40% of sum insured")
    parts.append("- Rate-on-Line: 6.0-15.0% (if coverage available)")
    parts.append("- Policy conditions: Strict operational requirements, real-time telemetry")
    parts.append("- Required: Independent failure analysis capability, immediate incident reporting")
    parts.append("- Portfolio treatment: Requires CEO approval, counts double for concentration limits")
    parts.append("- Co-insurance/Excess layers: Mandatory for sum insured >$50M")
    parts.append("- Market capacity: Limited, may require Lloyd's specialty syndicates")
    parts.append("")
    parts.append("Risk Score 85-100 (Critical Risk):")
    parts.append("- Coverage: Declining/Declination consideration, potentially uninsurable")
    parts.append("- Typical satellites: Near end-of-life, debris-heavy orbits, failed components")
    parts.append("- Authority level: Board approval required")
    parts.append("- Rate-on-Line: 15%+ (if coverage available at all)")
    parts.append("- Market approach: Refer to specialty markets, Lloyd's syndicates, political risk insurers")
    parts.append("- Alternative solutions: Government guarantees, self-insurance, captive structures")
    parts.append("- Typical outcome: Declined for commercial terms")
    parts.append("")
    parts.append("="*80)
    parts.append("HISTORICAL LOSS EXPERIENCE (What Insurers Actually Pay For):")
    parts.append("="*80)
    parts.append("")
    parts.append("COLLISION EVENTS:")
    parts.append("- Iridium 33/Cosmos 2251 collision (2009): Total loss, ~$40M claim")
    parts.append("  * Two satellites collided in LEO, creating thousands of debris fragments")
    parts.append("  * This event increased collision risk scores industry-wide by 10-15%")
    parts.append("- Close conjunctions requiring evasive maneuvers: $50K-$500K per event")
    parts.append("  * Fuel consumption reduces satellite lifetime (depreciation claim)")
    parts.append("  * Operational disruption, ground station costs")
    parts.append("- Loss frequency: Approximately 2-3 total loss collision events per 10,000 satellite-years")
    parts.append("- Severity: 100% total loss when collision occurs (satellites unrecoverable)")
    parts.append("- Subrogation: Extremely difficult (attribution, liability conventions, sovereign immunity)")
    parts.append("")
    parts.append("ENVIRONMENT EVENTS:")
    parts.append("- February 2022 Solar Storm: Destroyed 40 Starlink satellites shortly after launch")
    parts.append("  * Estimated loss: $50M+ (satellites, launch costs, business interruption)")
    parts.append("  * Geomagnetic storm increased atmospheric drag, satellites couldn't reach orbit")
    parts.append("  * Impact: Environment risk scores for <500km altitude increased 20%")
    parts.append("- Geomagnetic storms causing orbital decay: Common for satellites <500km altitude")
    parts.append("  * Gradual loss of altitude, premature reentry")
    parts.append("  * Claims: Early end-of-life, reduced revenue, replacement costs")
    parts.append("- Radiation-induced failures: Typical in satellites >5 years old")
    parts.append("  * Single Event Upsets (SEU), solar panel degradation, battery failures")
    parts.append("  * Claims: Partial loss (component replacement impossible), total loss if critical subsystem")
    parts.append("- Solar panel degradation: 1-2% per year normal, 5-10% during major solar events")
    parts.append("- Loss frequency: 1-2 major solar storm events per solar cycle (11 years)")
    parts.append("")
    parts.append("BORS (BASE/OPERATIONS RISK) CLAIMS:")
    parts.append("- Infant mortality (first year): Accounts for 60% of all satellite insurance claims")
    parts.append("  * Manufacturing defects, launch stress damage, deployment failures")
    parts.append("  * Highest risk: First 30 days (10x higher than mature operations)")
    parts.append("  * Tier 3 contractors: 3-5x higher infant mortality than Tier 1")
    parts.append("- Component failures by type (ranked by claim frequency):")
    parts.append("  1. Battery failures (30% of BORS claims) - thermal runaway, cell degradation")
    parts.append("  2. Solar panel issues (25%) - deployment failures, micrometeorite damage")
    parts.append("  3. Attitude control (20%) - reaction wheel failures, thruster issues")
    parts.append("  4. Communications (15%) - transponder failures, antenna deployment")
    parts.append("  5. Power distribution (10%) - short circuits, regulator failures")
    parts.append("- Age-related degradation: Failure rate doubles after design lifetime exceeded")
    parts.append("- Operator experience factor: New operators have 2-3x higher claim rates")
    parts.append("")
    parts.append("CLAIMS STATISTICS:")
    parts.append("- Average claim size: $75M (highly skewed by large satellite total losses)")
    parts.append("- Median claim size: $8M (more representative of typical claims)")
    parts.append("- Claim frequency: ~3-4 claims per 1,000 satellite-years in-orbit")
    parts.append("- Launch vs In-Orbit: Launch+1st year accounts for 65% of claims, 35% during mature operations")
    parts.append("- Partial vs Total Loss: 30% partial loss, 70% total loss (satellites hard to repair)")
    parts.append("")
    parts.append("="*80)
    parts.append("PORTFOLIO MANAGEMENT & AGGREGATION:")
    parts.append("="*80)
    parts.append("")
    parts.append("COLLISION RISK CONCENTRATION LIMITS:")
    parts.append("- Maximum 15 satellites in 550km ±50km altitude band (Starlink/OneWeb congestion zone)")
    parts.append("  * Reason: Single debris event could cascade, affecting all satellites in shell")
    parts.append("  * Kessler Syndrome risk: Collision creates debris → more collisions → exponential growth")
    parts.append("- Maximum 20 satellites in any single 100km altitude shell")
    parts.append("- Altitude bands by risk:")
    parts.append("  * 500-600km: HIGHEST risk (mega-constellation zone, 60% of all LEO satellites)")
    parts.append("  * 700-800km: HIGH risk (legacy satellites, debris from past missions)")
    parts.append("  * 300-400km: MEDIUM risk (atmosphere clears debris faster)")
    parts.append("  * >1000km: LOWER risk (sparse, but debris persists longer)")
    parts.append("- Inclination concentration: Max 25% of portfolio in polar orbits (>80° inclination)")
    parts.append("  * Reason: Polar orbits cross all other orbital planes (high conjunction rate)")
    parts.append("")
    parts.append("ENVIRONMENT RISK CONCENTRATION LIMITS:")
    parts.append("- Maximum aggregate sum insured: $500M for satellites <400km altitude")
    parts.append("  * Reason: High atmospheric drag risk, single solar storm can affect entire fleet")
    parts.append("- Solar storm correlation factor: 0.7 for satellites in same altitude ±100km")
    parts.append("  * All satellites in band experience similar drag increase simultaneously")
    parts.append("- Geographic correlation: 0.5 for satellites with similar orbital inclination")
    parts.append("- Portfolio diversification: Aim for 30% each in low (<500km), mid (500-800km), high (>800km) LEO")
    parts.append("")
    parts.append("OPERATOR/CONTRACTOR RISK CONCENTRATION:")
    parts.append("- Maximum 25% of total portfolio to any single operator (except US Gov/ESA/JAXA)")
    parts.append("- Maximum 35% of portfolio to Tier 3 contractors combined")
    parts.append("- Maximum 20% to any single mega-constellation operator (Starlink, OneWeb, etc.)")
    parts.append("- Required diversification across launch vehicles:")
    parts.append("  * No more than 30% on single launch vehicle type (avoid single-point failure)")
    parts.append("  * Example: SpaceX Falcon 9 failure could ground entire Starlink fleet temporarily")
    parts.append("")
    parts.append("AGGREGATE EXPOSURE LIMITS:")
    parts.append("- Total satellite portfolio PML (Probable Maximum Loss): $2B for 1-in-100 year event")
    parts.append("- Per-event limit: $500M (single collision cascade or major solar storm)")
    parts.append("- Reinsurance required: For portfolios >$1B aggregate sum insured")
    parts.append("- Catastrophe modeling: Annual aggregate loss <10% of total premiums (target)")
    parts.append("")
    parts.append("="*80)
    parts.append("MARKET PRICING BENCHMARKS:")
    parts.append("="*80)
    parts.append("")
    parts.append("RATE-ON-LINE BY PHASE:")
    parts.append("")
    parts.append("Launch + First Year (Infant Mortality Period):")
    parts.append("- Tier 1 contractors (NASA, Boeing, Lockheed, Airbus, Thales): 4-7% RoL")
    parts.append("- Tier 2 contractors (SSL, ISRO, Mitsubishi): 7-12% RoL")
    parts.append("- Tier 3 contractors (Startups, NewSpace, unproven): 12-20% RoL")
    parts.append("- Small satellites (<500kg): Add 2-5% premium loading (higher infant mortality)")
    parts.append("- CubeSats (<100kg): Add 5-10% premium loading (COTS components)")
    parts.append("")
    parts.append("In-Orbit Coverage (After Successful First Year):")
    parts.append("- Low risk (score <30): 1.0-2.0% RoL")
    parts.append("- Medium-Low risk (score 30-50): 2.0-3.5% RoL")
    parts.append("- Medium-High risk (score 50-70): 3.5-6.0% RoL")
    parts.append("- High risk (score 70-85): 6.0-15.0% RoL (capacity limited)")
    parts.append("- Critical risk (score 85+): 15%+ or declined")
    parts.append("")
    parts.append("MARKET CAPACITY & STRUCTURE:")
    parts.append("- Total global satellite insurance market: ~$600M annual gross written premium")
    parts.append("- Market growth: 15-20% annually (driven by mega-constellations)")
    parts.append("- Single risk limit: $300-500M (requires syndication/reinsurance above $150M)")
    parts.append("- Lloyd's of London: ~40% of global satellite insurance capacity")
    parts.append("- Major players: AXA XL, Allianz, AIG, Munich Re, Swiss Re, Atrium")
    parts.append("- Reinsurance: Typically 50-70% ceded for large risks")
    parts.append("")
    parts.append("DEDUCTIBLE STRUCTURES:")
    parts.append("- Total loss only: No deductible for total loss, no coverage for partial loss")
    parts.append("  * Common for mega-constellations (satellites treated as commodities)")
    parts.append("- Standard deductible: 10-20% of sum insured for all claims")
    parts.append("- Franchise deductible: No deductible if loss >threshold (e.g., 50% of value)")
    parts.append("- Aggregate deductible: For fleet policies (e.g., $10M aggregate before coverage)")
    parts.append("")
    parts.append("PREMIUM ADJUSTMENTS & LOADINGS:")
    parts.append("- New operator surcharge: +20-50% for operators with <3 successful missions")
    parts.append("- Congested orbit loading: +10-30% for satellites in 500-600km band")
    parts.append("- Age penalty: +5% per year after design lifetime exceeded")
    parts.append("- No collision avoidance capability: +15-25% loading")
    parts.append("- Non-compliance with debris mitigation: +25-50% or decline")
    parts.append("- Space Sustainability Rating discount: -5% to -15% for high scores")
    parts.append("")
    parts.append("="*80)
    parts.append("POLICY CONDITIONS & CLAIMS REQUIREMENTS:")
    parts.append("="*80)
    parts.append("")
    parts.append("PRE-LOSS RISK MITIGATION REQUIREMENTS:")
    parts.append("")
    parts.append("Collision Risk Mitigation:")
    parts.append("- MANDATORY: Subscription to Space-Track.org or equivalent CDM service")
    parts.append("- MANDATORY: Documented collision avoidance procedures")
    parts.append("- REQUIRED: Response to CDM warnings (PC >0.01%) within 24 hours")
    parts.append("- REQUIRED: Orbital maneuver capability (fuel reserves >10% EOL)")
    parts.append("- RECOMMENDED: Participation in Space Data Association (SDA)")
    parts.append("- Failure to comply: May void collision coverage or reduce payout 50-100%")
    parts.append("")
    parts.append("Environment Risk Mitigation:")
    parts.append("- MANDATORY: Solar weather monitoring subscription (NOAA SWPC or equivalent)")
    parts.append("- REQUIRED: Safe mode protocols for solar storm events")
    parts.append("- REQUIRED: Battery management procedures (prevent thermal runaway)")
    parts.append("- RECOMMENDED: Autonomous safe mode activation")
    parts.append("- For <400km altitude: Enhanced drag compensation capability required")
    parts.append("")
    parts.append("BORS Risk Mitigation:")
    parts.append("- MANDATORY: Minimum 90-day post-launch telemetry review before infant mortality premium reduction")
    parts.append("- REQUIRED: Daily telemetry during first year, weekly thereafter")
    parts.append("- REQUIRED: Component health monitoring (battery, solar panels, reaction wheels)")
    parts.append("- REQUIRED: Redundant critical subsystems for high-value satellites (>$100M)")
    parts.append("- Tier 3 contractors: Must provide independent quality assurance certification")
    parts.append("")
    parts.append("CLAIMS DOCUMENTATION REQUIRED:")
    parts.append("- Telemetry logs: 30 days pre-event (hourly resolution minimum)")
    parts.append("- Operator response logs: All CDM warnings and actions taken")
    parts.append("- Ground station logs: Commands sent, acknowledgments received")
    parts.append("- Independent failure analysis: From qualified third party (e.g., Aerospace Corporation)")
    parts.append("- Launch vehicle certification: For launch-related claims")
    parts.append("- Orbital tracking data: TLE (Two-Line Element) history, conjunction analysis")
    parts.append("- Component test records: For BORS-related claims")
    parts.append("- Photos/video: Launch, deployment, manufacturing (if available)")
    parts.append("")
    parts.append("SUBROGATION & LIABILITY:")
    parts.append("- Insurer retains subrogation rights against liable third parties")
    parts.append("- UN Outer Space Treaty (1967): Launching state liable for damage")
    parts.append("- UN Liability Convention (1972): Absolute liability for surface damage, fault-based for space")
    parts.append("- Challenge: Attribution extremely difficult (debris origin often unknown)")
    parts.append("- Typical recovery: <5% of claims (most claims deemed 'act of space')")
    parts.append("- Notable case: Cosmos 954 (1978) - USSR paid Canada $3M for cleanup")
    parts.append("")
    parts.append("="*80)
    parts.append("REGULATORY & COMPLIANCE CONSIDERATIONS:")
    parts.append("="*80)
    parts.append("")
    parts.append("FCC (US) / ITU (International) REQUIREMENTS:")
    parts.append("- Orbital debris mitigation: 25-year deorbit requirement for LEO satellites")
    parts.append("- Collision risk assessment: Must be <0.001 (1 in 1,000) probability")
    parts.append("- Non-compliance: May void insurance coverage or increase premium 50%+")
    parts.append("- Post-mission disposal: Controlled reentry or graveyard orbit required")
    parts.append("- Insurance impact: Satellites without deorbit capability may be uninsurable")
    parts.append("")
    parts.append("SPACE SUSTAINABILITY RATING (SSR):")
    parts.append("- Developed by: World Economic Forum, ESA, MIT")
    parts.append("- Scale: 0-100 (higher = more sustainable)")
    parts.append("- Factors: Debris mitigation, collision avoidance, end-of-life disposal, data sharing")
    parts.append("- Insurance application: Increasingly used for premium discounts")
    parts.append("  * SSR >75: 5-10% premium discount")
    parts.append("  * SSR 50-75: Standard premium")
    parts.append("  * SSR <50: 10-20% premium loading or decline")
    parts.append("- Trend: May become mandatory for coverage by 2025-2027")
    parts.append("")
    parts.append("MANDATORY INSURANCE REQUIREMENTS:")
    parts.append("- Commercial operators: Often required by launch service agreements")
    parts.append("  * SpaceX rideshare: $10M minimum third-party liability insurance")
    parts.append("  * Arianespace: €60M minimum for dedicated launches")
    parts.append("- Government operators: Typically self-insured or sovereign guarantee")
    parts.append("- Mega-constellations: May use captive insurance vehicles (tax efficiency)")
    parts.append("- Export controls: ITAR/EAR affect policy placement (must use approved markets)")
    parts.append("")
    parts.append("EMERGING REGULATIONS:")
    parts.append("- Active Debris Removal (ADR): May become mandatory for high-risk satellites")
    parts.append("- Traffic coordination: Space Traffic Management (STM) regulations pending")
    parts.append("- Licensing conditions: Some jurisdictions requiring insurance as condition of license")
    parts.append("- Liability caps: Discussions on capping operator liability (like aviation)")
    parts.append("")
    parts.append("="*80)
    parts.append("PRACTICAL UNDERWRITING EXAMPLES:")
    parts.append("="*80)
    parts.append("")
    parts.append("EXAMPLE 1: Favorable Risk - New Tier 1 Satellite")
    parts.append("Satellite: Commercial Earth observation, 800kg, Airbus Defence & Space")
    parts.append("Orbit: 600km, 98° sun-synchronous (polar)")
    parts.append("Age: 2 years, design lifetime 7 years")
    parts.append("Risk scores: Collision=45, Environment=55, BORS=35, Total=45")
    parts.append("Sum insured: €50M")
    parts.append("")
    parts.append("Underwriting decision:")
    parts.append("- Risk category: Medium-Low (30-50)")
    parts.append("- Premium: €1.2M annual (2.4% RoL)")
    parts.append("- Deductible: 10% (€5M)")
    parts.append("- Authority: Regional underwriter can bind")
    parts.append("- Conditions: Standard telemetry, Space-Track subscription, 24hr CDM response")
    parts.append("- Capacity: Full €50M on single insurer paper (no co-insurance needed)")
    parts.append("- Why favorable: Tier 1 contractor, past infant mortality, good altitude, polar orbit acceptable")
    parts.append("")
    parts.append("EXAMPLE 2: Challenging Risk - Aging Tier 3 Satellite in Congested Orbit")
    parts.append("Satellite: Communications, 400kg, new Chinese commercial operator")
    parts.append("Orbit: 550km, 53° (Starlink congestion zone)")
    parts.append("Age: 8 years, design lifetime 7 years (exceeded)")
    parts.append("Risk scores: Collision=85, Environment=60, BORS=70, Total=74")
    parts.append("Sum insured: €15M")
    parts.append("")
    parts.append("Underwriting decision:")
    parts.append("- Risk category: High (70-85)")
    parts.append("- Premium: €1.35M annual (9.0% RoL - if coverage offered)")
    parts.append("- Deductible: 30% (€4.5M)")
    parts.append("- Authority: Executive underwriter + actuarial review + CEO approval")
    parts.append("- Conditions: Daily telemetry, real-time CDM monitoring, proof of collision avoidance capability")
    parts.append("- Capacity: Likely DECLINE due to:")
    parts.append("  * Exceeded design lifetime (age penalty)")
    parts.append("  * Tier 3 operator with limited track record")
    parts.append("  * Starlink zone (portfolio concentration risk)")
    parts.append("  * Collision score 85 (very high, PC likely >5% annual)")
    parts.append("- Alternative: Refer to Lloyd's specialty syndicates or political risk markets")
    parts.append("- If forced to quote: 12-15% RoL with 40% deductible, max capacity €5M")
    parts.append("")
    parts.append("EXAMPLE 3: Mega-Constellation - Fleet Policy")
    parts.append("Operator: OneWeb-style constellation")
    parts.append("Fleet: 300 satellites, 150kg each, Tier 2 contractor")
    parts.append("Orbit: 1,200km, 87° polar")
    parts.append("Average risk score: 48 per satellite")
    parts.append("Total sum insured: €2.7B (300 × €9M each)")
    parts.append("")
    parts.append("Underwriting decision:")
    parts.append("- Premium: €70M annual (~2.6% RoL, bulk discount)")
    parts.append("- Structure: Total loss only coverage (no partial loss)")
    parts.append("- Deductible: 5 satellites per year (aggregate)")
    parts.append("- Reinsurance: 70% ceded (retain €20M premium)")
    parts.append("- Capacity: Syndicated across 8-10 insurers")
    parts.append("- Special conditions:")
    parts.append("  * Launch in batches (max 30 per launch, separate policies)")
    parts.append("  * Automated CDM response system required")
    parts.append("  * Captive insurance vehicle for first-loss layer (€50M)")
    parts.append("  * Market covers excess layer only (€50M-€500M)")
    parts.append("- Key considerations:")
    parts.append("  * High altitude (1,200km) reduces collision vs Starlink zone")
    parts.append("  * Polar orbit increases conjunction frequency (premium loading)")
    parts.append("  * Small satellites (150kg) increase BORS risk but low individual value")
    parts.append("  * Portfolio diversification benefit (300 sats, loss of 5-10 acceptable)")
    parts.append("")
    parts.append("="*80)
    return "\n".join(parts)

def _keyword_score(content: str, keywords: List[str]) -> int:
    score = 0
    lc = content.lower()
    for kw in keywords:
        if not kw:
            continue
        # Count occurrences; simple heuristic
        score += lc.count(kw)
    return score

def build_repo_context(query: str, max_files: int = 5, include_manifest: bool = True, full_mode: bool = False) -> Tuple[str, List[str]]:
    """Build repository context string and sources.

    - Keyword-based selection by default (top N files)
    - In full_mode, include a manifest of all files and broaden selection
    - Always prepends a high-level project overview
    """
    keywords = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", query) if len(w) > 2]
    candidates: List[Tuple[int, str]] = []

    all_paths = _iter_repo_files('.')
    for path in all_paths:
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in CSV_EXTS:
                preview = _safe_read_csv_head(path)
            else:
                preview = _safe_read_text(path)

            score = _keyword_score(preview, keywords)
            # Prioritize key calculation files
            filename = os.path.basename(path)
            base_bonus = 0
            if filename in {'risk_calculator_backend.py', 'risk_calculator.html', 'risk_dashboard.html'}:
                base_bonus = 5  # High priority for main app files
            elif filename.startswith('step') and filename.endswith('.py'):
                base_bonus = 10  # Highest priority for calculation step files
            elif filename in {'final_risk_assessment_leo.csv', 'collision_scores_leo.csv', 'environment_scores_leo.csv', 'bors_scores_leo.csv'}:
                base_bonus = 3  # Medium priority for data files
            total = score + base_bonus
            if full_mode:
                candidates.append((max(total, 1), path))
            elif total > 0:
                candidates.append((total, path))
        except Exception:
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates if full_mode else candidates[:max_files]
    sources = [p for _, p in top]

    sections = []
    sections.append("PROJECT OVERVIEW\n-----\n" + generate_project_overview())
    if include_manifest:
        if full_mode:
            sections.append("PROJECT MANIFEST (all files)\n-----\n" + _file_manifest(all_paths))
        else:
            sections.append("MATCHED FILES\n-----\n" + _file_manifest(sources))

    total_chars = 0
    for p in sources:
        ext = os.path.splitext(p)[1].lower()
        if ext in CSV_EXTS:
            content = _safe_read_csv_head(p)
        else:
            content = _safe_read_text(p)
        snippet = content.strip()
        if total_chars + len(snippet) > MAX_TOTAL_CONTEXT_CHARS:
            remain = max(0, MAX_TOTAL_CONTEXT_CHARS - total_chars)
            snippet = snippet[:remain]
            sections.append(f"FILE: {p}\n-----\n{snippet}\n... [truncated]\n")
            total_chars += len(snippet)
            break
        else:
            sections.append(f"FILE: {p}\n-----\n{snippet}\n")
            total_chars += len(snippet)

    context = "\n\n".join(sections)
    return context, sources

def call_openai(messages: List[dict], model: str = None) -> str:
    """Call OpenAI chat completion with provided messages. Requires OPENAI_API_KEY."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. Please `pip install openai`." )
    # Default to a lightweight, cost-effective model
    model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    client = OpenAI()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise

def calculate_collision_score(altitude_km, inclination, age_years):
    """Estimate collision score based on altitude and orbital params"""

    # Base score from altitude (higher altitudes = more congestion in LEO)
    if altitude_km < 400:
        base_score = 45.0  # Low orbits less congested (atmosphere clears debris)
    elif altitude_km < 600:
        base_score = 56.9  # Fleet average
    elif altitude_km < 1000:
        base_score = 65.0  # High congestion zone
    else:
        base_score = 55.0  # Very high LEO

    # Inclination adjustment (high inclination orbits cross more trajectories)
    if inclination > 70:
        base_score += 10.0  # Polar orbits cross more paths
    elif inclination < 10:
        base_score -= 5.0  # Equatorial less crowded

    # Age adjustment (older satellites may have outdated tracking)
    if age_years > 10:
        base_score += 5.0

    return min(100, max(15, base_score))

def calculate_environment_score(altitude_km, age_years):
    """Estimate environment risk based on altitude"""

    # Base score (fleet average P95 of space weather)
    base_score = 64.8

    # Altitude adjustment (lower orbits have more drag risk)
    if altitude_km < 400:
        altitude_factor = 1.20  # +20% risk
    elif altitude_km < 600:
        altitude_factor = 1.00  # Baseline
    elif altitude_km < 1000:
        altitude_factor = 0.90  # -10% risk
    else:
        altitude_factor = 0.75  # -25% risk

    adjusted_score = base_score * (0.60 + 0.40 * altitude_factor)

    return min(100, max(40, adjusted_score))

def calculate_bors_score(age_years, expected_lifetime, mass_kg, contractor_tier, operator_tier, purpose):
    """Calculate BORS using 4-pillar model"""

    # 1. Age Factor (30%)
    age_factor = min(age_years / expected_lifetime, 1.0)

    # 2. TRI - Technical Reliability Index (40%)
    # CQI - Component Quality Index (mass-based)
    if mass_kg < 100:
        cqi = 0.7  # COTS components
    elif mass_kg < 500:
        cqi = 0.5
    else:
        cqi = 0.3  # Space-grade

    # HI - Heritage Index (contractor tier)
    hi_map = {1: 0.2, 2: 0.5, 3: 0.8}
    hi = hi_map.get(contractor_tier, 0.8)

    # MRI - Mass Reliability Index
    if mass_kg < 500:
        mri = 0.7
    elif mass_kg < 2500:
        mri = 0.3  # Sweet spot
    else:
        mri = 0.6

    # SVI - Subsystem Vulnerability Index
    svi_map = {
        'Communications': 0.8,
        'Earth Observation': 0.6,
        'Technology Demo': 0.5,
        'Science': 0.4,
        'Navigation': 0.5
    }
    svi = svi_map.get(purpose, 0.6)

    # TRI calculation
    tri = 0.25 * cqi + 0.40 * hi + 0.20 * mri + 0.15 * svi

    # 3. Operator Heritage Index (15%)
    op_map = {1: 0.1, 2: 0.4, 3: 0.7}
    operator_index = op_map.get(operator_tier, 0.7)

    # 4. Infant Mortality (15%)
    infant_flag = 0.8 if age_years < 1 else 0.1

    # Final BORS
    raw_score = 0.30 * age_factor + 0.40 * tri + 0.15 * operator_index + 0.15 * infant_flag
    bors_score = raw_score * 100

    return min(100, max(25, bors_score))

def calculate_premium(total_risk_score, sum_insured):
    """Calculate insurance premium from risk score"""

    # Map score to annual loss probability
    if total_risk_score < 30:
        p_loss_annual = 0.005  # 0.5%
    elif total_risk_score < 50:
        p_loss_annual = 0.015  # 1.5%
    elif total_risk_score < 70:
        p_loss_annual = 0.035  # 3.5%
    elif total_risk_score < 85:
        p_loss_annual = 0.070  # 7.0%
    else:
        p_loss_annual = 0.150  # 15.0%

    # Calculate premium
    base_premium = sum_insured * p_loss_annual
    load_factor = 1.55  # 30% expenses + 15% profit + 10% buffer
    premium_annual = base_premium * load_factor

    rate_on_line = (premium_annual / sum_insured) * 100

    return {
        'premium_annual': premium_annual,
        'rate_on_line_pct': rate_on_line,
        'p_loss_annual': p_loss_annual * 100,
        'expected_loss': base_premium
    }

def apply_scenario(collision, environment, bors, scenario_name):
    """Apply scenario adjustments"""

    scenarios = {
        'Baseline': {'collision_mult': 1.0, 'environment_mult': 1.0, 'bors_mult': 1.0},
        'Solar Storm': {'collision_mult': 1.0, 'environment_mult': 1.30, 'bors_mult': 1.0},
        'Mega-Constellation Era': {'collision_mult': 1.50, 'environment_mult': 1.0, 'bors_mult': 1.0},
        'Solar Minimum': {'collision_mult': 1.0, 'environment_mult': 0.75, 'bors_mult': 1.0},
        'Active Debris Removal': {'collision_mult': 0.70, 'environment_mult': 1.0, 'bors_mult': 1.0}
    }

    scenario = scenarios.get(scenario_name, scenarios['Baseline'])

    return {
        'collision_score': min(100, collision * scenario['collision_mult']),
        'environment_score': min(100, environment * scenario['environment_mult']),
        'bors_score': min(100, bors * scenario['bors_mult'])
    }

@app.route('/')
def home():
    """Serve the interactive calculator"""
    return send_from_directory('.', 'risk_calculator.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_risk():
    """Calculate risk scores based on satellite parameters"""

    try:
        data = request.json

        # Extract parameters
        altitude_km = float(data.get('altitude_km', 550))
        inclination = float(data.get('inclination', 53))
        mass_kg = float(data.get('mass_kg', 500))
        age_years = float(data.get('age_years', 3))
        expected_lifetime = float(data.get('expected_lifetime', 5))
        contractor_tier = int(data.get('contractor_tier', 3))
        operator_tier = int(data.get('operator_tier', 3))
        purpose = data.get('purpose', 'Communications')
        sum_insured = float(data.get('sum_insured', 10000000))
        scenario_name = data.get('scenario', 'Baseline')

        # Calculate individual risk scores
        collision_score = calculate_collision_score(altitude_km, inclination, age_years)
        environment_score = calculate_environment_score(altitude_km, age_years)
        bors_score = calculate_bors_score(age_years, expected_lifetime, mass_kg,
                                          contractor_tier, operator_tier, purpose)

        # Apply scenario
        adjusted = apply_scenario(collision_score, environment_score, bors_score, scenario_name)

        # Calculate total risk (weighted combination)
        total_risk = (
            0.40 * adjusted['collision_score'] +
            0.30 * adjusted['environment_score'] +
            0.30 * adjusted['bors_score']
        )

        # Calculate premium
        pricing = calculate_premium(total_risk, sum_insured)

        # Risk category
        if total_risk < 30:
            risk_category = 'Low'
        elif total_risk < 45:
            risk_category = 'Medium-Low'
        elif total_risk < 65:
            risk_category = 'Medium-High'
        elif total_risk < 85:
            risk_category = 'High'
        else:
            risk_category = 'Critical'

        # Compare to fleet
        fleet_comparison = {
            'vs_fleet_avg': round(total_risk - models['fleet_stats']['mean_total_risk'], 1),
            'percentile': round(stats.percentileofscore(df['total_risk_score'], total_risk), 1),
            'fleet_avg': models['fleet_stats']['mean_total_risk']
        }

        result = {
            'success': True,
            'baseline_scores': {
                'collision': round(collision_score, 1),
                'environment': round(environment_score, 1),
                'bors': round(bors_score, 1)
            },
            'adjusted_scores': {
                'collision': round(adjusted['collision_score'], 1),
                'environment': round(adjusted['environment_score'], 1),
                'bors': round(adjusted['bors_score'], 1)
            },
            'total_risk_score': round(total_risk, 1),
            'risk_category': risk_category,
            'pricing': {
                'premium_annual': round(pricing['premium_annual'], 2),
                'rate_on_line_pct': round(pricing['rate_on_line_pct'], 2),
                'p_loss_annual_pct': round(pricing['p_loss_annual'], 2),
                'expected_loss': round(pricing['expected_loss'], 2)
            },
            'fleet_comparison': fleet_comparison,
            'scenario_applied': scenario_name
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/scenarios')
def get_scenarios():
    """Get available scenarios"""
    return jsonify({
        'scenarios': [
            'Baseline',
            'Solar Storm',
            'Mega-Constellation Era',
            'Solar Minimum',
            'Active Debris Removal'
        ]
    })

@app.route('/api/fleet_stats')
def get_fleet_stats():
    """Get fleet statistics for comparison"""
    return jsonify(models['fleet_stats'])


@app.route('/api/chat', methods=['POST'])
def chat_with_repo():
    """OpenAI-powered chat about the local project files and data.

    Expects JSON: { messages: [{role, content}, ...], max_context_files?: int }
    Returns: { success, answer, sources }
    """
    try:
        data = request.json or {}
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'success': False, 'error': 'No messages provided'}), 400

        # Extract the latest user question
        user_msgs = [m for m in messages if m.get('role') == 'user']
        question = user_msgs[-1]['content'] if user_msgs else ''
        if not question:
            return jsonify({'success': False, 'error': 'Missing user question'}), 400

        max_files = int(data.get('max_context_files', 5))
        full_mode = str(data.get('mode', '')).lower() == 'full'
        context, sources = build_repo_context(question, max_files=max_files, include_manifest=True, full_mode=full_mode)

        # Build chat prompt
        system_prompt = (
            "You are an assistant for the Satellite Risk MVP project. "
            "Answer questions using ONLY the provided repository context when citing details. "
            "Be concise and specific, and include file names when helpful. "
            "If something is not in the context, say you don't see it."
        )

        openai_messages = [
            { 'role': 'system', 'content': system_prompt },
            { 'role': 'system', 'content': f"Repository context below. Use it for answering.\n\n{context}" },
        ]
        # Optionally include prior conversation for continuity
        # but not to exceed token limits; we include last 4 messages
        tail = messages[-4:]
        for m in tail:
            if m.get('role') in {'user', 'assistant'}:
                openai_messages.append({'role': m['role'], 'content': m.get('content', '')})

        # Ensure API key is present
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'OPENAI_API_KEY not set. Please export your OpenAI API key in the environment.',
                'sources': sources
            }), 400

        answer = call_openai(openai_messages)
        return jsonify({'success': True, 'answer': answer, 'sources': sources})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc().splitlines()[-1:]}), 500

if __name__ == '__main__':
    print("="*60)
    print("🛰️  Satellite Risk Calculator Backend")
    print("="*60)
    print("Server starting at http://localhost:5555")
    print("Access the calculator at: http://localhost:5555/")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5555)
