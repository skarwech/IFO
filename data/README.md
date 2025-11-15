# Data Directory

This directory should contain the following files for the Intelligent Pump Scheduler:

## Required Files

### 1. Historical Data
**File**: `Hackathon_HSY_data.xlsx`

14-day time series with 15-minute intervals containing:
- Time stamp
- Water level in tunnel L1 (m)
- Water volume in tunnel V (m³)
- Sum of pumped flow to WWTP F2 (m³/h)
- Inflow to tunnel F1 (m³/15 min)
- Individual pump flows (m³/h)
- Individual pump power consumption (kW)
- Individual pump frequencies (Hz)
- Electricity price(s) (EUR/kWh)

### 2. Volume Table
**File**: `Volume of tunnel vs level Blominmäki.xlsx`

Piecewise function data for V = f(L1):
- Column 1: L1 (water level in meters)
- Column 2: V (volume in m³)

Used for accurate volume-level conversion based on tunnel geometry.

### 3. Pump Curves (Optional)
**Files**: 
- `Pumppukäyrä_pienet.PDF` - Small pumps performance curves
- `Pumppukäyrä_suuret.PDF` - Large pumps performance curves
- `Pumppukäyrä2_pienet.pdf` - Small pumps additional curves
- `Pumppukäyrä2_suuret.pdf` - Large pumps additional curves

Performance curves showing:
- Flow (Q) vs Head (H)
- Flow (Q) vs Power (P)
- Flow (Q) vs Efficiency (η)
- NPSHr requirements
- Variable frequency operation

### 4. Documentation (Optional)
**File**: `BLOM tulotunnelin tilavuuden laskenta pintamittauksen perusteella.docx`

Documentation of tunnel volume calculation methodology.

## Using Sample Data

If real data files are not available, the system can generate synthetic sample data:

```python
from src import create_sample_data

# Generate 14 days of synthetic data
data = create_sample_data(num_days=14)
```

Or use the command-line flag:

```bash
python main.py --mode simulate --use-sample-data --days 14
```

## Data Privacy

**Important**: Do not commit real operational data to version control. 

Add data files to `.gitignore`:
```
data/*.xlsx
data/*.pdf
data/*.docx
```

## Directory Structure

```
data/
├── README.md                          # This file
├── Hackathon_HSY_data.xlsx           # Historical data (not in git)
├── Volume of tunnel vs level Blominmäki.xlsx  # Volume table (not in git)
├── Pumppukäyrä_pienet.PDF            # Small pump curves (not in git)
├── Pumppukäyrä_suuret.PDF            # Large pump curves (not in git)
└── sample_data_generated.csv         # Auto-generated sample (git-ok)
```
