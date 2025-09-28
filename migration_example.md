# Data Loading Utilities Migration Example

This document shows how the new Data Loading Utilities can replace duplicate code patterns across notebooks.

## Before: Traditional Approach (Original Notebook Code)

Here's the typical data loading pattern found in `01a_data_loading_and_overview.ipynb`:

```python
# Traditional approach - multiple steps, manual processing
from core.data_manager import SolarDataManager
from core.data_source_detector import DataSourceDetector
from core.location_loader import create_notebook_location

# Initialize location from .env (with fallback)
location = create_notebook_location()
print(f"🌍 Location configured: {location.location_name}")

# Initialize the smart data source detector
detector = DataSourceDetector(location=location)

# Determine the optimal data loading strategy
strategy = detector.determine_data_strategy()

# Initialize data manager with the detected strategy
data_manager = SolarDataManager(
    csv_path=strategy['csv_path'],
    enphase_client=strategy['client'],
    cache_dir="../data/processed"
)

# Load CSV data
csv_data = data_manager.load_csv_data()

# Get data summary and analysis
data_summary = data_manager.get_data_summary()
recency_info = detector.analyze_data_recency(csv_data, data_summary)

# Generate report
detector.generate_final_report(strategy, data_summary, recency_info)

# Get daily data
daily_data = data_manager.get_daily_production(source_priority="csv_first")

# Manual data quality checks
print(f"Dataset shape: {csv_data.shape}")
print(f"Date range: {csv_data.index.min()} to {csv_data.index.max()}")
print(f"Missing values:")
print(csv_data.isnull().sum())

# Manual statistics
print(f"Data spans {(csv_data.index.max() - csv_data.index.min()).days} days")
expected_intervals = (csv_data.index.max() - csv_data.index.min()).days * 96
print(f"Data completeness: {len(csv_data) / expected_intervals * 100:.1f}%")
```

**Issues with this approach:**
- **15+ lines** of setup code
- **Manual data quality checks** (inconsistent across notebooks)
- **Duplicate setup patterns** in every notebook
- **No standardized reporting**

## After: New Data Loading Utilities

### Option 1: Ultra-Simple Loading

```python
# NEW: Ultra-simple approach - one line for most use cases
from core.notebook_data import quick_load_solar_data

# Load data with automatic detection and quality validation
fifteen_min_data, daily_data = quick_load_solar_data(source='smart')

print(f"✅ Loaded {len(fifteen_min_data):,} 15-min records and {len(daily_data)} daily records")
```

**Benefits:**
- **1 line** replaces 15+ lines
- **Automatic quality validation**
- **Consistent behavior** across notebooks

### Option 2: With Quality Report

```python
# NEW: Load with comprehensive quality analysis
from core.notebook_data import load_with_quality_report

result = load_with_quality_report(source='smart')

fifteen_min_data = result['fifteen_min_data']
daily_data = result['daily_data']

# Professional quality report automatically generated
print(result['quality_report'])
```

**Benefits:**
- **Standardized quality reporting**
- **Professional formatting**
- **Comprehensive validation**

### Option 3: Complete Dataset Setup

```python
# NEW: Setup all standard datasets at once
from core.notebook_data import setup_standard_datasets

datasets = setup_standard_datasets(source='smart', include_ml_features=True)

fifteen_min_data = datasets['fifteen_min']
daily_data = datasets['daily']
monthly_data = datasets['monthly']
daily_with_metrics = datasets['daily_with_metrics']
ml_ready_data = datasets['ml_ready']  # If ML features requested

print(f"✅ Created {len(datasets)} standard datasets")
```

**Benefits:**
- **All datasets created at once**
- **Consistent processing**
- **ML-ready features included**

### Option 4: Enhanced Data Manager Integration

```python
# NEW: Enhanced SolarDataManager with new utilities
from core.data_manager import SolarDataManager
from core.data_source_detector import DataSourceDetector
from core.location_loader import create_notebook_location

# Traditional setup (unchanged for compatibility)
location = create_notebook_location()
detector = DataSourceDetector(location=location)
strategy = detector.determine_data_strategy()

data_manager = SolarDataManager(
    csv_path=strategy['csv_path'],
    enphase_client=strategy['client']
)

# NEW: Enhanced methods using utilities
csv_data = data_manager.load_csv_data(validate_quality=True)  # Auto-validation
quality_report = data_manager.get_data_quality_report('csv')  # Professional report
daily_data = data_manager.create_enhanced_daily_summary(include_metrics=True)  # Rich metrics
ml_dataset = data_manager.prepare_ml_dataset()  # ML-ready features

print(quality_report)  # Comprehensive quality analysis
```

**Benefits:**
- **Backward compatible** with existing code
- **Enhanced functionality** using new utilities
- **Professional quality reporting**
- **ML-ready dataset preparation**

## Code Reduction Summary

| Approach | Original Lines | New Lines | Reduction |
|----------|----------------|-----------|-----------|
| Ultra-simple | 15+ lines | 1 line | **93% reduction** |
| With quality | 20+ lines | 3 lines | **85% reduction** |
| Complete setup | 25+ lines | 3 lines | **88% reduction** |
| Enhanced manager | 25+ lines | 8 lines | **68% reduction** |

## Migration Strategy

### Phase 1: New Notebooks
- Use new utilities for all new analysis notebooks
- Start with `quick_load_solar_data()` for simplicity

### Phase 2: Existing Notebooks
- Add new utility calls alongside existing code
- Gradually replace manual patterns
- Maintain backward compatibility

### Phase 3: Full Migration
- Replace all duplicate loading patterns
- Standardize quality reporting
- Leverage ML-ready datasets

## Example: Migrated Notebook Cell

```python
# 🆕 NEW APPROACH: Single cell replacing 3-4 original cells
from core.notebook_data import load_with_quality_report, print_data_overview

# Load data with comprehensive analysis
result = load_with_quality_report(source='smart')
fifteen_min_data = result['fifteen_min_data']
daily_data = result['daily_data']

# Professional data overview
print_data_overview(fifteen_min_data, "15-Minute Solar Data")
print_data_overview(daily_data, "Daily Solar Summary")

# Comprehensive quality report
print(result['quality_report'])

print("✅ Data loading and quality analysis complete!")
```

This **single cell** replaces the original **4 cells** (setup, loading, quality checks, statistics) while providing **better functionality** and **consistent reporting**.