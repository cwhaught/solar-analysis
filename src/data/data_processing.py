import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_and_process_enphase_data(filename="4136754_custom_report.csv"):
    """Load and process Enphase data with proper datetime handling"""
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "raw" / filename

    # Load data
    df = pd.read_csv(file_path)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df.set_index("Date/Time", inplace=True)

    # Convert to kWh
    df_kwh = df / 1000
    df_kwh.columns = [
        "Production (kWh)",
        "Consumption (kWh)",
        "Export (kWh)",
        "Import (kWh)",
    ]

    return df_kwh


def create_daily_summary(df_kwh):
    """Create daily aggregated data"""
    return df_kwh.resample("D").sum()
