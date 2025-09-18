from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


def prepare_ml_features(daily_data):
    """Prepare features for ML models"""
    daily_features = daily_data.copy()

    # Time-based features
    daily_features['month'] = daily_features.index.month
    daily_features['day_of_year'] = daily_features.index.dayofyear
    daily_features['day_of_week'] = daily_features.index.dayofweek
    daily_features['quarter'] = daily_features.index.quarter

    # Historical features
    daily_features['prod_yesterday'] = daily_features['Production (kWh)'].shift(1)
    daily_features['prod_2days_ago'] = daily_features['Production (kWh)'].shift(2)
    daily_features['prod_week_ago'] = daily_features['Production (kWh)'].shift(7)
    daily_features['prod_3day_std'] = daily_features['Production (kWh)'].rolling(3).std()
    daily_features['prod_7day_avg'] = daily_features['Production (kWh)'].rolling(7).mean()

    return daily_features.dropna()


def train_production_model(daily_features):
    """Train solar production forecasting model"""
    feature_columns = [
        'month', 'day_of_year', 'day_of_week', 'quarter',
        'prod_yesterday', 'prod_2days_ago', 'prod_week_ago',
        'prod_3day_std', 'prod_7day_avg'
    ]

    X = daily_features[feature_columns]
    y = daily_features['Production (kWh)']

    # Train/test split (last 6 months for testing)
    split_date = daily_features.index[-180]
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    # Train model
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, {'mae': mae, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred}