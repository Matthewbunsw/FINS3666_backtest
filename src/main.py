"""
FINS3666 High Grade Copper Futures (HG) Trading Strategy Backtest

This backtest implements a 3-factor regression model for copper futures trading
with carry-based position sizing and roll management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
import sys
warnings.filterwarnings('ignore')


# ============================================================================
# OUTPUT LOGGING
# ============================================================================

class ConsoleLogger:
    """Logs output to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Import configuration
from config import (
    SignalType,
    PositionStatus,
    SIGNAL_THRESHOLD,
    REGRESSION_WINDOW_DAYS,
    ATR_PERIOD,
    ATR_STOP_MULTIPLIER,
    RISK_PERCENT,
    CARRY_THRESHOLD_HALF,
    CARRY_THRESHOLD_QUARTER,
    DAYS_BEFORE_FND_TO_ROLL,
    BACKTEST_START,
    BACKTEST_END,
    INITIAL_EQUITY,
    DXY_FILE,
    PMI_FILE,
    HG_FRONT_FILE,
    LME_STOCKS_FILE
)

# Import data models for storing trades, positions, rolls, daily metrics, and results
from logging_structs import (
    Trade,
    Position,
    RollEvent,
    DailyMetrics,
    BacktestResults
)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(start_date='2021-01-04', end_date='2025-10-31'):
    """
    Load all market data and preprocess for backtesting
    
    Returns:
        pd.DataFrame: Merged dataset with all features
    """
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # ========================================================================
    # 1. Load DXY (US Dollar Index)
    # ========================================================================
    print("\n[1/4] Loading DXY data...")
    dxy = pd.read_csv(DXY_FILE)
    dxy['Date'] = pd.to_datetime(dxy['Date'], format='%m/%d/%y')
    dxy = dxy.sort_values('Date').reset_index(drop=True)
    
    # Calculate log-difference × 100
    dxy['DXY_logret'] = np.log(dxy['Price'] / dxy['Price'].shift(1)) * 100
    dxy = dxy[['Date', 'Price', 'DXY_logret']].rename(columns={'Price': 'DXY_Price'})
    print(f"   ✓ Loaded {len(dxy)} DXY observations")
    print(f"   ✓ Date range: {dxy['Date'].min().date()} to {dxy['Date'].max().date()}")
    
    # ========================================================================
    # 2. Load China PMI (Monthly) and interpolate to daily
    # ========================================================================
    print("\n[2/4] Loading PMI data...")
    pmi = pd.read_csv(PMI_FILE)
    pmi['Date'] = pd.to_datetime(pmi['Date'], format='%m/%d/%Y')
    pmi = pmi.sort_values('Date').reset_index(drop=True)
    pmi = pmi.rename(columns={'PMI Manufacturing *': 'PMI'})
    print(f"   ✓ Loaded {len(pmi)} monthly PMI observations")
    
    # Interpolate to daily frequency
    print("   ⟳ Interpolating monthly PMI to daily...")
    pmi_daily = pmi.set_index('Date').resample('D').interpolate(method='linear')
    pmi_daily = pmi_daily.reset_index()
    
    # Calculate level change (today - yesterday)
    pmi_daily['PMI_change'] = pmi_daily['PMI'].diff()
    print(f"   ✓ Interpolated to {len(pmi_daily)} daily observations")
    
    # ========================================================================
    # 3. Load HG Front Continuous (Settlement Price)
    # ========================================================================
    print("\n[3/4] Loading HG Copper futures data...")
    hg = pd.read_csv(HG_FRONT_FILE)
    hg['Date'] = pd.to_datetime(hg['Date'], format='%m/%d/%y')
    hg = hg.sort_values('Date').reset_index(drop=True)
    
    # Calculate log-difference × 100 (this is our target variable HG1)
    hg['HG1_logret'] = np.log(hg['Settlement Price'] / hg['Settlement Price'].shift(1)) * 100
    hg = hg[['Date', 'Settlement Price', 'HG1_logret']].rename(columns={'Settlement Price': 'HG_Price'})
    print(f"   ✓ Loaded {len(hg)} HG observations")
    print(f"   ✓ Date range: {hg['Date'].min().date()} to {hg['Date'].max().date()}")
    
    # ========================================================================
    # 4. Load LME Copper Stocks
    # ========================================================================
    print("\n[4/4] Loading LME copper stocks data...")
    lme = pd.read_csv(LME_STOCKS_FILE)
    lme['Date'] = pd.to_datetime(lme['Date'], format='%d/%m/%Y')
    lme = lme.sort_values('Date').reset_index(drop=True)
    
    # Calculate log-difference × 100
    lme['STOCKS_logret'] = np.log(
        lme['LME Copper Stock Level (tons) - Close'] / 
        lme['LME Copper Stock Level (tons) - Close'].shift(1)
    ) * 100
    lme = lme[['Date', 'LME Copper Stock Level (tons) - Close', 'STOCKS_logret']]
    lme = lme.rename(columns={'LME Copper Stock Level (tons) - Close': 'LME_Stocks'})
    print(f"   ✓ Loaded {len(lme)} LME observations")
    print(f"   ✓ Date range: {lme['Date'].min().date()} to {lme['Date'].max().date()}")
    
    # ========================================================================
    # 5. Merge all datasets
    # ========================================================================
    print("\n[5/5] Merging all datasets...")
    data = hg.merge(dxy, on='Date', how='inner')
    data = data.merge(pmi_daily, on='Date', how='inner')
    data = data.merge(lme, on='Date', how='inner')
    
    # Drop rows with NaN (from log-difference calculations)
    data = data.dropna().reset_index(drop=True)
    
    # Filter to backtest period
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data = data.reset_index(drop=True)
    
    print(f"   ✓ Merged dataset: {len(data)} observations")
    print(f"   ✓ Backtest period: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"   ✓ Features: {list(data.columns)}")
    
    # Data quality check
    print("\n[DATA QUALITY CHECK]")
    print(f"   Missing values per column:")
    for col in data.columns:
        missing = data[col].isna().sum()
        print(f"      {col}: {missing}")
    
    print("\n" + "="*80)
    return data


# ============================================================================
# SIGNAL GENERATION (3-FACTOR REGRESSION MODEL)
# ============================================================================

def generate_signals(data, backtest_start, window_days=504):
    """
    Generate trading signals using rolling 3-factor regression
    
    Model: HG1_t = α + β₁·DXY_t + β₂·ΔPMI_t + β₃·STOCKS_t + ε_t
    
    Args:
        data: DataFrame with all features (includes training period)
        backtest_start: Date when backtest starts (signals generated from here)
        window_days: Rolling window for regression (default: 504 days = 2 years)
    
    Returns:
        DataFrame with signals (only backtest period)
    """
    print("\n" + "="*80)
    print("GENERATING TRADING SIGNALS (3-FACTOR REGRESSION)")
    print("="*80)
    print(f"\nRegression window: {window_days} days (~{window_days/252:.1f} years)")
    print(f"Signal threshold: ±{SIGNAL_THRESHOLD}%")
    print(f"Generating signals from: {backtest_start}")
    
    # Prepare features and target
    feature_cols = ['DXY_logret', 'PMI_change', 'STOCKS_logret']
    target_col = 'HG1_logret'
    
    # Initialize columns for results
    data['Forecasted_Return'] = np.nan
    data['Signal'] = None  # Use None to distinguish unprocessed rows
    data['Model_Alpha'] = np.nan
    data['Model_Beta_DXY'] = np.nan
    data['Model_Beta_PMI'] = np.nan
    data['Model_Beta_STOCKS'] = np.nan
    data['Model_R2'] = np.nan
    
    # Find index where backtest starts
    backtest_start_idx = data[data['Date'] >= backtest_start].index[0]
    
    print(f"\nTraining period ends at index: {backtest_start_idx - 1} ({data.loc[backtest_start_idx - 1, 'Date'].date()})")
    print(f"Signal generation starts at index: {backtest_start_idx} ({data.loc[backtest_start_idx, 'Date'].date()})")
    print(f"Total signals to generate: {len(data) - backtest_start_idx}")
    
    # Counters for accurate signal distribution
    long_signals = 0
    short_signals = 0
    neutral_signals = 0
    
    # Rolling regression - only generate signals for backtest period
    print(f"\nRunning rolling regression...")
    
    for i in range(backtest_start_idx, len(data)):
        # Check if we have enough history
        if i < window_days:
            continue
        
        # Training window: last 'window_days' observations before today
        train_idx = range(i - window_days, i)
        X_train = data.loc[train_idx, feature_cols]
        y_train = data.loc[train_idx, target_col]
        
        # Skip if any NaN in training data
        if X_train.isna().any().any() or y_train.isna().any():
            continue
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store model coefficients
        data.loc[i, 'Model_Alpha'] = model.intercept_
        data.loc[i, 'Model_Beta_DXY'] = model.coef_[0]
        data.loc[i, 'Model_Beta_PMI'] = model.coef_[1]
        data.loc[i, 'Model_Beta_STOCKS'] = model.coef_[2]
        data.loc[i, 'Model_R2'] = model.score(X_train, y_train)
        
        # Generate forecast for today using today's features
        X_today = data.loc[i, feature_cols].values.reshape(1, -1)
        forecast = model.predict(X_today)[0]
        data.loc[i, 'Forecasted_Return'] = forecast
        
        # Generate signal based on threshold
        if forecast > SIGNAL_THRESHOLD:
            signal = SignalType.LONG.value
            long_signals += 1
        elif forecast < -SIGNAL_THRESHOLD:
            signal = SignalType.SHORT.value
            short_signals += 1
        else:
            signal = SignalType.NEUTRAL.value
            neutral_signals += 1
        
        data.loc[i, 'Signal'] = signal
        
        # Progress indicator
        if (i - backtest_start_idx) % 50 == 0:
            pct = ((i - backtest_start_idx + 1) / (len(data) - backtest_start_idx)) * 100
            print(f"   Progress: {pct:.1f}% ({i - backtest_start_idx + 1}/{len(data) - backtest_start_idx} days)", end='\r')
    
    signals_generated = long_signals + short_signals + neutral_signals
    print(f"\n   ✓ Generated {signals_generated} trading signals")
    
    # Filter data to only backtest period with signals
    backtest_data = data[data['Date'] >= backtest_start].copy()
    backtest_data = backtest_data[backtest_data['Signal'].notna()].reset_index(drop=True)
    
    # Signal statistics - NOW ACCURATE
    print(f"\n[SIGNAL DISTRIBUTION]")
    print(f"   LONG:    {long_signals:4d} ({long_signals/signals_generated*100:5.1f}%)")
    print(f"   SHORT:   {short_signals:4d} ({short_signals/signals_generated*100:5.1f}%)")
    print(f"   NEUTRAL: {neutral_signals:4d} ({neutral_signals/signals_generated*100:5.1f}%)")
    print(f"   TOTAL:   {signals_generated:4d} (100.0%)")
    
    # Model performance statistics
    print(f"\n[MODEL STATISTICS]")
    print(f"   Average R²: {backtest_data['Model_R2'].mean():.4f}")
    print(f"   Median R²:  {backtest_data['Model_R2'].median():.4f}")
    print(f"   Average |Forecast|: {backtest_data['Forecasted_Return'].abs().mean():.4f}%")
    print(f"   Max forecast: {backtest_data['Forecasted_Return'].max():+.4f}%")
    print(f"   Min forecast: {backtest_data['Forecasted_Return'].min():+.4f}%")
    
    # Beta coefficients summary
    print(f"\n[AVERAGE BETA COEFFICIENTS]")
    print(f"   β₁ (DXY):    {backtest_data['Model_Beta_DXY'].mean():+.6f}")
    print(f"   β₂ (PMI):    {backtest_data['Model_Beta_PMI'].mean():+.6f}")
    print(f"   β₃ (STOCKS): {backtest_data['Model_Beta_STOCKS'].mean():+.6f}")
    
    # Interpretation
    print(f"\n[INTERPRETATION]")
    if backtest_data['Model_Beta_DXY'].mean() < 0:
        print(f"   ✓ Copper falls when USD strengthens (β₁ < 0)")
    if backtest_data['Model_Beta_PMI'].mean() > 0:
        print(f"   ✓ Copper rises with Chinese PMI (β₂ > 0)")
    if backtest_data['Model_Beta_STOCKS'].mean() < 0:
        print(f"   ✓ Copper falls when inventories rise (β₃ < 0)")
    
    print("="*80)
    return backtest_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set up logging to file
    log_filename = f'output/script_output.txt'
    sys.stdout = ConsoleLogger(log_filename)
    
    print("\n")
    print("="*80)
    print(" " * 15 + "FINS3666 - HG COPPER FUTURES BACKTEST")
    print(" " * 20 + "3-Factor Regression Strategy")
    print("="*80)
    print(f"\nOutput is being saved to: {log_filename}")
    
    
    # Calculate how far back we need to load data
    # Need 504 trading days (~2 years) before backtest start
    # 504 trading days ≈ 720 calendar days (accounting for weekends/holidays)
    backtest_start_date = pd.to_datetime(BACKTEST_START)
    data_start_date = backtest_start_date - timedelta(days=750)  # Extra buffer
    DATA_START = data_start_date.strftime('%Y-%m-%d')
    
    print(f"\nBacktest Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"Data Loading Period: {DATA_START} to {BACKTEST_END}")
    print(f"  (Need ~2 years of history for rolling regression)")
    print(f"Initial Equity: ${INITIAL_EQUITY:,.0f}")
    
    # Step 1: Load and preprocess data (includes training period)
    data = load_and_preprocess_data(start_date=DATA_START, end_date=BACKTEST_END)
    
    # Verify we have enough data
    days_available = len(data)
    print(f"\n[DATA AVAILABILITY CHECK]")
    print(f"   Days loaded: {days_available}")
    print(f"   Days needed for regression: {REGRESSION_WINDOW_DAYS}")
    if days_available >= REGRESSION_WINDOW_DAYS:
        print(f"   ✓ Sufficient data available")
    else:
        print(f"   ⚠ WARNING: Insufficient data! Need {REGRESSION_WINDOW_DAYS - days_available} more days")
    
    # Step 2: Generate trading signals (only for backtest period)
    backtest_data = generate_signals(data, backtest_start=BACKTEST_START, window_days=REGRESSION_WINDOW_DAYS)
    
    # Save processed data for inspection
    output_file = 'output/backtest_signals.csv'
    backtest_data.to_csv(output_file, index=False)
    print(f"\n✓ Backtest data with signals saved to: {output_file}")
    
    # Display sample of signals
    print("\n" + "="*80)
    print("SAMPLE OF GENERATED SIGNALS")
    print("="*80)
    
    # Show first 5 signals
    print("\n[FIRST 5 TRADING DAYS]")
    print(backtest_data.head(5)[['Date', 'HG_Price', 'Forecasted_Return', 'Signal', 'Model_R2']].to_string(index=False))
    
    # Show last 5 signals
    print("\n[LAST 5 TRADING DAYS]")
    print(backtest_data.tail(5)[['Date', 'HG_Price', 'Forecasted_Return', 'Signal', 'Model_R2']].to_string(index=False))
    
    # Show some non-neutral signals
    print("\n[RECENT NON-NEUTRAL SIGNALS]")
    non_neutral = backtest_data[backtest_data['Signal'] != SignalType.NEUTRAL.value].tail(10)
    if len(non_neutral) > 0:
        print(non_neutral[['Date', 'HG_Price', 'Forecasted_Return', 'Signal', 'Model_R2']].to_string(index=False))
    else:
        print("   No non-neutral signals found")
    
    print("\n" + "="*80)
    print("DATA LOADING AND SIGNAL GENERATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. ✓ Load and preprocess market data")
    print("2. ✓ Implement signal generation (3-factor regression)")
    print("3. ⧗ Implement position sizing with carry filter")
    print("4. ⧗ Implement risk management (ATR stops)")
    print("5. ⧗ Run backtest simulation")
    print("6. ⧗ Analyze results")
    print("\n")
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✓ Output saved to: {log_filename}")
