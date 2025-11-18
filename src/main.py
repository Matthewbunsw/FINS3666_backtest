"""
FINS3666 High Grade Copper Futures (HG) Trading Strategy Backtest

This backtest implements a 3-factor regression model for copper futures trading
with carry-based position sizing and roll management.

AI Use Declaration:
- Claude and ChatGPT models were used to assist in code generation and debugging.
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
    CONTRACT_SIZE,
    SIGNAL_THRESHOLD,
    REGRESSION_WINDOW_DAYS,
    ATR_PERIOD,
    ATR_STOP_MULTIPLIER,
    RISK_PERCENT,
    CARRY_THRESHOLD_HALF,
    CARRY_THRESHOLD_QUARTER,
    DAYS_BEFORE_FND_TO_ROLL,
    VOL_LOOKBACK_DAYS,
    VOL_HIGH_MULTIPLIER,
    VOL_EXTREME_MULTIPLIER,
    VOL_RISK_REDUCTION,
    SUSPEND_TRADING_IN_EXTREME,
    MAX_DRAWDOWN_STOP,
    BACKTEST_START,
    BACKTEST_END,
    INITIAL_EQUITY,
    STOP_LIMIT_ENABLED,
    STOP_LIMIT_BAND_TICKS,
    STOP_LIMIT_BAND_MULTIPLIER,
    TRAILING_STOP_ENABLED,
    COMMISSION,
    SLIPPAGE,
    TOTAL_TRANSACTION_COST,
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
    # 2. Load China PMI (Monthly) and forward-fill to daily
    # ========================================================================
    print("\n[2/4] Loading PMI data...")
    pmi = pd.read_csv(PMI_FILE)
    pmi['Date'] = pd.to_datetime(pmi['Date'], format='%m/%d/%Y')
    pmi = pmi.sort_values('Date').reset_index(drop=True)
    pmi = pmi.rename(columns={'PMI Manufacturing *': 'PMI'})
    
    # Calculate PMI_change BEFORE resampling (using actual monthly values)
    pmi['PMI_change'] = pmi['PMI'].diff()
    
    print(f"   ✓ Loaded {len(pmi)} monthly PMI observations")
    
    # Forward-fill to daily frequency (preserves monthly PMI_change, avoids look-ahead bias)
    print("   ⟳ Resampling monthly PMI to daily (forward-fill)...")
    pmi_daily = pmi.set_index('Date').resample('D').ffill()
    pmi_daily = pmi_daily.reset_index()
    
    print(f"   ✓ Resampled to {len(pmi_daily)} daily observations")
    print(f"   ✓ Date range: {pmi_daily['Date'].min().date()} to {pmi_daily['Date'].max().date()}")
    
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
    print(f"Signal threshold: ±{SIGNAL_THRESHOLD}")
    print(f"Generating signals from: {backtest_start}")
    
    # Prepare features and target
    feature_cols = ['DXY_logret', 'PMI_change', 'STOCKS_logret']
    target_col = 'HG1_logret'

    # -----------------------------------------------------------------------
    # NEW (Part B): Volatility regime classification
    # -----------------------------------------------------------------------
    # Daily realised volatility
    realized_vol = data[target_col].rolling(VOL_LOOKBACK_DAYS).std()

    backtest_start_idx = data[data['Date'] >= backtest_start].index[0]
    long_run_vol_daily = realized_vol.iloc[:backtest_start_idx].mean()

    # Store for diagnostics if you like
    data['Realized_Vol_Daily'] = realized_vol
    data['Realized_Vol_Ann']   = realized_vol * np.sqrt(252)  # purely informational

    # Vol regime in *daily* units
    data['Vol_Regime'] = 'NORMAL'
    high_mask    = data['Realized_Vol_Daily'] > VOL_HIGH_MULTIPLIER    * long_run_vol_daily
    extreme_mask = data['Realized_Vol_Daily'] > VOL_EXTREME_MULTIPLIER * long_run_vol_daily

    data.loc[high_mask,    'Vol_Regime'] = 'HIGH'
    data.loc[extreme_mask, 'Vol_Regime'] = 'EXTREME'

    print("\n[Volatility Regime Distribution in training+backtest]:")
    print(data['Vol_Regime'].value_counts(dropna=False))


    # Initialize columns for results
    data['Forecasted_Return'] = np.nan
    data['Signal'] = None  # Use None to distinguish unprocessed rows
    data['Model_Alpha'] = np.nan
    data['Model_Beta_DXY'] = np.nan
    data['Model_Beta_PMI'] = np.nan
    data['Model_Beta_STOCKS'] = np.nan
    data['Model_R2'] = np.nan
    
    print(f"\nTraining period ends at index: {backtest_start_idx - 1} ({data.loc[backtest_start_idx - 1, 'Date'].date()})")
    print(f"Signal generation starts at index: {backtest_start_idx} ({data.loc[backtest_start_idx, 'Date'].date()})")
    print(f"Total signals to generate: {len(data) - backtest_start_idx}")
    
    # Counters for accurate signal distribution
    long_signals = 0
    short_signals = 0
    neutral_signals = 0
    
    # Rolling regression - only generate signals for backtest period
    print(f"\nRunning rolling regression...")
    print(f"   ⚠ IMPORTANT: Using t-1 data to predict t returns (no look-ahead bias)")
    
    for i in range(backtest_start_idx, len(data)):
        # Check if we have enough history (need window_days + 1 for yesterday's data)
        if i < window_days + 1:
            continue
        
        # Training window: use data UP TO yesterday (i-1), not including today
        train_idx = range(i - window_days - 1, i - 1)
        X_train = data.loc[train_idx, feature_cols]
        y_train = data.loc[train_idx, target_col]
        
        # Skip if any NaN in training data
        if X_train.isna().any().any() or y_train.isna().any():
            continue
        
        # Fit regression model on historical data (up to yesterday)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store model coefficients
        data.loc[i, 'Model_Alpha'] = model.intercept_
        data.loc[i, 'Model_Beta_DXY'] = model.coef_[0]
        data.loc[i, 'Model_Beta_PMI'] = model.coef_[1]
        data.loc[i, 'Model_Beta_STOCKS'] = model.coef_[2]
        data.loc[i, 'Model_R2'] = model.score(X_train, y_train)
        
        # CRITICAL FIX: Use YESTERDAY's features to predict TODAY's return
        # This simulates generating the signal before market open using available data
        X_yesterday = data.loc[i-1, feature_cols].values.reshape(1, -1)
        forecast = model.predict(X_yesterday)[0]
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
    
    # Correlation matrix
    print(f"\n[CORRELATION MATRIX]")
    feature_cols = ['HG1_logret', 'DXY_logret', 'PMI_change', 'STOCKS_logret']
    corr_matrix = backtest_data[feature_cols].corr()
    
    # Print correlation matrix with formatting
    print("\n   Correlation Matrix (Pearson):")
    print("   " + "-" * 70)
    header = f"   {'Variable':<15} | {'HG Returns':<12} | {'DXY':<12} | {'PMI':<12} | {'Stocks':<12}"
    print(header)
    print("   " + "-" * 70)
    
    var_names = ['HG Returns', 'DXY', 'PMI Change', 'Stocks']
    for i, (idx, row) in enumerate(corr_matrix.iterrows()):
        values = " | ".join([f"{val:>12.4f}" for val in row])
        print(f"   {var_names[i]:<15} | {values}")
    print("   " + "-" * 70)
    
    # Highlight key correlations
    print(f"\n   Key Observations:")
    hg_dxy_corr = corr_matrix.loc['HG1_logret', 'DXY_logret']
    hg_pmi_corr = corr_matrix.loc['HG1_logret', 'PMI_change']
    hg_stocks_corr = corr_matrix.loc['HG1_logret', 'STOCKS_logret']
    
    print(f"   • HG vs DXY correlation:    {hg_dxy_corr:+.4f} {'(negative - inverse relationship)' if hg_dxy_corr < 0 else '(positive)'}")
    print(f"   • HG vs PMI correlation:    {hg_pmi_corr:+.4f} {'(positive - direct relationship)' if hg_pmi_corr > 0 else '(negative)'}")
    print(f"   • HG vs Stocks correlation: {hg_stocks_corr:+.4f} {'(negative - inverse relationship)' if hg_stocks_corr < 0 else '(positive)'}")
    
    # Check for multicollinearity among predictors
    print(f"\n   Multicollinearity Check (among predictors):")
    dxy_pmi_corr = corr_matrix.loc['DXY_logret', 'PMI_change']
    dxy_stocks_corr = corr_matrix.loc['DXY_logret', 'STOCKS_logret']
    pmi_stocks_corr = corr_matrix.loc['PMI_change', 'STOCKS_logret']
    
    print(f"   • DXY vs PMI:               {dxy_pmi_corr:+.4f}")
    print(f"   • DXY vs Stocks:            {dxy_stocks_corr:+.4f}")
    print(f"   • PMI vs Stocks:            {pmi_stocks_corr:+.4f}")
    
    max_predictor_corr = max(abs(dxy_pmi_corr), abs(dxy_stocks_corr), abs(pmi_stocks_corr))
    if max_predictor_corr < 0.5:
        print(f"   ✓ Low multicollinearity (max |r| = {max_predictor_corr:.4f} < 0.5)")
    elif max_predictor_corr < 0.7:
        print(f"   ⚠ Moderate multicollinearity (max |r| = {max_predictor_corr:.4f})")
    else:
        print(f"   ⚠ High multicollinearity (max |r| = {max_predictor_corr:.4f} > 0.7)")
    
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


def generate_correlation_matrix_chart(backtest_data, output_dir):
    """
    Generate correlation matrix heatmap for regression variables
    
    Args:
        backtest_data: DataFrame with regression features and target
        output_dir: Directory to save the chart
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n[Generating Regression Variables Correlation Matrix Chart]")
    
    # Select regression variables
    feature_cols = ['HG1_logret', 'DXY_logret', 'PMI_change', 'STOCKS_logret']
    corr_matrix = backtest_data[feature_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap using matplotlib
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=25, fontsize=11)
    
    # Add text annotations with correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            value = corr_matrix.iloc[i, j]
            # Use white text for dark cells, black for light cells
            text_color = 'white' if abs(value) > 0.5 else 'black'
            text = ax.text(j, i, f'{value:.4f}',
                         ha="center", va="center", color=text_color, 
                         fontsize=12, fontweight='bold')
    
    # Add border to highlight top row (HG Returns correlations with predictors)
    from matplotlib.patches import Rectangle
    
    # Highlight the top row (i=0) - correlations of predictors with HG Returns
    for j in range(len(corr_matrix.columns)):
        rect = Rectangle((j - 0.5, -0.5), 1, 1, 
                        linewidth=8, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    
    # Set ticks and labels
    var_names = ['HG Returns', 'DXY Returns', 'PMI Change', 'LME Stocks']
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(var_names, fontsize=11)
    
    # Add title
    ax.set_title('Pearson Correlation Matrix - Regression Variables', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Add subtitle with key insights
    # hg_dxy_corr = corr_matrix.loc['HG1_logret', 'DXY_logret']
    # hg_pmi_corr = corr_matrix.loc['HG1_logret', 'PMI_change']
    # hg_stocks_corr = corr_matrix.loc['HG1_logret', 'STOCKS_logret']
    
    # subtitle = (f"Key: HG-DXY={hg_dxy_corr:+.4f} | "
    #            f"HG-PMI={hg_pmi_corr:+.4f} | "
    #            f"HG-Stocks={hg_stocks_corr:+.4f}")
    # ax.text(0.5, -0.15, subtitle, transform=ax.transAxes,
    #        ha='center', fontsize=10, style='italic',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save chart
    chart_filename = f'{output_dir}/correlation_matrix.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Correlation matrix chart saved to {chart_filename}")


# ============================================================================
# PHASE 1: CONTRACT MANAGEMENT & DATA LOADING
# ============================================================================

def load_fnd_calendar():
    """
    Load First Notice Day calendar for all HG contracts
    
    Returns:
        DataFrame with columns: Contract, FND_Date
    """
    print("\n[Loading FND Calendar]")
    from config import FND_CALENDAR_FILE
    fnd = pd.read_csv(FND_CALENDAR_FILE)
    fnd['FND'] = pd.to_datetime(fnd['FND'], format='%m/%d/%Y')
    fnd = fnd.rename(columns={'FND': 'FND_Date'})
    
    print(f"   ✓ Loaded {len(fnd)} contract FND dates")
    print(f"   ✓ Date range: {fnd['FND_Date'].min().date()} to {fnd['FND_Date'].max().date()}")
    
    return fnd[['Contract', 'FND_Date']]


def get_active_contracts(current_date, fnd_calendar):
    """
    Determine which contracts are M1 (front), M2 (first-deferred), M3 (second-deferred)
    based on current date and FND calendar
    
    Strategy: We trade M2 to avoid delivery risk
    
    Args:
        current_date: Current simulation date
        fnd_calendar: DataFrame with Contract, FND_Date
    
    Returns:
        dict: {
            'M1': 'HGK24',  # Front month (avoid this)
            'M2': 'HGN24',  # First-deferred (we trade this)
            'M3': 'HGQ24'   # Second-deferred (for carry calculation)
        }
    """
    # Get all contracts with FND >= current_date
    future_contracts = fnd_calendar[fnd_calendar['FND_Date'] >= current_date].sort_values('FND_Date')
    
    if len(future_contracts) < 3:
        raise ValueError(f"Not enough future contracts available for date {current_date}")
    
    contracts = {
        'M1': future_contracts.iloc[0]['Contract'],  # Front month
        'M2': future_contracts.iloc[1]['Contract'],  # First-deferred (we trade)
        'M3': future_contracts.iloc[2]['Contract']   # Second-deferred (for carry)
    }
    
    return contracts


def load_contract_data(contract_code, start_date, end_date):
    """
    Load OHLC data for a specific HG contract
    
    Args:
        contract_code: e.g., 'HGK24'
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Settlement
    """
    filepath = f'data/futures/{contract_code}.csv'
    
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter to date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Select relevant columns and rename (KEEP OPEN for execution)
        df = df[['Date', 'Open', 'High', 'Low', 'Last', 'Settlement Price']].copy()
        df = df.rename(columns={'Settlement Price': 'Settlement', 'Last': 'Close'})
        
        # Convert price columns to numeric (some may have strings like "@NA")
        price_cols = ['Open', 'High', 'Low', 'Close', 'Settlement']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing price data
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        print(f"   ⚠ WARNING: Contract file not found: {filepath}")
        return pd.DataFrame()


def build_contract_database(fnd_calendar, start_date, end_date):
    """
    Load all contract data and organize by contract code
    
    Returns:
        dict: {
            'HGK24': DataFrame with Date, Open, High, Low, Close, Settlement,
            'HGN24': DataFrame with Date, Open, High, Low, Close, Settlement,
            ...
        }
    """
    print("\n" + "="*80)
    print("LOADING CONTRACT DATABASE")
    print("="*80)
    
    contract_data = {}
    
    for _, row in fnd_calendar.iterrows():
        contract = row['Contract']
        print(f"\n[Loading {contract}]")
        
        df = load_contract_data(contract, start_date, end_date)
        
        if not df.empty:
            contract_data[contract] = df
            print(f"   ✓ Loaded {len(df)} observations")
            print(f"   ✓ Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        else:
            print(f"   ⚠ No data loaded")
    
    print(f"\n✓ Total contracts loaded: {len(contract_data)}")
    print("="*80)
    
    return contract_data


# ============================================================================
# PHASE 2: ATR CALCULATION & CARRY SPREAD
# ============================================================================

def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default 14)
    
    Returns:
        Series: ATR values
    """
    # True Range calculation
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the rolling mean of TR
    atr = tr.rolling(window=period).mean()
    
    return atr


def add_atr_to_contract(df, period=14):
    """
    Add ATR column to contract DataFrame
    
    Args:
        df: DataFrame with High, Low, Close
        period: ATR period
    
    Returns:
        DataFrame with added ATR column
    """
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period)
    return df


def calculate_overnight_spread(df):
    """
    Calculate overnight spread = Open - Previous Settlement
    Used to determine limit band for stop-limit orders
    
    Args:
        df: DataFrame with Open and Settlement columns
    
    Returns:
        Series: Overnight spread (absolute value)
    """
    prev_settlement = df['Settlement'].shift(1)
    overnight_spread = abs(df['Open'] - prev_settlement)
    return overnight_spread


def add_overnight_spread_to_contract(df):
    """
    Add overnight spread column to contract DataFrame
    
    Args:
        df: DataFrame with Open, Settlement
    
    Returns:
        DataFrame with added Overnight_Spread and Avg_Overnight_Spread columns
    """
    df['Overnight_Spread'] = calculate_overnight_spread(df)
    
    # Calculate rolling 20-day average overnight spread
    df['Avg_Overnight_Spread'] = df['Overnight_Spread'].rolling(window=20, min_periods=5).mean()
    
    return df


def calculate_limit_band(avg_overnight_spread, min_ticks=4, tick_size=0.0005):
    """
    Calculate limit band for stop-limit orders
    
    Formula: max(4 ticks, 1.5 × average overnight spread)
    
    Args:
        avg_overnight_spread: Average overnight spread ($/lb)
        min_ticks: Minimum number of ticks (default 4)
        tick_size: HG tick size (default $0.0005/lb)
    
    Returns:
        float: Limit band in $/lb
    """
    min_band = min_ticks * tick_size  # 4 ticks = $0.002/lb
    
    if pd.isna(avg_overnight_spread):
        return min_band
    
    dynamic_band = 1.5 * avg_overnight_spread
    
    return max(min_band, dynamic_band)


def calculate_carry_spread(m2_settlement, m3_settlement):
    """
    Calculate carry spread = M3_settlement - M2_settlement
    
    Positive = Contango (M3 > M2) - bearish for longs
    Negative = Backwardation (M3 < M2) - bullish for longs
    
    Args:
        m2_settlement: M2 settlement price ($/lb)
        m3_settlement: M3 settlement price ($/lb)
    
    Returns:
        float: Carry spread ($/lb)
    """
    return m3_settlement - m2_settlement


def apply_carry_filter(signal, carry_spread):
    """
    Apply carry-based position sizing dampening
    
    Args:
        signal: SignalType (LONG/SHORT/NEUTRAL)
        carry_spread: float ($/lb, M3 - M2)
    
    Returns:
        carry_multiplier: float (1.0, 0.5, 0.25)
    
    Logic:
        LONG positions (hurt by contango):
            - If carry_spread <= 0.01: multiplier = 1.0 (full size)
            - If 0.01 < carry_spread <= 0.02: multiplier = 0.5 (half)
            - If carry_spread > 0.02: multiplier = 0.25 (quarter)
        
        SHORT positions (hurt by backwardation):
            - If carry_spread >= -0.01: multiplier = 1.0 (full size)
            - If -0.02 <= carry_spread < -0.01: multiplier = 0.5 (half)
            - If carry_spread < -0.02: multiplier = 0.25 (quarter)
        
        NEUTRAL: multiplier = 0.0
    """
    # normalise to string
    if isinstance(signal, SignalType):
        sig = signal.value
    else:
        sig = str(signal)

    if sig == SignalType.NEUTRAL.value:
        return 0.0

    if sig == SignalType.LONG.value:
        # LONG hurt by contango (positive spread)
        if carry_spread <= CARRY_THRESHOLD_HALF:
            return 1.0
        elif carry_spread <= CARRY_THRESHOLD_QUARTER:
            return 0.5
        else:
            return 0.25

    if sig == SignalType.SHORT.value:
        # SHORT hurt by backwardation (negative spread)
        if carry_spread >= -CARRY_THRESHOLD_HALF:
            return 1.0
        elif carry_spread >= -CARRY_THRESHOLD_QUARTER:
            return 0.5
        else:
            return 0.25

    # fallback
    return 0.0


def calculate_limit_band(avg_overnight_spread):
    """
    Calculate the limit band for stop-limit orders
    
    Args:
        avg_overnight_spread: float (average overnight spread in $/lb)
    
    Returns:
        limit_band: float ($/lb) - maximum distance from stop price to still fill
    
    Logic per trading plan:
        "Limit band = max(4 ticks, 1.5 × average overnight spread)"
        - Minimum: 4 ticks = 4 × $0.0005 = $0.002/lb
        - Dynamic: 1.5 × avg overnight spread
        - Take the larger of these two values
    """
    from config import STOP_LIMIT_BAND_TICKS, STOP_LIMIT_BAND_MULTIPLIER
    
    # Calculate minimum band (4 ticks)
    tick_size = 0.0005  # $0.0005/lb
    min_band = STOP_LIMIT_BAND_TICKS * tick_size  # 4 × $0.0005 = $0.002
    
    # Calculate dynamic band (1.5 × avg overnight spread)
    if pd.notna(avg_overnight_spread) and avg_overnight_spread > 0:
        dynamic_band = STOP_LIMIT_BAND_MULTIPLIER * avg_overnight_spread
    else:
        # If avg overnight spread is missing, default to min band
        dynamic_band = min_band
    
    # Return the larger of the two
    limit_band = max(min_band, dynamic_band)
    
    return limit_band


# ============================================================================
# PHASE 3: POSITION SIZING & RISK MANAGEMENT
# ============================================================================

def calculate_position_size(equity, atr, carry_multiplier, risk_percent=RISK_PERCENT):
    """
    Calculate number of contracts to trade using ATR-based risk sizing
    
    Args:
        equity: Current account equity ($)
        atr: Current 14-day ATR ($/lb)
        carry_multiplier: From carry filter (1.0, 0.5, 0.25)
        risk_percent: Risk per trade (default 1%)
    
    Returns:
        int: Number of contracts to trade
    
    Logic:
        1. max_risk_dollars = equity * risk_percent
        2. stop_distance_per_lb = 1.5 * atr
        3. dollar_risk_per_contract = stop_distance_per_lb * 25000
        4. base_contracts = floor(max_risk_dollars / dollar_risk_per_contract)
        5. final_contracts = floor(base_contracts * carry_multiplier)
    """
    
    if atr == 0 or np.isnan(atr) or carry_multiplier == 0:
        return 0
    
    # Maximum risk in dollars
    max_risk_dollars = equity * risk_percent
    
    # Stop distance based on ATR
    stop_distance_per_lb = ATR_STOP_MULTIPLIER * atr
    
    # Risk per contract
    dollar_risk_per_contract = stop_distance_per_lb * CONTRACT_SIZE
    
    if dollar_risk_per_contract == 0:
        return 0
    
    # Base position size
    base_contracts = int(max_risk_dollars / dollar_risk_per_contract)
    
    # Apply carry filter
    final_contracts = int(base_contracts * carry_multiplier)
    
    return max(0, final_contracts)  # Ensure non-negative


def calculate_stop_loss(entry_price, atr, direction):
    """
    Calculate ATR-based stop-loss price
    
    Args:
        entry_price: Entry price ($/lb)
        atr: 14-day ATR ($/lb)
        direction: "LONG" or "SHORT"
    
    Returns:
        float: Stop-loss price ($/lb)
    
    Logic:
        LONG: stop = entry_price - (1.5 * atr)
        SHORT: stop = entry_price + (1.5 * atr)
    """
    stop_distance = ATR_STOP_MULTIPLIER * atr
    
    if direction == "LONG":
        return entry_price - stop_distance
    elif direction == "SHORT":
        return entry_price + stop_distance
    else:
        return None


def check_stop_loss(current_price, stop_price, direction):
    """
    Check if stop-loss is hit
    
    Args:
        current_price: Current market price ($/lb)
        stop_price: Stop-loss level ($/lb)
        direction: "LONG" or "SHORT"
    
    Returns:
        bool: True if stop hit
    """
    if stop_price is None:
        return False
    
    if direction == "LONG":
        return current_price <= stop_price
    elif direction == "SHORT":
        return current_price >= stop_price
    
    return False


# ============================================================================
# PHASE 3B: RISK SCANNERS (Circuit Breakers)
# ============================================================================

def check_atr_spike(contract_data, current_date, current_contract):
    """
    Scanner 2: ATR Change Detection
    
    Triggers when 5-day ATR / 100-day ATR > 2.5.
    Indicates volatility regime shift.
    
    Args:
        contract_data: Dictionary of contract DataFrames
        current_date: Date to check
        current_contract: Contract code (e.g., "HGK25")
    
    Returns:
        dict: {'triggered': bool, 'position_scalar': float, 'rationale': str}
    """
    if current_contract not in contract_data:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'No data'}
    
    df = contract_data[current_contract]
    df = df[df['Date'] <= current_date].copy()
    
    if len(df) < 100:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'Insufficient history'}
    
    # Check if ATR column exists
    if 'ATR' not in df.columns:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'No ATR data'}
    
    # Calculate 5-day and 100-day ATR
    atr_5d = df['ATR'].iloc[-5:].mean()
    atr_100d = df['ATR'].iloc[-100:].mean()
    
    if atr_100d == 0:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'ATR undefined'}
    
    atr_ratio = atr_5d / atr_100d
    
    # Trigger if ATR ratio > 2.5
    if atr_ratio > 2.0:
        return {
            'triggered': True,
            'position_scalar': 0.0,  # Flatten position
            'rationale': f'ATR spike: {atr_ratio:.2f}x baseline'
        }
    
    return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'ATR normal'}


def check_structure_shock(contract_data, current_date, m1_contract, m2_contract):
    """
    Scanner 3: Structure of Market Shock
    
    Triggers when (M2-M1 spread - 30d avg) / std > 4 standard deviations.
    Indicates severe supply/demand dislocation.
    
    Args:
        contract_data: Dictionary of contract DataFrames
        current_date: Date to check
        m1_contract: Front month contract code
        m2_contract: Second month contract code
    
    Returns:
        dict: {'triggered': bool, 'position_scalar': float, 'rationale': str}
    """
    if m1_contract not in contract_data or m2_contract not in contract_data:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'No data'}
    
    df_m1 = contract_data[m1_contract]
    df_m2 = contract_data[m2_contract]
    
    df_m1 = df_m1[df_m1['Date'] <= current_date].copy()
    df_m2 = df_m2[df_m2['Date'] <= current_date].copy()
    
    if len(df_m1) < 30 or len(df_m2) < 30:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'Insufficient history'}
    
    # Align dates using merge
    df_merged = df_m1[['Date', 'Settlement']].merge(
        df_m2[['Date', 'Settlement']], on='Date', suffixes=('_m1', '_m2')
    )
    
    if len(df_merged) < 30:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'Insufficient common dates'}
    
    # Calculate M2-M1 spread
    spread = df_merged['Settlement_m2'] - df_merged['Settlement_m1']
    
    # Get last 30 days
    spread_30d = spread.iloc[-30:]
    current_spread = spread.iloc[-1]
    
    # Calculate z-score
    mean_spread = spread_30d.mean()
    std_spread = spread_30d.std()
    
    if std_spread == 0:
        return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'No spread volatility'}
    
    z_score = (current_spread - mean_spread) / std_spread
    
    # Trigger if |z-score| > 4
    if abs(z_score) > 3:
        return {
            'triggered': True,
            'position_scalar': 0.0,  # Flatten position
            'rationale': f'Structure shock: z-score={z_score:.2f}'
        }
    
    return {'triggered': False, 'position_scalar': 1.0, 'rationale': 'Structure normal'}

# ============================================================================
# PHASE 4: ROLL MANAGEMENT
# ============================================================================

def should_roll(current_date, current_m2_contract, fnd_calendar, contract_data):
    """
    Check if we need to roll from current M2 to next M2
    
    Roll trigger: 5 trading days before M1's FND
    When M1's FND is near, we roll from current M2 (which becomes M1) to next M2 (current M3)
    
    Args:
        current_date: Current simulation date (pandas Timestamp)
        current_m2_contract: Current M2 contract code
        fnd_calendar: FND calendar DataFrame
        contract_data: dict of contract DataFrames (to count actual trading days)
    
    Returns:
        tuple: (should_roll: bool, next_m2_contract: str or None)
    """
    # Get future contracts (M1, M2, M3, etc.)
    future_contracts = fnd_calendar[fnd_calendar['FND_Date'] >= current_date].sort_values('FND_Date')
    
    if len(future_contracts) < 3:
        return False, None
    
    m1_contract = future_contracts.iloc[0]['Contract']  # Front month
    m1_fnd = future_contracts.iloc[0]['FND_Date']
    m2_contract = future_contracts.iloc[1]['Contract']  # First-deferred (we're trading this)
    m3_contract = future_contracts.iloc[2]['Contract']  # Second-deferred (roll to this)
    
    # If we already rolled (current_m2_contract != m2_contract), don't roll again
    if current_m2_contract != m2_contract:
        return False, None, None
    
    # Count actual trading days until M1 FND using M1 contract data
    if m1_contract in contract_data:
        m1_data = contract_data[m1_contract]
        
        # Get all trading days between current_date and m1_fnd
        trading_days = m1_data[
            (m1_data['Date'] > current_date) & 
            (m1_data['Date'] <= m1_fnd)
        ]
        
        trading_days_until_fnd = len(trading_days)
    else:
        # Fallback to approximation if contract data not available
        days_until_fnd = (m1_fnd - current_date).days
        trading_days_until_fnd = int(days_until_fnd * 5 / 7)  # Approximate weekdays
    
    # Roll if within threshold - roll from M2 to M3 (M3 becomes new M2)
    if trading_days_until_fnd <= DAYS_BEFORE_FND_TO_ROLL:
        return True, m3_contract, m1_fnd  # Roll to M3 (which will become the new M2)
    
    return False, None, None


def execute_roll(current_position, old_contract, new_contract, 
                 old_price, new_price, current_date, roll_id, fnd_date=None):
    """
    Execute contract roll (close old M2, open new M2)
    
    Args:
        current_position: Current Position object
        old_contract: Contract to exit (e.g., 'HGK24')
        new_contract: Contract to enter (e.g., 'HGN24')
        old_price: Exit price for old contract ($/lb)
        new_price: Entry price for new contract ($/lb)
        current_date: Roll date
        roll_id: Unique roll ID
        fnd_date: First Notice Day that triggered the roll
    
    Returns:
        tuple: (exit_trade, entry_trade, roll_event)
    """
    # Direction of position
    if "LONG" in current_position.entry_trade.action:
        direction = "LONG"
        exit_action = "CLOSE_LONG"
        entry_action = "OPEN_LONG"
    else:
        direction = "SHORT"
        exit_action = "CLOSE_SHORT"
        entry_action = "OPEN_SHORT"
    
    num_contracts = current_position.entry_trade.num_contracts
    
    # Create exit trade for old contract
    exit_trade = Trade(
        trade_id=current_position.entry_trade.trade_id,
        date=current_date,
        action=exit_action,
        contract_code=old_contract,
        price=old_price,
        num_contracts=num_contracts
    )
    
    # Create entry trade for new contract
    entry_trade = Trade(
        trade_id=current_position.entry_trade.trade_id + 1,
        date=current_date,
        action=entry_action,
        contract_code=new_contract,
        price=new_price,
        num_contracts=num_contracts,
        atr=current_position.entry_trade.atr,
        carry_spread=current_position.entry_trade.carry_spread,
        carry_multiplier=current_position.entry_trade.carry_multiplier
    )
    
    # Create roll event for logging
    roll_event = RollEvent(
        roll_id=roll_id,
        roll_date=current_date,
        from_contract=old_contract,
        to_contract=new_contract,
        exit_price=old_price,
        entry_price=new_price,
        num_contracts=num_contracts,
        fnd_date=fnd_date
    )
    
    # Calculate roll P&L
    roll_event.calculate_roll_pnl()
    
    return exit_trade, entry_trade, roll_event


# ============================================================================
# PHASE 4: BACKTEST ENGINE
# ============================================================================

def determine_action(current_status, new_signal, carry_multiplier):
    """
    Determine what trading action to take based on current state and new signal
    
    Args:
        current_status: PositionStatus (FLAT, LONG, SHORT)
        new_signal: SignalType or str ("LONG", "SHORT", "NEUTRAL")
        carry_multiplier: float (0.0 if stand aside)
    """

    # --- normalise new_signal to a plain string -----------------------------
    if isinstance(new_signal, SignalType):
        ns = new_signal.value
    else:
        ns = str(new_signal)

    # If carry filter says stand aside, treat as NEUTRAL
    if carry_multiplier == 0.0:
        ns = SignalType.NEUTRAL.value

    # ----------------------- FLAT (no position) -----------------------------
    if current_status == PositionStatus.FLAT:
        if ns == SignalType.LONG.value:
            return "OPEN_LONG"
        elif ns == SignalType.SHORT.value:
            return "OPEN_SHORT"
        else:
            return "HOLD"

    # ----------------------- LONG position ---------------------------------
    if current_status == PositionStatus.LONG:
        if ns == SignalType.LONG.value:
            return "HOLD"
        elif ns == SignalType.NEUTRAL.value:
            return "CLOSE_LONG"
        elif ns == SignalType.SHORT.value:
            if carry_multiplier > 0:
                return "REVERSE_TO_SHORT"
            else:
                return "CLOSE_LONG"

    # ----------------------- SHORT position --------------------------------
    if current_status == PositionStatus.SHORT:
        if ns == SignalType.SHORT.value:
            return "HOLD"
        elif ns == SignalType.NEUTRAL.value:
            return "CLOSE_SHORT"
        elif ns == SignalType.LONG.value:
            if carry_multiplier > 0:
                return "REVERSE_TO_LONG"
            else:
                return "CLOSE_SHORT"

    return "HOLD"



def run_backtest(signal_data, contract_data, fnd_calendar, initial_equity):
    """
    Main backtest simulation loop
    
    Args:
        signal_data: DataFrame with dates, signals, forecasts
        contract_data: dict of {contract_code: DataFrame}
        fnd_calendar: FND calendar DataFrame
        initial_equity: Starting capital
    
    Returns:
        BacktestResults object
    """
    
    print("\n" + "="*80)
    print("RUNNING BACKTEST SIMULATION")
    print("="*80)
    
    # Initialize results container
    results = BacktestResults(initial_equity=initial_equity)
    
    # Initialize state variables
    equity = initial_equity
    peak_equity = equity                    # NEW: track running peak
    current_drawdown = 0.0                  # NEW: initialise drawdown
    baseline_risk_percent = RISK_PERCENT    # NEW: base risk per trade
    position_status = PositionStatus.FLAT
    current_position = None
    current_m2_contract = None

    print(f"Initial Equity: ${equity:,.2f}")
    print(f"Risk per trade: {RISK_PERCENT*100}%")
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER}x")
    
    # Counters
    trade_id_counter = 1
    position_id_counter = 1
    roll_id_counter = 1
    
    # Track entry price for unrealized P&L
    entry_price = None
    entry_direction = None
    
    print(f"\nStarting simulation...")
    print(f"Initial Equity: ${equity:,.2f}")
    print(f"Risk per trade: {RISK_PERCENT*100}%")
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER}x")
    
    # Main loop - iterate through each trading day
    for idx, row in signal_data.iterrows():
        current_date = row['Date']
        current_signal = row['Signal']
        forecasted_return = row['Forecasted_Return']

        # ---------------------------------------------------------------
        # NEW (Part B): portfolio drawdown kill-switch
        # ---------------------------------------------------------------
        if equity > peak_equity:
            peak_equity = equity
            
        current_drawdown = (equity - peak_equity) / peak_equity  # <= 0 when below peak

        if current_drawdown <= -MAX_DRAWDOWN_STOP:
            # After a large drawdown we stop opening new positions
            # We still allow existing positions to be managed/closed.
            current_signal = SignalType.NEUTRAL.value

        # ---------------------------------------------------------------
        # NEW (Part B): volatility-regime-based risk adjustment
        # ---------------------------------------------------------------
        vol_regime = row.get('Vol_Regime', 'NORMAL')

        # Start with baseline risk per trade
        risk_percent = baseline_risk_percent

        if vol_regime == 'HIGH':
            # Cut risk (e.g. 1% -> 0.5%) in high-vol environments
            risk_percent *= VOL_RISK_REDUCTION

        elif vol_regime == 'EXTREME':
            if SUSPEND_TRADING_IN_EXTREME:
                # Don’t open new positions in extreme volatility regimes
                current_signal = SignalType.NEUTRAL.value
            else:
                # If you choose not to suspend, at least crush the risk
                risk_percent *= VOL_RISK_REDUCTION * 0.5

        # Progress indicator
        if idx % 20 == 0:
            pct = (idx / len(signal_data)) * 100
            print(f"   Progress: {pct:.1f}% ({idx}/{len(signal_data)} days) | Equity: ${equity:,.0f}", end='\r')
        
        # ====================================================================
        # 1. DETERMINE ACTIVE CONTRACTS (M1, M2, M3)
        # ====================================================================
        try:
            contracts = get_active_contracts(current_date, fnd_calendar)
        except ValueError:
            continue
        
        m2_contract = contracts['M2']
        m3_contract = contracts['M3']
        
        # Check if M2 changed (roll detected)
        if current_m2_contract is None:
            current_m2_contract = m2_contract
        
        # Get price data for M2 and M3
        if m2_contract not in contract_data or m3_contract not in contract_data:
            continue
        
        m2_data = contract_data[m2_contract]
        m3_data = contract_data[m3_contract]
        
        # Get today's prices
        m2_today = m2_data[m2_data['Date'] == current_date]
        m3_today = m3_data[m3_data['Date'] == current_date]
        
        if m2_today.empty or m3_today.empty:
            continue
        
        # EXECUTION PRICES: Use SETTLEMENT for entry/exit (end-of-day execution)
        # HIGH/LOW: Use for stop-loss trigger detection (intraday)
        m2_settlement = m2_today.iloc[0]['Settlement']
        m2_high = m2_today.iloc[0]['High']
        m2_low = m2_today.iloc[0]['Low']
        m3_settlement = m3_today.iloc[0]['Settlement']
        m2_atr = m2_today.iloc[0]['ATR'] if 'ATR' in m2_today.columns else np.nan
        
        # Get average overnight spread for stop-limit calculation
        m2_avg_overnight_spread = m2_today.iloc[0]['Avg_Overnight_Spread'] if 'Avg_Overnight_Spread' in m2_today.columns else np.nan
        
        # Get YESTERDAY's ATR for position sizing (known before market open)
        idx_today = m2_data[m2_data['Date'] == current_date].index
        if len(idx_today) > 0 and idx_today[0] > 0:
            m2_atr_yesterday = m2_data.loc[idx_today[0] - 1, 'ATR'] if 'ATR' in m2_data.columns else np.nan
        else:
            m2_atr_yesterday = m2_atr  # Fallback to today's ATR if no yesterday
        
        # Skip if ATR not available yet or Settlement price missing
        if np.isnan(m2_atr) or np.isnan(m2_settlement) or np.isnan(m2_atr_yesterday):
            continue
        
        # ====================================================================
        # TRADING LOGIC: Check Exit Criteria if Position Held, Entry Criteria if Flat
        # ====================================================================
        
        # Initialize daily P&L tracking
        realized_pnl_today = 0.0
        roll_pnl_today = 0.0
        
        # Calculate carry spread (needed for both entry and exit logic)
        carry_spread = calculate_carry_spread(m2_settlement, m3_settlement)
        
        # ====================================================================
        # 2. RISK SCANNER CHECKS (Circuit Breakers)
        # ====================================================================
        # Run all four risk scanners to detect abnormal market conditions
        m1_contract = contracts['M1']
        
        scanner_results = {
            'atr': check_atr_spike(contract_data, current_date, m2_contract),
            'structure': check_structure_shock(contract_data, current_date, m1_contract, m2_contract),
        }
        
        # Aggregate scanner results
        any_scanner_triggered = any(s['triggered'] for s in scanner_results.values())
        
        # If any scanner triggers, flatten position immediately
        if any_scanner_triggered and current_position is not None:
            # Log which scanners triggered
            triggered_scanners = [name for name, result in scanner_results.items() if result['triggered']]
            rationale = "; ".join([scanner_results[name]['rationale'] for name in triggered_scanners])
            
            # Close position at settlement price
            exit_action = "CLOSE_LONG" if "LONG" in current_position.entry_trade.action else "CLOSE_SHORT"
            
            exit_trade = Trade(
                trade_id=trade_id_counter,
                date=current_date,
                action=exit_action,
                contract_code=current_m2_contract,
                price=m2_settlement,
                num_contracts=current_position.entry_trade.num_contracts
            )
            trade_id_counter += 1
            
            current_position.exit_trade = exit_trade
            current_position.exit_reason = f"RISK_SCANNER: {rationale}"
            current_position.calculate_pnl()
            results.positions.append(current_position)
            
            realized_pnl_today = current_position.net_pnl
            equity += current_position.net_pnl
            
            # Reset state
            position_status = PositionStatus.FLAT
            current_position = None
            entry_price = None
            entry_direction = None
            
            # Log daily metrics with scanner status
            daily_metric = DailyMetrics(
                date=current_date,
                position_status=position_status,
                num_contracts=0,
                current_price=m2_settlement,
                signal=current_signal,
                forecasted_return=forecasted_return,
                atr=m2_atr,
                stop_loss=None,
                carry_spread=carry_spread,
                carry_multiplier=0.0,
                atr_scanner_triggered=scanner_results['atr']['triggered'],
                structure_scanner_triggered=scanner_results['structure']['triggered'],
                any_scanner_triggered=any_scanner_triggered,
                unrealized_pnl=0.0,
                realized_pnl=realized_pnl_today,
                roll_pnl=0.0,
                daily_total_pnl=realized_pnl_today,
                cumulative_pnl=equity - initial_equity,
                equity=equity
            )
            results.daily_metrics.append(daily_metric)
            
            continue  # Skip to next day after scanner-triggered exit
        
        # If scanners triggered but no position held, block new entries
        if any_scanner_triggered:
            current_signal = SignalType.NEUTRAL.value
        
        # ====================================================================
        # PATH A: POSITION HELD - CHECK EXIT CRITERIA
        # ====================================================================
        if current_position is not None:
            
            # ----------------------------------------------------------------
            # UPDATE TRAILING STOP DAILY
            # ----------------------------------------------------------------
            from config import TRAILING_STOP_ENABLED
            
            if TRAILING_STOP_ENABLED:
                old_stop = current_position.current_stop
                new_stop = current_position.update_trailing_stop(m2_settlement, m2_atr)
                
                # Optional: Log stop movements for debugging
                # if new_stop != old_stop:
                #     direction = "LONG" if "LONG" in current_position.entry_trade.action else "SHORT"
                #     print(f"  [{current_date.strftime('%Y-%m-%d')}] Trailing stop moved: {old_stop:.4f} → {new_stop:.4f} ({direction})")
            
            # ----------------------------------------------------------------
            # EXIT CRITERION 1: Roll (Forced Exit - 5 days before FND)
            # ----------------------------------------------------------------
            needs_roll, next_m2, fnd_date = should_roll(current_date, current_m2_contract, fnd_calendar, contract_data)
            
            if needs_roll and next_m2 is not None:
                # Execute roll at SETTLEMENT price (end of day)
                old_price = m2_settlement
                
                if next_m2 in contract_data:
                    new_m2_data = contract_data[next_m2]
                    new_m2_today = new_m2_data[new_m2_data['Date'] == current_date]
                    
                    if not new_m2_today.empty:
                        new_price = new_m2_today.iloc[0]['Settlement']
                        
                        # Close old position, open new position
                        exit_trade, entry_trade, roll_event = execute_roll(
                            current_position,
                            current_m2_contract,
                            next_m2,
                            old_price,
                            new_price,
                            current_date,
                            roll_id_counter,
                            fnd_date
                        )
                        
                        # Close old position
                        current_position.exit_trade = exit_trade
                        current_position.exit_reason = "ROLL"
                        current_position.calculate_pnl()
                        results.positions.append(current_position)
                        
                        # Capture roll P&L for daily metrics
                        roll_pnl_today = roll_event.roll_pnl
                        realized_pnl_today = current_position.net_pnl
                        
                        # Update equity with position P&L
                        equity += current_position.net_pnl  # Position P&L (directional + costs)
                        
                        # Open new position
                        current_position = Position(
                            position_id=position_id_counter,
                            entry_trade=entry_trade
                        )
                        current_position.initialize_trailing_stop()
                        position_id_counter += 1
                        trade_id_counter += 2
                        
                        # Log roll event
                        results.roll_events.append(roll_event)
                        roll_id_counter += 1
                        
                        # Update contract tracking
                        current_m2_contract = next_m2
                        entry_price = new_price
                        
                        # Skip rest of day after roll - prevent using old contract prices
                        continue
            
            # ----------------------------------------------------------------
            # EXIT CRITERION 2: Stop-Loss (1.5 × ATR, Trailing Stop)
            # ----------------------------------------------------------------
            # Use CURRENT trailing stop, not the original entry stop
            stop_price = current_position.current_stop
            direction = "LONG" if "LONG" in current_position.entry_trade.action else "SHORT"
            
            # Check if stop was triggered intraday using high/low
            stop_triggered = False
            
            if stop_price is not None:
                if direction == "LONG":
                    # LONG stop: check if price fell to/below stop during the day
                    if m2_low <= stop_price:
                        stop_triggered = True
                        
                elif direction == "SHORT":
                    # SHORT stop: check if price rose to/above stop during the day
                    if m2_high >= stop_price:
                        stop_triggered = True
            
            if stop_triggered:
                # Stop was triggered intraday
                # Now check if we can fill at settlement using stop-limit logic
                
                exit_filled = True  # Default: assume stop-market behavior (always fills)
                exit_reason = "STOP_LIMIT_FILLED"
                
                if STOP_LIMIT_ENABLED:
                    # Calculate limit band using average overnight spread
                    limit_band = calculate_limit_band(m2_avg_overnight_spread)
                    
                    # Check if settlement price is within acceptable limit band
                    if direction == "LONG":
                        # For LONG: stop triggered if price fell to/below stop
                        # Check if settlement didn't gap too far below stop price
                        if m2_settlement < (stop_price - limit_band):
                            # Settlement is beyond limit band - order doesn't fill
                            exit_filled = False
                            exit_reason = "STOP_LIMIT_NOT_FILLED"
                    
                    elif direction == "SHORT":
                        # For SHORT: stop triggered if price rose to/above stop
                        # Check if settlement didn't gap too far above stop price
                        if m2_settlement > (stop_price + limit_band):
                            # Settlement is beyond limit band - order doesn't fill
                            exit_filled = False
                            exit_reason = "STOP_LIMIT_NOT_FILLED"
                
                # Only execute exit if stop-limit order would have filled
                if exit_filled:
                    # Exit at settlement price (end of day)
                    exit_trade = Trade(
                        trade_id=trade_id_counter,
                        date=current_date,
                        action="CLOSE_LONG" if direction == "LONG" else "CLOSE_SHORT",
                        contract_code=current_m2_contract,
                        price=m2_settlement,
                        num_contracts=current_position.entry_trade.num_contracts
                    )
                    trade_id_counter += 1
                    
                    current_position.exit_trade = exit_trade
                    current_position.exit_reason = exit_reason
                    current_position.calculate_pnl()
                    results.positions.append(current_position)
                    
                    # Update equity
                    equity += current_position.net_pnl
                    
                    # Reset state
                    position_status = PositionStatus.FLAT
                    current_position = None
                    entry_price = None
                    entry_direction = None
                    
                    # Log daily metrics and continue
                    realized_pnl_value = current_position.net_pnl if current_position else 0.0
                    daily_metric = DailyMetrics(
                        date=current_date,
                        position_status=position_status,
                        num_contracts=0,
                        current_price=m2_settlement,
                        signal=current_signal,
                        forecasted_return=forecasted_return,
                        atr=m2_atr,
                        stop_loss=None,
                        carry_spread=carry_spread,
                        carry_multiplier=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=realized_pnl_value,
                        roll_pnl=0.0,
                        daily_total_pnl=realized_pnl_value,
                        cumulative_pnl=equity - initial_equity,
                        equity=equity
                    )
                    results.daily_metrics.append(daily_metric)
                    
                    continue  # Skip to next day
                
                else:
                    # Stop-limit order did NOT fill - position remains open
                    # Continue with normal position updates (trailing stop, unrealized P&L, etc.)
                    # The stop will be checked again tomorrow
                    pass  # Fall through to normal position maintenance logic below
            
            # ----------------------------------------------------------------
            # EXIT CRITERION 3: Signal Reversal
            # ----------------------------------------------------------------
            # Long position closed if signal flips to Neutral or Short
            # Short position closed if signal flips to Neutral or Long
            carry_multiplier = apply_carry_filter(current_signal, carry_spread)
            action = determine_action(position_status, current_signal, carry_multiplier)
            
            # Handle exits and reversals for existing positions
            if action in ["CLOSE_LONG", "CLOSE_SHORT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT"]:
                # Close existing position at settlement (end of day)
                exit_action = "CLOSE_LONG" if "LONG" in current_position.entry_trade.action else "CLOSE_SHORT"
                
                exit_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action=exit_action,
                    contract_code=current_m2_contract,
                    price=m2_settlement,  # Exit at settlement
                    num_contracts=current_position.entry_trade.num_contracts
                )
                trade_id_counter += 1
                
                current_position.exit_trade = exit_trade
                current_position.exit_reason = "SIGNAL_REVERSAL"
                current_position.calculate_pnl()
                results.positions.append(current_position)
                
                realized_pnl_today = current_position.net_pnl
                equity += current_position.net_pnl
                
                # Reset state
                position_status = PositionStatus.FLAT
                current_position = None
                entry_price = None
                entry_direction = None
                
                # If reversing, open new position in opposite direction at settlement
                if action == "REVERSE_TO_LONG":
                    # Calculate position size for new long
                    num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier, risk_percent=risk_percent)   #NEW
                    
                    if num_contracts > 0:
                        stop_price = calculate_stop_loss(m2_settlement, m2_atr_yesterday, "LONG")
                        
                        entry_trade = Trade(
                            trade_id=trade_id_counter,
                            date=current_date,
                            action="OPEN_LONG",
                            contract_code=current_m2_contract,
                            price=m2_settlement,  # Enter at settlement
                            num_contracts=num_contracts,
                            stop_loss=stop_price,
                            atr=m2_atr,
                            carry_spread=carry_spread,
                            carry_multiplier=carry_multiplier
                        )
                        trade_id_counter += 1
                        
                        current_position = Position(
                            position_id=position_id_counter,
                            entry_trade=entry_trade
                        )
                        current_position.initialize_trailing_stop()
                        position_id_counter += 1
                        
                        position_status = PositionStatus.LONG
                        entry_price = m2_settlement
                        entry_direction = "LONG"
                
                elif action == "REVERSE_TO_SHORT":
                    # Calculate position size for new short
                    num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier, risk_percent=risk_percent)   #NEW
                    
                    if num_contracts > 0:
                        stop_price = calculate_stop_loss(m2_settlement, m2_atr_yesterday, "SHORT")
                        
                        entry_trade = Trade(
                            trade_id=trade_id_counter,
                            date=current_date,
                            action="OPEN_SHORT",
                            contract_code=current_m2_contract,
                            price=m2_settlement,  # Enter at settlement
                            num_contracts=num_contracts,
                            stop_loss=stop_price,
                            atr=m2_atr,
                            carry_spread=carry_spread,
                            carry_multiplier=carry_multiplier
                        )
                        trade_id_counter += 1
                        
                        current_position = Position(
                            position_id=position_id_counter,
                            entry_trade=entry_trade
                        )
                        current_position.initialize_trailing_stop()
                        position_id_counter += 1
                        
                        position_status = PositionStatus.SHORT
                        entry_price = m2_settlement
                        entry_direction = "SHORT"
        
        # ====================================================================
        # PATH B: NO POSITION (FLAT) - CHECK ENTRY CRITERIA
        # ====================================================================
        else:
            # Initialize contract tracking if first trade
            if current_m2_contract is None:
                current_m2_contract = m2_contract
            
            # ----------------------------------------------------------------
            # ENTRY CRITERION: Signal (forecasted return > threshold)
            #                  AND Carry Filter (position sizing multiplier)
            # ----------------------------------------------------------------
            carry_multiplier = apply_carry_filter(current_signal, carry_spread)
            action = determine_action(position_status, current_signal, carry_multiplier)
            
            # Execute entry trades only if we're flat - enter at settlement (end of day)
            if action == "OPEN_LONG":
                # Calculate position size using YESTERDAY's ATR (known before market open)
                num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier, risk_percent=risk_percent)   #NEW
                
                if num_contracts > 0:
                    stop_price = calculate_stop_loss(m2_settlement, m2_atr_yesterday, "LONG")
                    
                    entry_trade = Trade(
                        trade_id=trade_id_counter,
                        date=current_date,
                        action="OPEN_LONG",
                        contract_code=current_m2_contract,
                        price=m2_settlement,  # Enter at settlement (end of day)
                        num_contracts=num_contracts,
                        stop_loss=stop_price,
                        atr=m2_atr,
                        carry_spread=carry_spread,
                        carry_multiplier=carry_multiplier
                    )
                    trade_id_counter += 1
                    
                    current_position = Position(
                        position_id=position_id_counter,
                        entry_trade=entry_trade
                    )
                    current_position.initialize_trailing_stop()
                    position_id_counter += 1
                    
                    position_status = PositionStatus.LONG
                    entry_price = m2_settlement
                    entry_direction = "LONG"
            
            elif action == "OPEN_SHORT":
                num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier, risk_percent=risk_percent)   #NEW
                
                if num_contracts > 0:
                    stop_price = calculate_stop_loss(m2_settlement, m2_atr_yesterday, "SHORT")
                    
                    entry_trade = Trade(
                        trade_id=trade_id_counter,
                        date=current_date,
                        action="OPEN_SHORT",
                        contract_code=current_m2_contract,
                        price=m2_settlement,  # Enter at settlement (end of day)
                        num_contracts=num_contracts,
                        stop_loss=stop_price,
                        atr=m2_atr,
                        carry_spread=carry_spread,
                        carry_multiplier=carry_multiplier
                    )
                    trade_id_counter += 1
                    
                    current_position = Position(
                        position_id=position_id_counter,
                        entry_trade=entry_trade
                    )
                    current_position.initialize_trailing_stop()
                    position_id_counter += 1
                    
                    position_status = PositionStatus.SHORT
                    entry_price = m2_settlement
                    entry_direction = "SHORT"
        
        # ====================================================================
        # CALCULATE UNREALIZED P&L (use settlement for mark-to-market)
        # ====================================================================
        unrealized_pnl = 0.0
        if current_position is not None and entry_price is not None:
            price_change = m2_settlement - entry_price if entry_direction == "LONG" else entry_price - m2_settlement
            unrealized_pnl = price_change * CONTRACT_SIZE * current_position.entry_trade.num_contracts
        
        # ====================================================================
        # LOG DAILY METRICS (use settlement for price tracking)
        # ====================================================================
        # Calculate daily total P&L (realized + roll P&L)
        daily_total = realized_pnl_today + roll_pnl_today
        
        daily_metric = DailyMetrics(
            date=current_date,
            position_status=position_status,
            num_contracts=current_position.entry_trade.num_contracts if current_position else 0,
            current_price=m2_settlement,
            signal=current_signal,
            forecasted_return=forecasted_return,
            atr=m2_atr,
            stop_loss=current_position.current_stop if current_position else None,  # Use current trailing stop
            carry_spread=carry_spread,
            carry_multiplier=carry_multiplier if 'carry_multiplier' in locals() else 0.0,
            atr_scanner_triggered=scanner_results['atr']['triggered'] if 'scanner_results' in locals() else False,
            structure_scanner_triggered=scanner_results['structure']['triggered'] if 'scanner_results' in locals() else False,
            any_scanner_triggered=any_scanner_triggered if 'any_scanner_triggered' in locals() else False,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl_today,
            roll_pnl=roll_pnl_today,
            daily_total_pnl=daily_total,
            cumulative_pnl=equity - initial_equity,
            equity=equity
        )
        
        results.daily_metrics.append(daily_metric)
    
    # ====================================================================
    # CLOSE ANY OPEN POSITIONS AT END OF BACKTEST
    # ====================================================================
    if current_position is not None:
        last_date = signal_data.iloc[-1]['Date']
        last_price = m2_settlement
        
        exit_trade = Trade(
            trade_id=trade_id_counter,
            date=last_date,
            action="CLOSE_LONG" if entry_direction == "LONG" else "CLOSE_SHORT",
            contract_code=current_m2_contract,
            price=last_price,
            num_contracts=current_position.entry_trade.num_contracts
        )
        
        current_position.exit_trade = exit_trade
        current_position.exit_reason = "END_OF_BACKTEST"
        current_position.calculate_pnl()
        results.positions.append(current_position)
        
        equity += current_position.net_pnl
    
    # ====================================================================
    # 10. CALCULATE SUMMARY STATISTICS
    # ====================================================================
    results.final_equity = equity
    results.calculate_summary_statistics()
    
    print(f"\n\n✓ Backtest simulation complete!")
    print(f"   Total positions: {len(results.positions)}")
    print(f"   Total rolls: {len(results.roll_events)}")
    print(f"   Final equity: ${equity:,.2f}")
    print("="*80)
    
    return results


# ============================================================================
# SUMMARY & REPORTING
# ============================================================================

def generate_backtest_summary(results):
    """
    Generate comprehensive backtest summary with:
    - Performance statistics printed to console
    - CSV exports of all results
    - Equity curve and returns chart
    
    Args:
        results: BacktestResults object
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    print("\n" + "="*80)
    print("GENERATING BACKTEST SUMMARY & REPORTS")
    print("="*80)
    
    # ========================================================================
    # 1. PRINT PERFORMANCE SUMMARY
    # ========================================================================
    results.print_summary()
    
    # ========================================================================
    # 2. EXPORT TO CSV FILES
    # ========================================================================
    print("\n[Exporting Results to CSV]")
    result_dfs = results.to_dataframes()
    
    from config import POSITIONS_OUTPUT, ROLLS_OUTPUT, DAILY_OUTPUT, SUMMARY_OUTPUT
    
    if len(result_dfs['positions']) > 0:
        result_dfs['positions'].to_csv(POSITIONS_OUTPUT, index=False)
        print(f"   ✓ Positions exported to {POSITIONS_OUTPUT}")
    else:
        print("   ⚠ No positions to export")
    
    if len(result_dfs['rolls']) > 0:
        result_dfs['rolls'].to_csv(ROLLS_OUTPUT, index=False)
        print(f"   ✓ Rolls exported to {ROLLS_OUTPUT}")
    else:
        print("   ⚠ No rolls to export")
    
    if len(result_dfs['daily']) > 0:
        result_dfs['daily'].to_csv(DAILY_OUTPUT, index=False)
        print(f"   ✓ Daily metrics exported to {DAILY_OUTPUT}")
    else:
        print("   ⚠ No daily metrics to export")
    
    result_dfs['summary'].to_csv(SUMMARY_OUTPUT)
    print(f"   ✓ Summary statistics exported to {SUMMARY_OUTPUT}")
    
    # ========================================================================
    # 3. GENERATE SEPARATE PERFORMANCE CHARTS
    # ========================================================================
    
    if len(result_dfs['daily']) > 0:
        print("\n[Generating Performance Charts]")
        
        # Create charts subdirectory
        try:
            from config import CHART_OUTPUT
            charts_dir = CHART_OUTPUT.replace('performance_chart.png', 'charts')
        except (ImportError, AttributeError):
            charts_dir = 'output/charts'
        
        import os
        os.makedirs(charts_dir, exist_ok=True)
        
        daily_df = result_dfs['daily'].copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # ====================================================================
        # Chart 1: Equity Curve
        # ====================================================================
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        
        ax1.plot(daily_df['date'], daily_df['equity'], 
                linewidth=2, color='#2E86AB', label='Equity')
        ax1.axhline(y=results.initial_equity, color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Equity')
        
        ax1.set_title('Account Equity Over Time', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Equity ($)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add performance metrics text box
        metrics_text = (
            f"Initial: ${results.initial_equity:,.0f}\n"
            f"Final: ${results.final_equity:,.0f}\n"
            f"Total Return: {results.total_return:.2f}%\n"
            f"Annualized Return: {results.annualized_return:.2f}%\n"
            f"Max Drawdown: {results.max_drawdown:.2f}%\n"
            f"Sharpe Ratio: {results.annualized_sharpe_ratio:.2f}"
        )
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        equity_chart = f'{charts_dir}/equity_curve.png'
        plt.savefig(equity_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Equity curve saved to {equity_chart}")
        
        # ====================================================================
        # Chart 2: Cumulative P&L
        # ====================================================================
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        
        ax2.plot(daily_df['date'], daily_df['cumulative_pnl'], 
                linewidth=2, color='#A23B72', label='Cumulative P&L')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.fill_between(daily_df['date'], daily_df['cumulative_pnl'], 0, 
                         where=(daily_df['cumulative_pnl'] >= 0), 
                         color='green', alpha=0.3, label='Profit')
        ax2.fill_between(daily_df['date'], daily_df['cumulative_pnl'], 0, 
                         where=(daily_df['cumulative_pnl'] < 0), 
                         color='red', alpha=0.3, label='Loss')
        
        ax2.set_title('Cumulative Profit & Loss', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add P&L metrics text box
        pnl_text = (
            f"Net P&L: ${results.net_pnl:,.0f}\n"
            f"Directional P&L: ${results.total_directional_pnl:,.0f}\n"
            f"Roll P&L: ${results.total_roll_pnl:,.0f}\n"
            f"Transaction Costs: ${results.total_transaction_costs:,.0f}"
        )
        ax2.text(0.02, 0.98, pnl_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        pnl_chart = f'{charts_dir}/cumulative_pnl.png'
        plt.savefig(pnl_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Cumulative P&L saved to {pnl_chart}")
        
        # ====================================================================
        # Chart 3: Daily Returns Distribution
        # ====================================================================
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        # Calculate daily returns (%)
        daily_df['daily_return_pct'] = (daily_df['daily_total_pnl'] / 
                                         daily_df['equity'].shift(1)) * 100
        daily_df['daily_return_pct'] = daily_df['daily_return_pct'].fillna(0)
        
        # Bar chart of daily returns
        colors = ['green' if x >= 0 else 'red' for x in daily_df['daily_return_pct']]
        ax3.bar(daily_df['date'], daily_df['daily_return_pct'], 
               color=colors, alpha=0.6, width=1.0)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Daily Return (%)', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Add statistics text box
        returns_stats = (
            f"Mean: {daily_df['daily_return_pct'].mean():.3f}%\n"
            f"Std Dev: {daily_df['daily_return_pct'].std():.3f}%\n"
            f"Best Day: {daily_df['daily_return_pct'].max():.2f}%\n"
            f"Worst Day: {daily_df['daily_return_pct'].min():.2f}%\n"
            f"Win Days: {(daily_df['daily_return_pct'] > 0).sum()}\n"
            f"Loss Days: {(daily_df['daily_return_pct'] < 0).sum()}"
        )
        ax3.text(0.02, 0.98, returns_stats, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        returns_chart = f'{charts_dir}/daily_returns.png'
        plt.savefig(returns_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Daily returns saved to {returns_chart}")
        
        # ====================================================================
        # Additional Statistics
        # ====================================================================
        print("\n[Additional Performance Metrics]")
        print(f"   Total Trading Days: {len(daily_df)}")
        print(f"   Days in Position: {(daily_df['num_contracts'] > 0).sum()}")
        print(f"   Days Flat: {(daily_df['num_contracts'] == 0).sum()}")
        print(f"   Average Daily Return: {daily_df['daily_return_pct'].mean():.4f}%")
        print(f"   Daily Return Std Dev: {daily_df['daily_return_pct'].std():.4f}%")
        print(f"   Best Day: {daily_df['daily_return_pct'].max():.2f}% on {daily_df.loc[daily_df['daily_return_pct'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
        print(f"   Worst Day: {daily_df['daily_return_pct'].min():.2f}% on {daily_df.loc[daily_df['daily_return_pct'].idxmin(), 'date'].strftime('%Y-%m-%d')}")
        
        # Calculate win/loss streaks
        daily_df['is_win'] = daily_df['daily_total_pnl'] > 0
        daily_df['streak'] = (daily_df['is_win'] != daily_df['is_win'].shift()).cumsum()
        win_streaks = daily_df[daily_df['is_win']].groupby('streak').size()
        loss_streaks = daily_df[~daily_df['is_win']].groupby('streak').size()
        
        if len(win_streaks) > 0:
            print(f"   Longest Win Streak: {win_streaks.max()} days")
        if len(loss_streaks) > 0:
            print(f"   Longest Loss Streak: {loss_streaks.max()} days")
    
    else:
        print("\n⚠ No daily metrics available for chart generation")
    
    print("\n" + "="*80)

def plot_signal_distribution_pie(backtest_data, outpath="output/signal_distribution.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Compute distribution from data
    counts = backtest_data['Signal'].value_counts(dropna=False)

    labels = ['Neutral', 'Long', 'Short']
    values = [
        counts.get('NEUTRAL', 0),
        counts.get('LONG', 0),
        counts.get('SHORT', 0),
    ]

    # Color palette (professional but visually appealing)
    colors = ['#4C72B0',  '#55A868', '#C44E52']  
    # blue, green, red

    fig, ax = plt.subplots(figsize=(7, 7))

    # Slight donut effect (looks good in reports)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.05,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.2)
    )

    # Improve text contrast & professionalism
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title('Signal Distribution', fontsize=14, pad=14)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_factor_miniplots(data, outpath="output/factors_three_panel.png"):
    """Three stacked mini-plots for your drivers over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Drivers Overview', y=0.98, fontsize=14)

    axes[0].plot(df['Date'], df['DXY_logret'], linewidth=1.2)
    axes[0].set_ylabel('DXY\nLog Returns')
    axes[0].grid(alpha=0.25)

    axes[1].plot(df['Date'], df['PMI_change'], linewidth=1.2)
    axes[1].set_ylabel('PMI\nΔ (Daily Interp.)')
    axes[1].grid(alpha=0.25)

    axes[2].plot(df['Date'], df['STOCKS_logret'], linewidth=1.2)
    axes[2].set_ylabel('LME Stocks\nLog Returns')
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel('Date')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pnl_waterfall(outpath="output/pnl_waterfall.png",
                       directional=866000, roll_pnl=-55500, tx_costs=-69680):
    """Waterfall: Start at $0 → Directional → Roll → Costs → Total."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    steps = ['Start', 'Directional P&L', 'Roll P&L', 'Transaction Costs', 'Total Net P&L']
    vals  = [0, directional, roll_pnl, tx_costs]  # last bar is computed total

    # cumulative bases for intermediate bars
    bases = np.concatenate(([0], np.cumsum(vals[:-1])))
    totals = np.cumsum(vals)[-1]  # final total (e.g., 796,320)

    # Build bar heights; final bar is total from zero
    heights = [vals[1], vals[2], vals[3], totals]  # omit the initial 0 bar

    # Colors: green for positive, red for negative; final total in a neutral dark
    colors = []
    for h in heights[:-1]:
        colors.append('#2ca02c' if h >= 0 else '#d62728')
    colors.append('#1f77b4')  # total

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw intermediate bars
    x = np.arange(len(steps))
    # Start bar (invisible baseline)
    ax.bar(x[0], 0, color='none')

    # Directional / Roll / Costs (stacked on their bases)
    ax.bar(x[1], heights[0], bottom=bases[1], color=colors[0], width=0.6)
    ax.bar(x[2], heights[1], bottom=bases[2], color=colors[1], width=0.6)
    ax.bar(x[3], heights[2], bottom=bases[3], color=colors[2], width=0.6)

    # Connector lines between bars
    for i in range(1, 4):
        y = bases[i] + max(0, heights[i-1])
        ax.plot([x[i]-0.3, x[i]+0.3], [y, y], color='gray', linewidth=1)

    # Final total bar (from zero)
    ax.bar(x[4], heights[3], bottom=0, color=colors[3], width=0.6)

    # Labels & formatting
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=0)
    ax.set_ylabel('P&L (USD)')
    ax.set_title('P&L Attribution Waterfall')

    # Value annotations
    def fmt_usd(v): 
        return f"${v:,.0f}"
    ann = [
        (x[1], bases[1] + heights[0]),   # Directional
        (x[2], bases[2] + (heights[1] if heights[1] > 0 else 0)),  # Roll
        (x[3], bases[3] + (heights[2] if heights[2] > 0 else 0)),  # Costs
        (x[4], heights[3])               # Total
    ]
    texts = [directional, roll_pnl, tx_costs, np.sum(vals)]
    va = ['bottom', 'top' if roll_pnl < 0 else 'bottom', 'top' if tx_costs < 0 else 'bottom', 'bottom']
    for (xi, yi), v, valign in zip(ann, texts, va):
        ax.text(xi, yi + (20000 if valign=='bottom' else -20000), fmt_usd(v),
                ha='center', va=valign, fontsize=10)

    # Currency y-axis
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda n, _: f'${n:,.0f}')
    )
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_single_backtest(period_name, backtest_start, backtest_end, initial_equity):
    """
    Run a single backtest for a given period
    
    Args:
        period_name: Name of the period (e.g., "IN_SAMPLE_2024", "OUT_OF_SAMPLE_2025")
        backtest_start: Start date for backtest
        backtest_end: End date for backtest
        initial_equity: Starting equity
    
    Returns:
        BacktestResults object
    """
    import os
    
    # Create output directory for this period
    period_dir = f'output/{period_name.lower()}'
    os.makedirs(period_dir, exist_ok=True)
    
    # Set up logging to file
    log_filename = f'{period_dir}/backtest_output.txt'
    sys.stdout = ConsoleLogger(log_filename)
    
    print("\n")
    print("="*80)
    print(" " * 15 + "FINS3666 - HG COPPER FUTURES BACKTEST")
    print(" " * 20 + "3-Factor Regression Strategy")
    print(" " * 25 + f"[{period_name}]")
    print("="*80)
    print(f"\nOutput is being saved to: {log_filename}")
    
    # ----------------------------------------------------------------------
    # Calculate how far back we need to load data based on REGRESSION_WINDOW_DAYS
    # ----------------------------------------------------------------------
    backtest_start_date = pd.to_datetime(backtest_start)

    # Each trading year ~252 days. 756 trading days ≈ 3 years.
    # Use 1.5x as a cushion to account for weekends/holidays/NaNs.
    lookback_calendar_days = int(REGRESSION_WINDOW_DAYS * 1.5)  # e.g. 756 * 1.5 = 1134 days

    data_start_date = backtest_start_date - timedelta(days=lookback_calendar_days)
    DATA_START = data_start_date.strftime('%Y-%m-%d')

    print(f"\nBacktest Period: {backtest_start} to {backtest_end}")
    print(f"Data Loading Period: {DATA_START} to {backtest_end}")
    print(f"  (Need ~{REGRESSION_WINDOW_DAYS/252:.1f} years of history for rolling regression)")
    print(f"Initial Equity: ${initial_equity:,.0f}")
    
    # Step 1: Load and preprocess data (includes training period)
    data = load_and_preprocess_data(start_date=DATA_START, end_date=backtest_end)
    
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
    backtest_data = generate_signals(data, backtest_start=backtest_start, window_days=REGRESSION_WINDOW_DAYS)
    
    # Save processed data for inspection
    output_file = f'{period_dir}/signals.csv'
    backtest_data.to_csv(output_file, index=False)
    print(f"\n✓ Backtest data with signals saved to: {output_file}")
    
    # Step 2.5: Generate correlation matrix chart for regression variables
    generate_correlation_matrix_chart(backtest_data, period_dir)
    
    # Step 3: Load FND calendar
    fnd_calendar = load_fnd_calendar()
    
    # Step 4: Build contract database
    contract_data = build_contract_database(fnd_calendar, DATA_START, backtest_end)
    
    # Step 5: Add ATR and overnight spread to all contracts
    print("\n[Calculating ATR and overnight spreads for all contracts]")
    for contract_code, df in contract_data.items():
        contract_data[contract_code] = add_atr_to_contract(df, period=ATR_PERIOD)
        contract_data[contract_code] = add_overnight_spread_to_contract(df)
    print("   ✓ ATR and overnight spreads calculated for all contracts")
    
    # Step 6: Run backtest
    results = run_backtest(backtest_data, contract_data, fnd_calendar, initial_equity)
    
    # Step 7: Generate comprehensive summary with charts (will save to period-specific folder)
    # Temporarily rename output files to save in period-specific directory
    import config
    original_positions = config.POSITIONS_OUTPUT
    original_rolls = config.ROLLS_OUTPUT
    original_daily = config.DAILY_OUTPUT
    original_summary = config.SUMMARY_OUTPUT
    
    config.POSITIONS_OUTPUT = f'{period_dir}/positions.csv'
    config.ROLLS_OUTPUT = f'{period_dir}/rolls.csv'
    config.DAILY_OUTPUT = f'{period_dir}/daily_metrics.csv'
    config.SUMMARY_OUTPUT = f'{period_dir}/summary.csv'
    config.CHART_OUTPUT = f'{period_dir}/performance_chart.png'

    plot_signal_distribution_pie(backtest_data)
    plot_factor_miniplots(data)
    plot_pnl_waterfall()
    
    generate_backtest_summary(results)
    
    # Restore original output paths
    config.POSITIONS_OUTPUT = original_positions
    config.ROLLS_OUTPUT = original_rolls
    config.DAILY_OUTPUT = original_daily
    config.SUMMARY_OUTPUT = original_summary
    
    print("\n" + "="*80)
    print(f"{period_name} BACKTEST COMPLETE")
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✓ {period_name} output saved to: {log_filename}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 10 + "FINS3666 - HG COPPER FUTURES STRATEGY BACKTEST")
    print(" " * 15 + "Running Both In-Sample and Out-of-Sample Tests")
    print("="*80)
    
    # ========================================================================
    # RUN 1: IN-SAMPLE PERIOD (2024)
    # ========================================================================
    print("\n" + "🔵 "*40)
    print("RUNNING IN-SAMPLE BACKTEST (2024)")
    print("🔵 "*40 + "\n")
    
    results_2024 = run_single_backtest(
        period_name="IN_SAMPLE_2024",
        backtest_start='2024-01-02',
        backtest_end='2024-12-31',
        initial_equity=INITIAL_EQUITY
    )
    
    # ========================================================================
    # RUN 2: OUT-OF-SAMPLE PERIOD (2025)
    # ========================================================================
    print("\n" + "🟢 "*40)
    print("RUNNING OUT-OF-SAMPLE BACKTEST (2025)")
    print("🟢 "*40 + "\n")
    
    results_2025 = run_single_backtest(
        period_name="OUT_OF_SAMPLE_2025",
        backtest_start='2025-01-02',
        backtest_end='2025-10-31',
        initial_equity=INITIAL_EQUITY
    )
    
    # ========================================================================
    # COMBINED SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print(" " * 25 + "COMBINED RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'IN-SAMPLE (2024)':-^40} | {'OUT-OF-SAMPLE (2025)':-^40}")
    print("-"*80)
    print(f"{'Initial Equity:':<30} ${results_2024.initial_equity:>12,.0f} | ${results_2025.initial_equity:>12,.0f}")
    print(f"{'Final Equity:':<30} ${results_2024.final_equity:>12,.0f} | ${results_2025.final_equity:>12,.0f}")
    print(f"{'Total Return:':<30} {results_2024.total_return:>11.2f}% | {results_2025.total_return:>11.2f}%")
    print(f"{'Annualized Return:':<30} {results_2024.annualized_return:>11.2f}% | {results_2025.annualized_return:>11.2f}%")
    print(f"{'Net P&L:':<30} ${results_2024.net_pnl:>12,.0f} | ${results_2025.net_pnl:>12,.0f}")
    print(f"{'Trading Period:':<30} {results_2024.trading_days:>9} days | {results_2025.trading_days:>9} days")
    print("-"*80)
    print(f"{'Win Rate:':<30} {results_2024.win_rate*100:>11.2f}% | {results_2025.win_rate*100:>11.2f}%")
    print(f"{'Profit Factor:':<30} {results_2024.profit_factor:>15.2f} | {results_2025.profit_factor:>15.2f}")
    print(f"{'Sharpe Ratio (Annualized):':<30} {results_2024.annualized_sharpe_ratio:>15.2f} | {results_2025.annualized_sharpe_ratio:>15.2f}")
    print(f"{'Max Drawdown:':<30} {results_2024.max_drawdown:>11.2f}% | {results_2025.max_drawdown:>11.2f}%")
    print("-"*80)
    print(f"{'Total Trades:':<30} {results_2024.total_trades:>15} | {results_2025.total_trades:>15}")
    print(f"{'Number of Rolls:':<30} {len(results_2024.roll_events):>15} | {len(results_2025.roll_events):>15}")
    print("="*80)
    
    # ========================================================================
    # STRATEGY PARAMETERS
    # ========================================================================
    print("\n" + "="*80)
    print(" " * 25 + "STRATEGY PARAMETERS USED")
    print("="*80)
    print(f"\n{'Signal Generation:':<40}")
    print(f"   Signal Threshold:                    ±{SIGNAL_THRESHOLD}")
    print(f"   Regression Window:                   {REGRESSION_WINDOW_DAYS} days (~{REGRESSION_WINDOW_DAYS/252:.1f} years)")
    print(f"\n{'Risk Management:':<40}")
    print(f"   ATR Period:                          {ATR_PERIOD} days")
    print(f"   ATR Stop Multiplier:                 {ATR_STOP_MULTIPLIER}x")
    print(f"   Risk per Trade:                      {RISK_PERCENT*100}%")
    print(f"   Trailing Stop:                       {'Enabled' if TRAILING_STOP_ENABLED else 'Disabled'}")
    print(f"   Stop-Limit Orders:                   {'Enabled' if STOP_LIMIT_ENABLED else 'Disabled'}")
    if STOP_LIMIT_ENABLED:
        print(f"      - Limit Band (min):               {STOP_LIMIT_BAND_TICKS} ticks (${STOP_LIMIT_BAND_TICKS * 0.0005}/lb)")
        print(f"      - Limit Band (multiplier):        {STOP_LIMIT_BAND_MULTIPLIER}x avg overnight spread")
    print(f"\n{'Position Sizing (Carry Filter):':<40}")
    print(f"   Half Position Threshold:             ${CARRY_THRESHOLD_HALF}/lb")
    print(f"   Quarter Position Threshold:          ${CARRY_THRESHOLD_QUARTER}/lb")
    print(f"\n{'Contract Rolling:':<40}")
    print(f"   Days Before FND to Roll:             {DAYS_BEFORE_FND_TO_ROLL} days")
    print(f"\n{'Transaction Costs:':<40}")
    print(f"   Commission per Contract:             ${COMMISSION:.2f}")
    print(f"   Slippage per Contract:               ${SLIPPAGE:.2f}")
    print(f"   Total per Round-Turn:                ${TOTAL_TRANSACTION_COST:.2f}")
    print("="*80)
    
    print("\n📁 Output Files Generated:")
    print("   In-Sample (2024):")
    print("      - output/in_sample_2024/backtest_output.txt")
    print("      - output/in_sample_2024/positions.csv")
    print("      - output/in_sample_2024/rolls.csv")
    print("      - output/in_sample_2024/daily_metrics.csv")
    print("      - output/in_sample_2024/summary.csv")
    print("      - output/in_sample_2024/signals.csv")
    print("      - output/in_sample_2024/performance_chart.png")
    print("      - output/in_sample_2024/correlation_matrix.png")
    print("\n   Out-of-Sample (2025):")
    print("      - output/out_of_sample_2025/backtest_output.txt")
    print("      - output/out_of_sample_2025/positions.csv")
    print("      - output/out_of_sample_2025/rolls.csv")
    print("      - output/out_of_sample_2025/daily_metrics.csv")
    print("      - output/out_of_sample_2025/summary.csv")
    print("      - output/out_of_sample_2025/signals.csv")
    print("      - output/out_of_sample_2025/performance_chart.png")
    print("      - output/out_of_sample_2025/correlation_matrix.png")
    
    print("\n" + "="*80)
    print("✅ ALL BACKTESTS COMPLETE")
    print("="*80 + "\n")
