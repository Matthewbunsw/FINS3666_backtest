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
    CONTRACT_SIZE,
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
    STOP_LIMIT_ENABLED,
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
    if signal == SignalType.NEUTRAL.value:
        return 0.0
    
    elif signal == SignalType.LONG.value:
        # LONG hurt by contango (positive spread)
        if carry_spread <= CARRY_THRESHOLD_HALF:
            return 1.0  # Full position
        elif carry_spread <= CARRY_THRESHOLD_QUARTER:
            return 0.5  # Half position
        else:
            return 0.25  # Quarter position
    
    elif signal == SignalType.SHORT.value:
        # SHORT hurt by backwardation (negative spread)
        if carry_spread >= -CARRY_THRESHOLD_HALF:
            return 1.0  # Full position
        elif carry_spread >= -CARRY_THRESHOLD_QUARTER:
            return 0.5  # Half position
        else:
            return 0.25  # Quarter position
    
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
        new_signal: SignalType (LONG, SHORT, NEUTRAL)
        carry_multiplier: float (0.0 if stand aside)
    
    Returns:
        str: Action to take
            - "HOLD" (no change)
            - "OPEN_LONG" (enter long)
            - "OPEN_SHORT" (enter short)
            - "CLOSE_LONG" (exit long)
            - "CLOSE_SHORT" (exit short)
            - "REVERSE_TO_LONG" (close short + open long)
            - "REVERSE_TO_SHORT" (close long + open short)
    """
    # If carry filter says stand aside, treat as NEUTRAL
    if carry_multiplier == 0.0:
        new_signal = SignalType.NEUTRAL.value
    
    # FLAT (no position)
    if current_status == PositionStatus.FLAT:
        if new_signal == SignalType.LONG.value:
            return "OPEN_LONG"
        elif new_signal == SignalType.SHORT.value:
            return "OPEN_SHORT"
        else:
            return "HOLD"
    
    # LONG position
    elif current_status == PositionStatus.LONG:
        if new_signal == SignalType.LONG.value:
            return "HOLD"
        elif new_signal == SignalType.NEUTRAL.value:
            return "CLOSE_LONG"
        elif new_signal == SignalType.SHORT.value:
            # Check if we can reverse
            if carry_multiplier > 0:
                return "REVERSE_TO_SHORT"
            else:
                return "CLOSE_LONG"
    
    # SHORT position
    elif current_status == PositionStatus.SHORT:
        if new_signal == SignalType.SHORT.value:
            return "HOLD"
        elif new_signal == SignalType.NEUTRAL.value:
            return "CLOSE_SHORT"
        elif new_signal == SignalType.LONG.value:
            # Check if we can reverse
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
    position_status = PositionStatus.FLAT
    current_position = None
    current_m2_contract = None
    
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
                        
                        # Update equity with roll P&L
                        equity += roll_event.roll_pnl
                        
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
                    num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier)
                    
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
                    num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier)
                    
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
                num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier)
                
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
                num_contracts = calculate_position_size(equity, m2_atr_yesterday, carry_multiplier)
                
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
    
    if len(result_dfs['positions']) > 0:
        result_dfs['positions'].to_csv('output/positions.csv', index=False)
        print("   ✓ Positions exported to output/positions.csv")
    else:
        print("   ⚠ No positions to export")
    
    if len(result_dfs['rolls']) > 0:
        result_dfs['rolls'].to_csv('output/rolls.csv', index=False)
        print("   ✓ Rolls exported to output/rolls.csv")
    else:
        print("   ⚠ No rolls to export")
    
    if len(result_dfs['daily']) > 0:
        result_dfs['daily'].to_csv('output/daily_metrics.csv', index=False)
        print("   ✓ Daily metrics exported to output/daily_metrics.csv")
    else:
        print("   ⚠ No daily metrics to export")
    
    result_dfs['summary'].to_csv('output/summary.csv')
    print("   ✓ Summary statistics exported to output/summary.csv")
    
    # ========================================================================
    # 3. GENERATE EQUITY CURVE & RETURNS CHART
    # ========================================================================
    if len(result_dfs['daily']) > 0:
        print("\n[Generating Performance Charts]")
        
        daily_df = result_dfs['daily'].copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('HG Copper Futures Backtest - Performance Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # ====================================================================
        # Subplot 1: Equity Curve
        # ====================================================================
        ax1 = axes[0]
        ax1.plot(daily_df['date'], daily_df['equity'], 
                linewidth=2, color='#2E86AB', label='Equity')
        ax1.axhline(y=results.initial_equity, color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Equity')
        
        ax1.set_title('Account Equity Over Time', fontsize=12, fontweight='bold', pad=10)
        ax1.set_ylabel('Equity ($)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add performance metrics text box
        metrics_text = (
            f"Initial: ${results.initial_equity:,.0f}\n"
            f"Final: ${results.final_equity:,.0f}\n"
            f"Return: {results.total_return:.2f}%\n"
            f"Max DD: {results.max_drawdown:.2f}%"
        )
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ====================================================================
        # Subplot 2: Cumulative P&L
        # ====================================================================
        ax2 = axes[1]
        ax2.plot(daily_df['date'], daily_df['cumulative_pnl'], 
                linewidth=2, color='#A23B72', label='Cumulative P&L')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.fill_between(daily_df['date'], daily_df['cumulative_pnl'], 0, 
                         where=(daily_df['cumulative_pnl'] >= 0), 
                         color='green', alpha=0.3, label='Profit')
        ax2.fill_between(daily_df['date'], daily_df['cumulative_pnl'], 0, 
                         where=(daily_df['cumulative_pnl'] < 0), 
                         color='red', alpha=0.3, label='Loss')
        
        ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # ====================================================================
        # Subplot 3: Daily Returns Distribution
        # ====================================================================
        ax3 = axes[2]
        
        # Calculate daily returns (%)
        daily_df['daily_return_pct'] = (daily_df['daily_total_pnl'] / 
                                         daily_df['equity'].shift(1)) * 100
        daily_df['daily_return_pct'] = daily_df['daily_return_pct'].fillna(0)
        
        # Bar chart of daily returns
        colors = ['green' if x >= 0 else 'red' for x in daily_df['daily_return_pct']]
        ax3.bar(daily_df['date'], daily_df['daily_return_pct'], 
               color=colors, alpha=0.6, width=1.0)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax3.set_title('Daily Returns (%)', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Daily Return (%)', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Add statistics text box
        returns_stats = (
            f"Mean: {daily_df['daily_return_pct'].mean():.3f}%\n"
            f"Std Dev: {daily_df['daily_return_pct'].std():.3f}%\n"
            f"Best Day: {daily_df['daily_return_pct'].max():.2f}%\n"
            f"Worst Day: {daily_df['daily_return_pct'].min():.2f}%"
        )
        ax3.text(0.02, 0.98, returns_stats, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ====================================================================
        # Format and save
        # ====================================================================
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Rotate x-axis labels for better readability
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        # Save chart
        chart_filename = 'output/performance_chart.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Performance chart saved to {chart_filename}")
        
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
    
    # Step 3: Load FND calendar
    fnd_calendar = load_fnd_calendar()
    
    # Step 4: Build contract database
    contract_data = build_contract_database(fnd_calendar, DATA_START, BACKTEST_END)
    
    # Step 5: Add ATR and overnight spread to all contracts
    print("\n[Calculating ATR and overnight spreads for all contracts]")
    for contract_code, df in contract_data.items():
        contract_data[contract_code] = add_atr_to_contract(df, period=ATR_PERIOD)
        contract_data[contract_code] = add_overnight_spread_to_contract(df)
    print("   ✓ ATR and overnight spreads calculated for all contracts")
    
    # Step 6: Run backtest
    results = run_backtest(backtest_data, contract_data, fnd_calendar, INITIAL_EQUITY)
    
    # Step 7: Generate comprehensive summary with charts
    generate_backtest_summary(results)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✓ Output saved to: {log_filename}")
