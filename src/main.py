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
        
        # Select relevant columns and rename
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
        
        m2_price = m2_today.iloc[0]['Settlement']
        m3_price = m3_today.iloc[0]['Settlement']
        m2_atr = m2_today.iloc[0]['ATR'] if 'ATR' in m2_today.columns else np.nan
        
        # Skip if ATR not available yet
        if np.isnan(m2_atr):
            continue
        
        # ====================================================================
        # 2. CHECK FOR ROLL TRIGGER
        # ====================================================================
        needs_roll, next_m2, fnd_date = should_roll(current_date, current_m2_contract, fnd_calendar, contract_data)
        
        if needs_roll and current_position is not None and next_m2 is not None:
            # Execute roll
            old_price = m2_price
            
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
                    
                    # Update equity with roll P&L
                    equity += roll_event.roll_pnl
                    
                    # Open new position
                    current_position = Position(
                        position_id=position_id_counter,
                        entry_trade=entry_trade
                    )
                    position_id_counter += 1
                    trade_id_counter += 2
                    
                    # Log roll event
                    results.roll_events.append(roll_event)
                    roll_id_counter += 1
                    
                    # Update contract tracking
                    current_m2_contract = next_m2
                    entry_price = new_price
        
        # Update current contract if it changed
        if current_m2_contract != m2_contract and current_position is None:
            current_m2_contract = m2_contract
        
        # ====================================================================
        # 3. CHECK STOP-LOSS (if position open)
        # ====================================================================
        if current_position is not None:
            stop_price = current_position.entry_trade.stop_loss
            direction = "LONG" if "LONG" in current_position.entry_trade.action else "SHORT"
            
            if check_stop_loss(m2_price, stop_price, direction):
                # Stop hit - close position
                exit_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action="CLOSE_LONG" if direction == "LONG" else "CLOSE_SHORT",
                    contract_code=current_m2_contract,
                    price=m2_price,
                    num_contracts=current_position.entry_trade.num_contracts
                )
                trade_id_counter += 1
                
                current_position.exit_trade = exit_trade
                current_position.exit_reason = "STOP_LOSS"
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
                daily_metric = DailyMetrics(
                    date=current_date,
                    position_status=position_status,
                    num_contracts=0,
                    current_price=m2_price,
                    signal=current_signal,
                    forecasted_return=forecasted_return,
                    atr=m2_atr,
                    stop_loss=None,
                    carry_spread=calculate_carry_spread(m2_price, m3_price),
                    carry_multiplier=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=current_position.net_pnl if current_position else 0.0,
                    cumulative_pnl=equity - initial_equity,
                    equity=equity
                )
                results.daily_metrics.append(daily_metric)
                
                continue  # Skip to next day
        
        # ====================================================================
        # 4. CALCULATE CARRY SPREAD AND FILTER
        # ====================================================================
        carry_spread = calculate_carry_spread(m2_price, m3_price)
        carry_multiplier = apply_carry_filter(current_signal, carry_spread)
        
        # ====================================================================
        # 5. DETERMINE ACTION BASED ON SIGNAL
        # ====================================================================
        action = determine_action(position_status, current_signal, carry_multiplier)
        
        # ====================================================================
        # 6. EXECUTE TRADES
        # ====================================================================
        realized_pnl_today = 0.0
        
        if action == "OPEN_LONG":
            # Calculate position size
            num_contracts = calculate_position_size(equity, m2_atr, carry_multiplier)
            
            if num_contracts > 0:
                stop_price = calculate_stop_loss(m2_price, m2_atr, "LONG")
                
                entry_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action="OPEN_LONG",
                    contract_code=current_m2_contract,
                    price=m2_price,
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
                position_id_counter += 1
                
                position_status = PositionStatus.LONG
                entry_price = m2_price
                entry_direction = "LONG"
        
        elif action == "OPEN_SHORT":
            num_contracts = calculate_position_size(equity, m2_atr, carry_multiplier)
            
            if num_contracts > 0:
                stop_price = calculate_stop_loss(m2_price, m2_atr, "SHORT")
                
                entry_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action="OPEN_SHORT",
                    contract_code=current_m2_contract,
                    price=m2_price,
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
                position_id_counter += 1
                
                position_status = PositionStatus.SHORT
                entry_price = m2_price
                entry_direction = "SHORT"
        
        elif action == "CLOSE_LONG" or action == "CLOSE_SHORT":
            if current_position is not None:
                exit_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action=action,
                    contract_code=current_m2_contract,
                    price=m2_price,
                    num_contracts=current_position.entry_trade.num_contracts
                )
                trade_id_counter += 1
                
                current_position.exit_trade = exit_trade
                current_position.exit_reason = "SIGNAL_REVERSAL"
                current_position.calculate_pnl()
                results.positions.append(current_position)
                
                realized_pnl_today = current_position.net_pnl
                equity += current_position.net_pnl
                
                position_status = PositionStatus.FLAT
                current_position = None
                entry_price = None
                entry_direction = None
        
        elif action == "REVERSE_TO_LONG":
            # Close short, open long
            if current_position is not None:
                # Close short
                exit_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action="CLOSE_SHORT",
                    contract_code=current_m2_contract,
                    price=m2_price,
                    num_contracts=current_position.entry_trade.num_contracts
                )
                trade_id_counter += 1
                
                current_position.exit_trade = exit_trade
                current_position.exit_reason = "SIGNAL_REVERSAL"
                current_position.calculate_pnl()
                results.positions.append(current_position)
                
                realized_pnl_today = current_position.net_pnl
                equity += current_position.net_pnl
                
                # Open long
                num_contracts = calculate_position_size(equity, m2_atr, carry_multiplier)
                if num_contracts > 0:
                    stop_price = calculate_stop_loss(m2_price, m2_atr, "LONG")
                    
                    entry_trade = Trade(
                        trade_id=trade_id_counter,
                        date=current_date,
                        action="OPEN_LONG",
                        contract_code=current_m2_contract,
                        price=m2_price,
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
                    position_id_counter += 1
                    
                    position_status = PositionStatus.LONG
                    entry_price = m2_price
                    entry_direction = "LONG"
                else:
                    position_status = PositionStatus.FLAT
                    current_position = None
                    entry_price = None
                    entry_direction = None
        
        elif action == "REVERSE_TO_SHORT":
            # Close long, open short
            if current_position is not None:
                # Close long
                exit_trade = Trade(
                    trade_id=trade_id_counter,
                    date=current_date,
                    action="CLOSE_LONG",
                    contract_code=current_m2_contract,
                    price=m2_price,
                    num_contracts=current_position.entry_trade.num_contracts
                )
                trade_id_counter += 1
                
                current_position.exit_trade = exit_trade
                current_position.exit_reason = "SIGNAL_REVERSAL"
                current_position.calculate_pnl()
                results.positions.append(current_position)
                
                realized_pnl_today = current_position.net_pnl
                equity += current_position.net_pnl
                
                # Open short
                num_contracts = calculate_position_size(equity, m2_atr, carry_multiplier)
                if num_contracts > 0:
                    stop_price = calculate_stop_loss(m2_price, m2_atr, "SHORT")
                    
                    entry_trade = Trade(
                        trade_id=trade_id_counter,
                        date=current_date,
                        action="OPEN_SHORT",
                        contract_code=current_m2_contract,
                        price=m2_price,
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
                    position_id_counter += 1
                    
                    position_status = PositionStatus.SHORT
                    entry_price = m2_price
                    entry_direction = "SHORT"
                else:
                    position_status = PositionStatus.FLAT
                    current_position = None
                    entry_price = None
                    entry_direction = None
        
        # ====================================================================
        # 7. CALCULATE UNREALIZED P&L
        # ====================================================================
        unrealized_pnl = 0.0
        if current_position is not None and entry_price is not None:
            price_change = m2_price - entry_price if entry_direction == "LONG" else entry_price - m2_price
            unrealized_pnl = price_change * CONTRACT_SIZE * current_position.entry_trade.num_contracts
        
        # ====================================================================
        # 8. LOG DAILY METRICS
        # ====================================================================
        daily_metric = DailyMetrics(
            date=current_date,
            position_status=position_status,
            num_contracts=current_position.entry_trade.num_contracts if current_position else 0,
            current_price=m2_price,
            signal=current_signal,
            forecasted_return=forecasted_return,
            atr=m2_atr,
            stop_loss=current_position.entry_trade.stop_loss if current_position else None,
            carry_spread=carry_spread,
            carry_multiplier=carry_multiplier,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl_today,
            cumulative_pnl=equity - initial_equity,
            equity=equity
        )
        
        results.daily_metrics.append(daily_metric)
    
    # ====================================================================
    # 9. CLOSE ANY OPEN POSITIONS AT END
    # ====================================================================
    if current_position is not None:
        last_date = signal_data.iloc[-1]['Date']
        last_price = m2_price
        
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
    
    # Step 5: Add ATR to all contracts
    print("\n[Calculating ATR for all contracts]")
    for contract_code, df in contract_data.items():
        contract_data[contract_code] = add_atr_to_contract(df, period=ATR_PERIOD)
    print("   ✓ ATR calculated for all contracts")
    
    # Step 6: Run backtest
    results = run_backtest(backtest_data, contract_data, fnd_calendar, INITIAL_EQUITY)
    
    # Step 7: Print summary
    results.print_summary()
    
    # Step 8: Export results
    print("\n[Exporting Results]")
    result_dfs = results.to_dataframes()
    
    if len(result_dfs['positions']) > 0:
        result_dfs['positions'].to_csv('output/positions.csv', index=False)
        print("   ✓ Positions exported to output/positions.csv")
    
    if len(result_dfs['rolls']) > 0:
        result_dfs['rolls'].to_csv('output/rolls.csv', index=False)
        print("   ✓ Rolls exported to output/rolls.csv")
    
    if len(result_dfs['daily']) > 0:
        result_dfs['daily'].to_csv('output/daily_metrics.csv', index=False)
        print("   ✓ Daily metrics exported to output/daily_metrics.csv")
    
    result_dfs['summary'].to_csv('output/summary.csv')
    print("   ✓ Summary exported to output/summary.csv")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. ✓ Load and preprocess market data")
    print("2. ✓ Implement signal generation (3-factor regression)")
    print("3. ✓ Implement position sizing with carry filter")
    print("4. ✓ Implement risk management (ATR stops)")
    print("5. ✓ Run backtest simulation")
    print("6. ✓ Analyze results")
    print("\n")
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✓ Output saved to: {log_filename}")
