"""
Configuration file for FINS3666 HG Copper Futures Backtest

Contains all constants, enums, and configuration parameters.

AI Use Declaration:
- Claude and ChatGPT models were used to assist in code generation and debugging.
"""

from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(Enum):
    """Trading signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class PositionStatus(Enum):
    """Current position status"""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


# ============================================================================
# CONTRACT SPECIFICATIONS
# ============================================================================

CONTRACT_SIZE = 25000  # lbs per contract
TICK_SIZE = 0.0005  # $/lb
TICK_VALUE = 12.50  # $ per tick
POINT_VALUE = 250.0  # $ per $0.01/lb move


# ============================================================================
# TRANSACTION COSTS
# ============================================================================

# Per contract, per round-turn
COMMISSION = 7.50  # $ per contract
SLIPPAGE = 25.00  # $ per contract
TOTAL_TRANSACTION_COST = COMMISSION + SLIPPAGE  # $32.50 per contract


# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Signal thresholds
SIGNAL_THRESHOLD = 0.16  # ±0.1% forecasted return threshold

# Regression parameters
REGRESSION_WINDOW_DAYS = 504 # 2 years of trading days

# Risk management
ATR_PERIOD = 14  # Days for ATR calculation
ATR_STOP_MULTIPLIER = 1.5  # Stop distance = 1.5 × ATR
RISK_PERCENT = 0.018  # Risk 1.8% of equity per trade

# Trailing stop settings
TRAILING_STOP_ENABLED = True  # Enable trailing stop (as per trading plan)
MOVE_TO_BREAKEVEN_AFTER_ATR = 1.0  # Optional: Move stop to breakeven after 1× ATR favorable move

# Stop-limit order settings
STOP_LIMIT_ENABLED = True  # Use stop-limit orders instead of stop-market
STOP_LIMIT_BAND_TICKS = 4  # Minimum limit band in ticks (4 ticks = $0.002/lb)
STOP_LIMIT_BAND_MULTIPLIER = 1.5  # Limit band = max(4 ticks, 1.5 × avg overnight spread)

# Carry filter thresholds ($/lb)
CARRY_THRESHOLD_HALF = 0.01  # Halve position if spread exceeds this
CARRY_THRESHOLD_QUARTER = 0.02  # Quarter position if spread exceeds this

# Roll management
DAYS_BEFORE_FND_TO_ROLL = 5  # Exit 5 trading days before First Notice Day

# ============================================================================
# VOLATILITY REGIME FILTER (for Part B refinements)
# ============================================================================

# Rolling window to estimate realized volatility of HG daily returns
VOL_LOOKBACK_DAYS = 20          # 1 trading month

# Define what "high" and "extreme" volatility mean, relative to long-run vol
VOL_HIGH_MULTIPLIER = 2.0       # high-vol regime if > 2x normal vol
VOL_EXTREME_MULTIPLIER = 3.0    # extreme-vol regime if > 3x normal vol

# How much to reduce risk in high-vol regime
VOL_RISK_REDUCTION = 0.5        # cut risk per trade in half

# If True, don't open new positions when volatility is extreme
SUSPEND_TRADING_IN_EXTREME = True

# ============================================================================
# PORTFOLIO-LEVEL RISK OVERLAY
# ============================================================================

# If equity drawdown from peak exceeds this, stop opening new trades
MAX_DRAWDOWN_STOP = 0.05        # 5% portfolio drawdown kill-switch

# ============================================================================
# BACKTEST SETTINGS
# ============================================================================

# Define backtest period
BACKTEST_START = '2024-01-02'
BACKTEST_END = '2024-12-31'

INITIAL_EQUITY = 10_000_000  # $10 million starting capital

# Data file paths
DATA_DIR = "data/"
DXY_FILE = DATA_DIR + "DXY_Daily.csv"
PMI_FILE = DATA_DIR + "China_PMI_Monthly.csv"
HG_FRONT_FILE = DATA_DIR + "Front_Continuous_HG_Daily.csv"
LME_STOCKS_FILE = DATA_DIR + "LME_Copper_Tonnage_Daily.csv"
FND_CALENDAR_FILE = DATA_DIR + "HG_FND_2025.csv"  # To be created

# Output file paths
OUTPUT_DIR = "output/"
POSITIONS_OUTPUT = OUTPUT_DIR + "positions.csv"
ROLLS_OUTPUT = OUTPUT_DIR + "rolls.csv"
DAILY_OUTPUT = OUTPUT_DIR + "daily_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR + "summary.csv"
CHART_OUTPUT = OUTPUT_DIR + "performance_chart.png"
