"""
Data models and logging structures for FINS3666 HG Copper Futures Backtest

Contains all dataclasses for trades, positions, rolls, and results tracking.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from datetime import datetime

from config import (
    SignalType,
    PositionStatus,
    CONTRACT_SIZE,
    TOTAL_TRANSACTION_COST
)


# ============================================================================
# TRADE EXECUTION MODELS
# ============================================================================

@dataclass
class Trade:
    """
    Represents a single trade execution (entry or exit)
    """
    trade_id: int
    date: datetime
    action: Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"]
    contract_code: str  # e.g., "HGK25" (May 2025)
    price: float  # Entry/exit price in $/lb
    num_contracts: int
    stop_loss: Optional[float] = None  # Stop price in $/lb
    atr: Optional[float] = None  # ATR at trade time
    carry_spread: Optional[float] = None  # M3 - M2 spread
    carry_multiplier: float = 1.0  # Position size adjustment (1.0, 0.5, 0.25)
    
    def notional_value(self) -> float:
        """Calculate notional value of trade"""
        return self.price * CONTRACT_SIZE * self.num_contracts
    
    def transaction_cost(self) -> float:
        """Calculate transaction cost for this trade leg"""
        return TOTAL_TRANSACTION_COST * self.num_contracts


@dataclass
class Position:
    """
    Represents an open position (entry to exit)
    """
    position_id: int
    entry_trade: Trade
    exit_trade: Optional[Trade] = None
    
    # P&L tracking
    directional_pnl: float = 0.0  # P&L from price movement
    transaction_costs: float = 0.0  # Total transaction costs
    net_pnl: float = 0.0  # Net P&L after costs
    
    # Exit reason tracking
    exit_reason: Optional[str] = None  # "SIGNAL_REVERSAL", "STOP_LOSS", "ROLL", "END_OF_BACKTEST"
    
    # Position metrics
    holding_days: int = 0
    
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_trade is None
    
    def calculate_pnl(self) -> None:
        """
        Calculate P&L for closed position
        Only call this after exit_trade is set
        """
        if not self.exit_trade:
            raise ValueError("Cannot calculate P&L for open position")
        
        entry_price = self.entry_trade.price
        exit_price = self.exit_trade.price
        contracts = self.entry_trade.num_contracts
        
        # Directional P&L calculation
        if self.entry_trade.action in ["OPEN_LONG"]:
            price_change = exit_price - entry_price
        else:  # SHORT
            price_change = entry_price - exit_price
        
        # Calculate dollar P&L
        self.directional_pnl = price_change * CONTRACT_SIZE * contracts
        
        # Transaction costs (entry + exit)
        self.transaction_costs = (
            self.entry_trade.transaction_cost() + 
            self.exit_trade.transaction_cost()
        )
        
        # Net P&L
        self.net_pnl = self.directional_pnl - self.transaction_costs
        
        # Holding period
        self.holding_days = (self.exit_trade.date - self.entry_trade.date).days
    
    def to_dict(self) -> dict:
        """Convert position to dictionary for logging"""
        return {
            'position_id': self.position_id,
            'entry_date': self.entry_trade.date,
            'exit_date': self.exit_trade.date if self.exit_trade else None,
            'contract': self.entry_trade.contract_code,
            'entry_price': self.entry_trade.price,
            'exit_price': self.exit_trade.price if self.exit_trade else None,
            'num_contracts': self.entry_trade.num_contracts,
            'direction': 'LONG' if 'LONG' in self.entry_trade.action else 'SHORT',
            'directional_pnl': self.directional_pnl,
            'transaction_costs': self.transaction_costs,
            'net_pnl': self.net_pnl,
            'exit_reason': self.exit_reason,
            'holding_days': self.holding_days,
            'stop_loss': self.entry_trade.stop_loss,
            'atr': self.entry_trade.atr,
            'carry_spread': self.entry_trade.carry_spread,
            'carry_multiplier': self.entry_trade.carry_multiplier
        }


@dataclass
class RollEvent:
    """
    Represents a contract roll (forced exit and re-entry)
    """
    roll_id: int
    roll_date: datetime
    from_contract: str  # e.g., "HGK25"
    to_contract: str  # e.g., "HGN25"
    exit_price: float  # Price we exited old contract
    entry_price: float  # Price we entered new contract
    num_contracts: int
    
    # Roll P&L calculation
    roll_pnl: float = 0.0  # P&L from the roll itself (negative = cost, positive = gain)
    roll_spread: float = 0.0  # Spread at roll time (to_price - from_price)
    
    # Context
    fnd_date: datetime = None  # First Notice Day that triggered roll
    days_before_fnd: int = 5  # Should always be 5 per strategy
    
    def calculate_roll_pnl(self) -> None:
        """
        Calculate P&L from rolling contracts
        This is separate from directional P&L
        """
        # Roll spread (positive = contango cost, negative = backwardation gain)
        self.roll_spread = self.entry_price - self.exit_price
        
        # Roll P&L (negative spread = we gain, positive spread = we lose)
        # Because we sell old contract and buy new contract
        self.roll_pnl = -self.roll_spread * CONTRACT_SIZE * self.num_contracts
        
    def to_dict(self) -> dict:
        """Convert roll event to dictionary"""
        return {
            'roll_id': self.roll_id,
            'roll_date': self.roll_date,
            'from_contract': self.from_contract,
            'to_contract': self.to_contract,
            'exit_price': self.exit_price,
            'entry_price': self.entry_price,
            'num_contracts': self.num_contracts,
            'roll_spread': self.roll_spread,
            'roll_pnl': self.roll_pnl,
            'fnd_date': self.fnd_date,
            'days_before_fnd': self.days_before_fnd
        }


# ============================================================================
# DAILY METRICS
# ============================================================================

@dataclass
class DailyMetrics:
    """
    Daily performance metrics for the backtest
    """
    date: datetime
    
    # Portfolio state
    position_status: PositionStatus
    num_contracts: int = 0
    current_price: float = 0.0
    
    # Signal generation
    signal: SignalType = SignalType.NEUTRAL
    forecasted_return: float = 0.0
    
    # Risk metrics
    atr: float = 0.0
    stop_loss: Optional[float] = None
    carry_spread: float = 0.0
    carry_multiplier: float = 1.0
    
    # P&L tracking
    unrealized_pnl: float = 0.0  # Mark-to-market for open positions
    realized_pnl: float = 0.0  # P&L from closed trades today
    roll_pnl: float = 0.0  # P&L from rolls today
    daily_total_pnl: float = 0.0  # Total P&L for the day
    cumulative_pnl: float = 0.0  # Running total P&L
    
    # Account metrics
    equity: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert daily metrics to dictionary"""
        return {
            'date': self.date,
            'position_status': self.position_status.value,
            'num_contracts': self.num_contracts,
            'current_price': self.current_price,
            'signal': self.signal if isinstance(self.signal, str) else self.signal.value,
            'forecasted_return': self.forecasted_return,
            'atr': self.atr,
            'stop_loss': self.stop_loss,
            'carry_spread': self.carry_spread,
            'carry_multiplier': self.carry_multiplier,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'roll_pnl': self.roll_pnl,
            'daily_total_pnl': self.daily_total_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'equity': self.equity
        }


# ============================================================================
# BACKTEST RESULTS
# ============================================================================

@dataclass
class BacktestResults:
    """
    Container for all backtest results and performance metrics
    """
    # Trade logs
    positions: List[Position] = field(default_factory=list)
    roll_events: List[RollEvent] = field(default_factory=list)
    daily_metrics: List[DailyMetrics] = field(default_factory=list)
    
    # Summary statistics
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    
    # P&L breakdown
    total_directional_pnl: float = 0.0
    total_roll_pnl: float = 0.0
    total_transaction_costs: float = 0.0
    net_pnl: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def calculate_summary_statistics(self):
        """Calculate all summary statistics from position and daily logs"""
        if not self.positions:
            return
        
        # Filter closed positions
        closed_positions = [p for p in self.positions if not p.is_open()]
        
        if not closed_positions:
            return
        
        # P&L breakdown
        self.total_directional_pnl = sum(p.directional_pnl for p in closed_positions)
        self.total_roll_pnl = sum(r.roll_pnl for r in self.roll_events)
        self.total_transaction_costs = sum(p.transaction_costs for p in closed_positions)
        self.net_pnl = sum(p.net_pnl for p in closed_positions) + self.total_roll_pnl
        
        # Final equity
        self.final_equity = self.initial_equity + self.net_pnl
        self.total_return = (self.final_equity / self.initial_equity - 1) * 100
        
        # Trade statistics
        self.total_trades = len(closed_positions)
        self.winning_trades = sum(1 for p in closed_positions if p.net_pnl > 0)
        self.losing_trades = sum(1 for p in closed_positions if p.net_pnl < 0)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Average win/loss
        wins = [p.net_pnl for p in closed_positions if p.net_pnl > 0]
        losses = [p.net_pnl for p in closed_positions if p.net_pnl < 0]
        
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown calculation
        if self.daily_metrics:
            equity_curve = [m.equity for m in self.daily_metrics]
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            self.max_drawdown = abs(np.min(drawdown)) * 100
        
        # Sharpe ratio (assuming daily returns)
        if self.daily_metrics and len(self.daily_metrics) > 1:
            daily_returns = [m.daily_total_pnl / m.equity if m.equity > 0 else 0 
                           for m in self.daily_metrics]
            if np.std(daily_returns) > 0:
                self.sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    
    def print_summary(self):
        """Print backtest summary to console"""
        print("\n" + "="*80)
        print("BACKTEST SUMMARY - HG Copper Futures Strategy")
        print("="*80)
        
        print(f"\n{'PERFORMANCE':-^80}")
        print(f"Initial Equity:              ${self.initial_equity:,.2f}")
        print(f"Final Equity:                ${self.final_equity:,.2f}")
        print(f"Total Return:                {self.total_return:.2f}%")
        print(f"Net P&L:                     ${self.net_pnl:,.2f}")
        
        print(f"\n{'P&L BREAKDOWN':-^80}")
        print(f"Directional P&L:             ${self.total_directional_pnl:,.2f}")
        print(f"Roll P&L:                    ${self.total_roll_pnl:,.2f}")
        print(f"Transaction Costs:           -${self.total_transaction_costs:,.2f}")
        print(f"  (Commission + Slippage @ ${TOTAL_TRANSACTION_COST}/contract)")
        
        print(f"\n{'TRADE STATISTICS':-^80}")
        print(f"Total Trades:                {self.total_trades}")
        print(f"Winning Trades:              {self.winning_trades}")
        print(f"Losing Trades:               {self.losing_trades}")
        print(f"Win Rate:                    {self.win_rate*100:.2f}%")
        print(f"Average Win:                 ${self.avg_win:,.2f}")
        print(f"Average Loss:                ${self.avg_loss:,.2f}")
        print(f"Profit Factor:               {self.profit_factor:.2f}")
        
        print(f"\n{'RISK METRICS':-^80}")
        print(f"Maximum Drawdown:            {self.max_drawdown:.2f}%")
        print(f"Sharpe Ratio (annualized):   {self.sharpe_ratio:.2f}")
        print(f"Number of Rolls:             {len(self.roll_events)}")
        
        print("="*80 + "\n")
    
    def to_dataframes(self) -> dict:
        """
        Export all results to pandas DataFrames for further analysis
        
        Returns:
            dict with keys: 'positions', 'rolls', 'daily', 'summary'
        """
        positions_df = pd.DataFrame([p.to_dict() for p in self.positions if not p.is_open()])
        rolls_df = pd.DataFrame([r.to_dict() for r in self.roll_events])
        daily_df = pd.DataFrame([m.to_dict() for m in self.daily_metrics])
        
        summary_dict = {
            'Initial Equity': self.initial_equity,
            'Final Equity': self.final_equity,
            'Total Return (%)': self.total_return,
            'Net P&L': self.net_pnl,
            'Directional P&L': self.total_directional_pnl,
            'Roll P&L': self.total_roll_pnl,
            'Transaction Costs': self.total_transaction_costs,
            'Total Trades': self.total_trades,
            'Win Rate (%)': self.win_rate * 100,
            'Profit Factor': self.profit_factor,
            'Max Drawdown (%)': self.max_drawdown,
            'Sharpe Ratio': self.sharpe_ratio
        }
        
        return {
            'positions': positions_df,
            'rolls': rolls_df,
            'daily': daily_df,
            'summary': pd.Series(summary_dict)
        }
