"""
Analysis of July-August 2025 Large Loss Event
FINS3666 Assignment - Out-of-Sample Period Investigation

Hypothesis: Large losses due to external shock from Trump tariff announcement
This script analyzes the correlation between HG price movements and strategy P&L
during the critical July-August 2025 period.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ============================================================================
# DATA LOADING
# ============================================================================

def load_positions_data():
    """Load positions data from out-of-sample results"""
    positions = pd.read_csv('output/out_of_sample_2025/positions.csv')
    positions['entry_date'] = pd.to_datetime(positions['entry_date'])
    positions['exit_date'] = pd.to_datetime(positions['exit_date'])
    return positions

def load_daily_metrics():
    """Load daily metrics for cumulative P&L tracking"""
    daily = pd.read_csv('output/out_of_sample_2025/daily_metrics.csv')
    daily['date'] = pd.to_datetime(daily['date'])
    return daily

def load_hg_continuous():
    """Load HG front continuous contract for price reference"""
    hg = pd.read_csv('data/Front_Continuous_HG_Daily.csv')
    hg['Date'] = pd.to_datetime(hg['Date'], format='%m/%d/%y')
    hg = hg.sort_values('Date').reset_index(drop=True)
    hg = hg.rename(columns={'Settlement Price': 'HG_Price'})
    return hg[['Date', 'HG_Price']]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_position_level_pnl(positions):
    """
    Calculate daily P&L from positions data
    Returns DataFrame with date, position_id, direction, daily_pnl
    """
    pnl_records = []
    
    for _, pos in positions.iterrows():
        # Create daily range for this position
        date_range = pd.date_range(start=pos['entry_date'], 
                                   end=pos['exit_date'], 
                                   freq='D')
        
        # Distribute P&L across holding days (simplified approximation)
        holding_days = len(date_range)
        daily_pnl = pos['net_pnl'] / holding_days if holding_days > 0 else 0
        
        for date in date_range:
            pnl_records.append({
                'date': date,
                'position_id': pos['position_id'],
                'direction': pos['direction'],
                'daily_pnl_approx': daily_pnl,
                'net_pnl': pos['net_pnl'],
                'entry_price': pos['entry_price'],
                'exit_price': pos['exit_price'],
                'num_contracts': pos['num_contracts']
            })
    
    return pd.DataFrame(pnl_records)

def identify_large_loss_trades(positions, threshold=-20000):
    """Identify trades with losses exceeding threshold"""
    large_losses = positions[positions['net_pnl'] < threshold].copy()
    large_losses = large_losses.sort_values('net_pnl')
    return large_losses

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_july_august_analysis_chart(daily_metrics, hg_continuous, positions):
    """
    Create comprehensive chart showing:
    1. HG1 continuous price
    2. Cumulative P&L
    3. Position markers for entries/exits
    """
    
    # Filter to June-September 2025
    start_date = pd.Timestamp('2025-06-01')
    end_date = pd.Timestamp('2025-09-30')
    
    daily_july_aug = daily_metrics[(daily_metrics['date'] >= start_date) & 
                                    (daily_metrics['date'] <= end_date)].copy()
    
    hg_july_aug = hg_continuous[(hg_continuous['Date'] >= start_date) & 
                                 (hg_continuous['Date'] <= end_date)].copy()
    
    positions_july_aug = positions[
        ((positions['entry_date'] >= start_date) & (positions['entry_date'] <= end_date)) |
        ((positions['exit_date'] >= start_date) & (positions['exit_date'] <= end_date))
    ].copy()
    
    # Create figure with two y-axes sharing x-axis
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # ========================================================================
    # Primary Y-axis: HG1 Continuous Price
    # ========================================================================
    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('HG Copper Price ($/lb)', fontsize=12, fontweight='bold', color=color1)
    
    line1 = ax1.plot(hg_july_aug['Date'], hg_july_aug['HG_Price'], 
                     linewidth=2.5, color=color1, label='HG Front Continuous', 
                     alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================================================
    # Secondary Y-axis: Cumulative P&L
    # ========================================================================
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Cumulative P&L ($)', fontsize=12, fontweight='bold', color=color2)
    
    line2 = ax2.plot(daily_july_aug['date'], daily_july_aug['cumulative_pnl'], 
                     linewidth=2.5, color=color2, label='Cumulative P&L', 
                     alpha=0.8, linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Format P&L axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ========================================================================
    # Mark Position #36 Entry and Exit Only
    # ========================================================================
    # Find position #36 (the largest loss trade)
    pos_36 = positions_july_aug[positions_july_aug['position_id'] == 36]
    
    if len(pos_36) > 0:
        pos_36 = pos_36.iloc[0]
        
        # Entry marker and label
        if pos_36['entry_date'] >= start_date and pos_36['entry_date'] <= end_date:
            entry_hg_price = hg_continuous[hg_continuous['Date'] == pos_36['entry_date']]['HG_Price'].values
            if len(entry_hg_price) > 0:
                ax1.scatter(pos_36['entry_date'], entry_hg_price[0], 
                           color='red', marker='v', s=200, 
                           alpha=0.8, edgecolors='black', linewidths=2,
                           zorder=5)
                
                # Entry price label
                ax1.annotate(
                    f'Entry: ${pos_36["entry_price"]:.2f}/lb',
                    xy=(pos_36['entry_date'], entry_hg_price[0]),
                    xytext=(10, -30),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
                )
        
        # Exit marker and label
        if pos_36['exit_date'] >= start_date and pos_36['exit_date'] <= end_date:
            exit_hg_price = hg_continuous[hg_continuous['Date'] == pos_36['exit_date']]['HG_Price'].values
            if len(exit_hg_price) > 0:
                ax1.scatter(pos_36['exit_date'], exit_hg_price[0], 
                           color='darkred', marker='x', s=200, 
                           alpha=0.8, linewidths=3, zorder=5)
                
                # Exit price label
                ax1.annotate(
                    f'Exit: ${pos_36["exit_price"]:.2f}/lb',
                    xy=(pos_36['exit_date'], exit_hg_price[0]),
                    xytext=(10, 30),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='darkred', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5)
                )
    
    # ========================================================================
    # Mark Trump Tariff Announcement (July 8, 2025 evening)
    # ========================================================================
    trump_announcement = pd.Timestamp('2025-07-08')
    
    # Add vertical line for announcement date
    ax1.axvline(x=trump_announcement, color='orange', linestyle='--', 
               linewidth=2.5, alpha=0.7, zorder=4)
    
    # Add annotation for the announcement
    # Find the HG price on July 8 for positioning
    july_8_price = hg_continuous[hg_continuous['Date'] == trump_announcement]['HG_Price'].values
    if len(july_8_price) > 0:
        y_pos = july_8_price[0]
    else:
        # Use midpoint of y-axis if price not available
        y_pos = (ax1.get_ylim()[0] + ax1.get_ylim()[1]) / 2
    
    ax1.annotate(
        'Trump Tariff Announcement\nJuly 8, 2025 (Evening)\n"50% copper tariff"',
        xy=(trump_announcement, y_pos),
        xytext=(15, 60),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='darkorange',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                 edgecolor='orange', alpha=0.9, linewidth=2),
        arrowprops=dict(arrowstyle='->', color='orange', lw=2)
    )
    
    # ========================================================================
    # Title and Legend
    # ========================================================================
    plt.title('Out-of-Sample 2025 Loss Analysis: HG Price vs Strategy P&L\n' + 
              'Hypothesis: External Shock from Trade Policy Announcement',
              fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add custom legend entries for Position #36 markers and Trump announcement
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=10, label='Position #36 Entry (SHORT)', markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='darkred', 
               markersize=10, label='Position #36 Exit', markeredgewidth=2),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2.5,
               label='Trump Tariff Announcement')
    ]
    
    ax1.legend(lines1 + lines2 + custom_lines, 
              labels1 + labels2 + [l.get_label() for l in custom_lines],
              loc='upper left', fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # Format x-axis
    # ========================================================================
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # ========================================================================
    # Statistics Box
    # ========================================================================
    total_pnl_period = daily_july_aug['cumulative_pnl'].iloc[-1] - daily_july_aug['cumulative_pnl'].iloc[0]
    num_trades = len(positions_july_aug)
    winning_trades = (positions_july_aug['net_pnl'] > 0).sum()
    losing_trades = (positions_july_aug['net_pnl'] < 0).sum()
    
    hg_price_change = ((hg_july_aug['HG_Price'].iloc[-1] / hg_july_aug['HG_Price'].iloc[0]) - 1) * 100
    
    stats_text = (
        f"Jun-Sep 2025 Statistics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Period P&L: ${total_pnl_period:,.0f}\n"
        f"Total Trades: {num_trades}\n"
        f"Winning: {winning_trades} | Losing: {losing_trades}\n"
        f"HG Price Change: {hg_price_change:+.2f}%\n"
        f"Start Price: ${hg_july_aug['HG_Price'].iloc[0]:.2f}/lb\n"
        f"End Price: ${hg_july_aug['HG_Price'].iloc[-1]:.2f}/lb"
    )
    
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    # Save chart
    plt.show()
    output_path = 'output/out_of_sample_2025/june_september_loss_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")
    
    plt.close()
    
    return positions_july_aug, total_pnl_period

# ============================================================================
# DETAILED TRADE ANALYSIS
# ============================================================================

def print_trade_analysis(positions_july_aug):
    """Print detailed analysis of trades during June-September"""
    
    print("\n" + "="*80)
    print("JUNE-SEPTEMBER 2025 TRADE ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Trades in Period: {len(positions_july_aug)}")
    print(f"Winning Trades: {(positions_july_aug['net_pnl'] > 0).sum()}")
    print(f"Losing Trades: {(positions_july_aug['net_pnl'] < 0).sum()}")
    print(f"Total P&L: ${positions_july_aug['net_pnl'].sum():,.2f}")
    
    print("\n" + "-"*80)
    print("TOP 5 WORST TRADES:")
    print("-"*80)
    
    worst_trades = positions_july_aug.nsmallest(5, 'net_pnl')
    
    for idx, (_, trade) in enumerate(worst_trades.iterrows(), 1):
        print(f"\n#{idx} - Position ID: {trade['position_id']}")
        print(f"   Direction: {trade['direction']} | Contracts: {trade['num_contracts']}")
        print(f"   Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}/lb")
        print(f"   Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}/lb")
        print(f"   Exit Reason: {trade['exit_reason']}")
        print(f"   Net P&L: ${trade['net_pnl']:,.2f}")
        print(f"   Holding Days: {trade['holding_days']}")
        
        price_move = ((trade['exit_price'] / trade['entry_price']) - 1) * 100
        print(f"   Price Move: {price_move:+.2f}%")
    
    print("\n" + "-"*80)
    print("TOP 3 BEST TRADES:")
    print("-"*80)
    
    best_trades = positions_july_aug.nlargest(3, 'net_pnl')
    
    for idx, (_, trade) in enumerate(best_trades.iterrows(), 1):
        print(f"\n#{idx} - Position ID: {trade['position_id']}")
        print(f"   Direction: {trade['direction']} | Contracts: {trade['num_contracts']}")
        print(f"   Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}/lb")
        print(f"   Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}/lb")
        print(f"   Net P&L: ${trade['net_pnl']:,.2f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis execution"""
    
    print("\n" + "="*80)
    print("INVESTIGATING JUNE-SEPTEMBER 2025 LARGE LOSS EVENT")
    print("Hypothesis: External shock from Trump tariff announcement")
    print("="*80)
    
    # Load data
    print("\n[Loading Data...]")
    positions = load_positions_data()
    daily_metrics = load_daily_metrics()
    hg_continuous = load_hg_continuous()
    print("   ✓ Data loaded successfully")
    
    # Filter to June-September period
    start_date = pd.Timestamp('2025-06-01')
    end_date = pd.Timestamp('2025-10-31')
    
    positions_july_aug = positions[
        ((positions['entry_date'] >= start_date) & (positions['entry_date'] <= end_date)) |
        ((positions['exit_date'] >= start_date) & (positions['exit_date'] <= end_date))
    ].copy()
    
    # Create visualization
    print("\n[Generating Analysis Chart...]")
    positions_july_aug_filtered, total_pnl = create_july_august_analysis_chart(
        daily_metrics, hg_continuous, positions_july_aug
    )
    
    # Print detailed trade analysis
    print_trade_analysis(positions_july_aug)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nJune-September Period P&L: ${total_pnl:,.2f}")
    print(f"\nThe largest single loss was ${positions_july_aug['net_pnl'].min():,.2f}")
    print(f"This occurred on trade #{positions_july_aug.loc[positions_july_aug['net_pnl'].idxmin(), 'position_id']}")
    
    # Calculate HG price volatility during period
    daily_july_aug = daily_metrics[(daily_metrics['date'] >= start_date) & 
                                    (daily_metrics['date'] <= end_date)]
    
    if len(daily_july_aug) > 0:
        hg_july_aug_prices = hg_continuous[(hg_continuous['Date'] >= start_date) & 
                                            (hg_continuous['Date'] <= end_date)]
        
        if len(hg_july_aug_prices) > 1:
            price_returns = hg_july_aug_prices['HG_Price'].pct_change().dropna()
            volatility = price_returns.std() * np.sqrt(252) * 100  # Annualized
            
            print(f"\nPrice Volatility (Annualized): {volatility:.2f}%")
            print(f"Max Daily Price Move: {price_returns.abs().max()*100:+.2f}%")
    
    print("\n" + "="*80)
    print("Analysis complete. Chart saved to output/out_of_sample_2025/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
