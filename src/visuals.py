# assignment_visuals.py
#
# Visualisations for FINS3666 Part B report:
#   1. Autopsy chart (July 2025 gap failure – initial model)
#   2. Performance Decay table (Initial vs Refined 2025)
#   3. Adaptive Risk Engine flowchart
#   4. Smoking Gun chart (structure scanner evidence – refined model)
#

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------
# Utility: read summary.csv into a convenient Series
# ---------------------------------------------------------------------
def load_summary(summary_path: Path) -> pd.Series:
    """
    summary.csv is written as two columns:
        'Unnamed: 0' = metric name
        '0'          = value
    This returns a Series indexed by metric name.
    """
    df = pd.read_csv(summary_path)
    df = df.rename(columns={"Unnamed: 0": "metric", "0": "value"})
    return df.set_index("metric")["value"]


# ---------------------------------------------------------------------
# 1. Autopsy Chart: July 5–15 2025 fail (INITIAL MODEL)
# ---------------------------------------------------------------------

import matplotlib.dates as mdates  # make sure this is at the top of the file

def plot_autopsy_july_2025_initial(
    initial_oos_folder: Path,
    save_path: Path,
    band_width: float = 0.20,
):
    """
    Autopsy chart for the July 2025 loss in the ORIGINAL model.

    Implements assignment requirements:
    - HG M2 price for 5–15 July 2025
    - Mark: entry 7.72, stop 7.90, stop-limit band, gap 8–9 July, exit 8.47
    """

    dm = pd.read_csv(initial_oos_folder / "daily_metrics.csv", parse_dates=["date"])
    pos = pd.read_csv(
        initial_oos_folder / "positions.csv",
        parse_dates=["entry_date", "exit_date"],
    )

    # --- Trade details (force prices to assignment values) ---
    trade = pos.loc[
        (pos["entry_date"] == "2025-07-07") & (pos["direction"] == "SHORT")
    ].iloc[0]

    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]
    entry_price = 7.72      # assignment spec
    stop_price = 7.90       # assignment spec
    exit_price = 8.47       # assignment spec

    # --- Price window: 5–15 July 2025 ---
    mask = (dm["date"] >= "2025-07-05") & (dm["date"] <= "2025-07-15")
    window = dm.loc[mask].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Price line (HG M2 / front)
    ax.plot(
        window["date"],
        window["current_price"],
        marker="o",
        linewidth=2,
        label="HG M2 / Front price",
    )

    # Short entry
    ax.scatter(
        entry_date,
        entry_price,
        marker="v",
        s=80,
        color="tab:red",
        label="Short entry 7.72",
        zorder=5,
    )

    # Gap open on 8 July
    gap_date = pd.Timestamp("2025-07-08")
    if (window["date"] == gap_date).any():
        gap_price = window.loc[window["date"] == gap_date, "current_price"].iloc[0]
        ax.scatter(
            gap_date,
            gap_price,
            marker="^",
            s=90,
            color="tab:orange",
            label="Gap open (8 Jul)",
            zorder=5,
        )

    # Optional: point on 9 July to emphasise “gap July 8–9”
    gap2_date = pd.Timestamp("2025-07-09")
    if (window["date"] == gap2_date).any():
        gap2_price = window.loc[window["date"] == gap2_date, "current_price"].iloc[0]
        ax.scatter(
            gap2_date,
            gap2_price,
            marker="^",
            s=70,
            color="tab:orange",
            zorder=5,
        )

    # Final exit
    ax.scatter(
        exit_date,
        exit_price,
        marker="x",
        s=90,
        color="black",
        label="Final exit 8.47",
        zorder=5,
    )

    # Stop line at 7.90
    ax.axhline(
        stop_price,
        linestyle="--",
        linewidth=1.5,
        color="tab:gray",
        label="Stop loss 7.90",
    )

    # Stop-limit band: stop -> stop + band_width
    ax.fill_between(
        window["date"],
        stop_price,
        stop_price + band_width,
        color="gold",
        alpha=0.2,
        label="Stop-limit band",
    )

    # Small note that refined model would be flat here
    ax.text(
        window["date"].min(),
        window["current_price"].max() + 0.02,
        "Refined model: flat / no position\n(see Section 2)",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.set_title("Autopsy: July 2025 Short Failure (Initial Model)")
    ax.set_ylabel("HG Copper Price ($/lb)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



# ---------------------------------------------------------------------
# 2. Performance Decay TABLE (Initial vs Refined 2025)
# ---------------------------------------------------------------------
def build_performance_decay_table(
    initial_summary_path: Path,
    refined_summary_path: Path,
) -> pd.DataFrame:
    """
    Creates a comparative table: Initial 2025 vs Refined 2025.

    Columns: Net P&L, Sharpe, Max Drawdown (bold in your report),
             Calmar, and % improvement.

    Returns a DataFrame you can export to CSV / LaTeX and also use
    for a bar chart if you want.
    """
    s_init = load_summary(initial_summary_path)
    s_ref = load_summary(refined_summary_path)

    def metric_row(metric_name, initial_key, refined_key=None):
        key_ref = refined_key or initial_key
        initial_val = float(s_init[initial_key])
        refined_val = float(s_ref[key_ref])
        pct_change = (refined_val - initial_val) / abs(initial_val) * 100.0
        return metric_name, initial_val, refined_val, pct_change

    rows = []

    # Net P&L
    rows.append(metric_row("Net P&L ($)", "Net P&L"))

    # Sharpe
    rows.append(metric_row("Sharpe Ratio", "Sharpe Ratio (Annualized)"))

    # Max Drawdown
    rows.append(metric_row("Max Drawdown (%)", "Max Drawdown (%)"))

    # Calmar = Annualised return / Max DD
    ann_init = float(s_init["Annualized Return (%)"])
    ann_ref = float(s_ref["Annualized Return (%)"])
    dd_init = float(s_init["Max Drawdown (%)"])
    dd_ref = float(s_ref["Max Drawdown (%)"])
    calmar_init = ann_init / dd_init if dd_init != 0 else float("nan")
    calmar_ref = ann_ref / dd_ref if dd_ref != 0 else float("nan")
    pct_change_calmar = (calmar_ref - calmar_init) / abs(calmar_init) * 100.0
    rows.append(
        (
            "Calmar Ratio",
            calmar_init,
            calmar_ref,
            pct_change_calmar,
        )
    )

    df = pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Initial_2025",
            "Refined_2025",
            "% Change vs Initial",
        ],
    )
    return df


def save_performance_decay_table(
    initial_summary_path: Path,
    refined_summary_path: Path,
    save_csv_path: Path,
):
    """Convenience wrapper that also writes the CSV."""
    df = build_performance_decay_table(initial_summary_path, refined_summary_path)
    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    return df


# ---------------------------------------------------------------------
# 3. Adaptive Risk Engine Flowchart (Diagram)
# ---------------------------------------------------------------------
def plot_adaptive_risk_flowchart(save_path: Path):
    """
    Draws the logic flow of the Adaptive Risk Engine as a simple
    boxes-and-arrows diagram using matplotlib (no extra libraries).

    The structure:

       3-factor signal
              |
       Volatility Gate (σ > 2x?)
          /           \
        yes           no
        flat       Risk scanners
                     /     \
               triggered?   safe
                 |          |
            force exit   size position (1.8% * vol scalar)

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    def box(x, y, text, width=2.6, height=0.8):
        rect = plt.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, wrap=True)
        return rect

    def arrow(x1, y1, x2, y2, text=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
        if text:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.1,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Layout coordinates
    y_start = 5.0
    y_vol_gate = 4.0
    y_flat = 2.8
    y_scanners = 3.0
    y_exit = 1.8
    y_size = 1.8

    # Nodes
    b_signal = box(0, y_start, "Inputs:\n3-factor signal\n+ macro data")
    b_vol = box(0, y_vol_gate, "Gate 1:\nVolatility regime\nσ > 2× baseline?")
    b_flat = box(-3.0, y_flat, "High / extreme vol:\nSuspend new trades\n(position = FLAT)")
    b_scan = box(3.0, y_scanners, "Gate 2:\nATR & structure scanners\n(ATR ratio > 2.0 or\nspread Z-score > 3)")
    b_exit = box(3.0, y_exit, "Any scanner triggered:\nForce exit / prevent entry")
    b_size = box(0, y_size, "Safe regime:\nSize position\n1.8% of equity ×\nvolatility scalar")

    # Arrows
    arrow(0, y_start - 0.5, 0, y_vol_gate + 0.4)
    arrow(0.7, y_vol_gate - 0.4, 2.4, y_scanners + 0.4, text="no")
    arrow(-0.7, y_vol_gate - 0.4, -2.4, y_flat + 0.4, text="yes")

    arrow(3.0, y_scanners - 0.4, 3.0, y_exit + 0.4, text="triggered")
    arrow(2.0, y_scanners - 0.4, 0.4, y_size + 0.4, text="safe")

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(1.0, 5.5)
    ax.set_title("Adaptive Risk Engine – Logic Flow", fontsize=12)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. Smoking Gun Chart: structure scanner vs refined P&L
# ---------------------------------------------------------------------

def plot_smoking_gun_structure_scanner(
    refined_oos_folder: Path,
    save_path: Path,
    initial_oos_folder: Path | None = None,
):
    """
    Dual-axis chart:
    - Left: M1–M2 spread (carry_spread)
    - Right: refined cumulative P&L (dashed) and optional initial cumulative P&L (dotted)
    - Highlights structure scanner triggers (Z > threshold)
    - Shades July 2025 window and annotates flatten trigger & avoided crash
    """
    dm_ref = pd.read_csv(
        refined_oos_folder / "daily_metrics.csv",
        parse_dates=["date"],
    )

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # LEFT axis – term-structure spread
    ax1.plot(
        dm_ref["date"],
        dm_ref["carry_spread"],
        linewidth=1.5,
        label="M1–M2 spread (refined)",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("M1–M2 Spread ($/lb)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # Structure scanner triggers
    trigger_mask = dm_ref["structure_scanner_triggered"]
    trigger_dates = dm_ref.loc[trigger_mask, "date"]
    trigger_spreads = dm_ref.loc[trigger_mask, "carry_spread"]
    ax1.scatter(
        trigger_dates,
        trigger_spreads,
        color="red",
        s=40,
        zorder=4,
        label="Structure scanner trigger\n(Z-score > threshold)",
    )

    # Shade July 2025 event window
    july_start = pd.Timestamp("2025-07-01")
    july_end = pd.Timestamp("2025-07-31")
    ax1.axvspan(july_start, july_end, color="lightgrey", alpha=0.2,
                label="July 2025 event window")

    # RIGHT axis – refined cumulative P&L
    ax2 = ax1.twinx()
    ax2.plot(
        dm_ref["date"],
        dm_ref["cumulative_pnl"],
        linestyle="--",
        linewidth=1.8,
        label="Refined cumulative P&L",
    )
    ax2.set_ylabel("Cumulative P&L ($)")

    # Optional: initial model P&L overlay for "avoided crash"
    if initial_oos_folder is not None:
        dm_init = pd.read_csv(
            initial_oos_folder / "daily_metrics.csv",
            parse_dates=["date"],
        )
        ax2.plot(
            dm_init["date"],
            dm_init["cumulative_pnl"],
            linestyle=":",
            linewidth=1.2,
            color="gray",
            label="Initial cumulative P&L",
        )

        # Annotate worst cumulative P&L day in July 2025 (crash)
        july_mask = (dm_init["date"] >= july_start) & (dm_init["date"] <= july_end)
        if july_mask.any():
            july_slice = dm_init.loc[july_mask]
            crash_idx = july_slice["cumulative_pnl"].idxmin()
            crash_date = dm_init.loc[crash_idx, "date"]
            crash_pnl = dm_init.loc[crash_idx, "cumulative_pnl"]
            ax2.scatter(crash_date, crash_pnl, color="black", s=50, zorder=5)
            ax2.annotate(
                "Initial model crash\n(July 2025)",
                xy=(crash_date, crash_pnl),
                xytext=(0, -50),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=1.0),
            )

    # Annotate first structure scanner trigger as "flatten trigger"
    if trigger_mask.any():
        first_idx = dm_ref[trigger_mask].index[0]
        flat_date = dm_ref.loc[first_idx, "date"]
        flat_spread = dm_ref.loc[first_idx, "carry_spread"]
        ax1.annotate(
            "Structure scanner\nflatten trigger",
            xy=(flat_date, flat_spread),
            xytext=(20, 30),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=1.0),
        )

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)

    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
