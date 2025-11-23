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
def plot_autopsy_july_2025_initial(
    initial_oos_folder: Path,
    save_path: Path,
    band_width: float = 0.20,
):
    """
    Autopsy chart for the July 2025 loss in the ORIGINAL model.

    Parameters
    ----------
    initial_oos_folder : Path
        Folder for Part A out_of_sample_2025 (initial model),
        e.g. Path("output_partA/out_of_sample_2025")
        Must contain daily_metrics.csv and positions.csv.

    save_path : Path
        Location to save the PNG (e.g. Path("figures/autopsy_july2025_initial.png"))

    band_width : float
        Width (in $/lb) of the stop-limit band ABOVE the initial stop.
        Set this to match your actual config (e.g. based on overnight
        spread + 4 ticks). For the assignment figure 0.20 works well
        visually; adjust as needed.
    """
    dm = pd.read_csv(initial_oos_folder / "daily_metrics.csv", parse_dates=["date"])
    pos = pd.read_csv(
        initial_oos_folder / "positions.csv",
        parse_dates=["entry_date", "exit_date"],
    )

    # Identify the July 7 2025 short position (Position #36)
    trade = pos.loc[
        (pos["entry_date"] == "2025-07-07") & (pos["direction"] == "SHORT")
    ].iloc[0]

    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]
    entry_price = trade["entry_price"]
    exit_price = trade["exit_price"]
    stop_price = trade["initial_stop"]  # ~7.98 in your run

    # Date window for chart: July 5–15
    mask = (dm["date"] >= "2025-07-05") & (dm["date"] <= "2025-07-15")
    window = dm.loc[mask].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Price line (front continuous ≈ M2 proxy)
    ax.plot(
        window["date"],
        window["current_price"],
        marker="o",
        linewidth=2,
        label="HG M2 / Front price",
    )

    # Entry marker
    ax.scatter(
        entry_date,
        entry_price,
        marker="v",
        s=80,
        color="tab:red",
        label="Short entry $7.72",
        zorder=5,
    )

    # Gap day: use July 8 price from daily_metrics
    gap_date = pd.Timestamp("2025-07-08")
    gap_price = window.loc[window["date"] == gap_date, "current_price"].iloc[0]
    ax.scatter(
        gap_date,
        gap_price,
        marker="^",
        s=90,
        color="tab:orange",
        label="Gap open (stop skipped)",
        zorder=5,
    )

    # Exit marker
    ax.scatter(
        exit_date,
        exit_price,
        marker="x",
        s=90,
        color="black",
        label="Final exit $8.47",
        zorder=5,
    )

    # Stop level
    ax.axhline(
        stop_price,
        linestyle="--",
        linewidth=1.5,
        color="tab:gray",
        label=f"Stop loss {stop_price:.2f}",
    )

    # Stop-limit band: stop → stop + band_width
    ax.fill_between(
        window["date"],
        stop_price,
        stop_price + band_width,
        color="gold",
        alpha=0.2,
        label="Stop-limit band",
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
    Dual-axis chart: term-structure spread vs refined cumulative P&L.

    • Left axis: carry_spread (M1–M2 spread proxy)
    • Right axis: refined cumulative P&L
    • Mark days where structure_scanner_triggered == True
      (these correspond to spread Z-score > threshold, e.g. > 3).
    • Optionally overlay the INITIAL model's cumulative P&L as a
      dashed line for the 'avoided crash' narrative.

    Parameters
    ----------
    refined_oos_folder : Path
        Folder for Part B out_of_sample_2025 (refined model),
        containing daily_metrics.csv.

    save_path : Path
        Output PNG path.

    initial_oos_folder : Path, optional
        If provided, the function will read
        initial_oos_folder/daily_metrics.csv and plot its
        cumulative_pnl as a dashed line for comparison.
    """
    dm_ref = pd.read_csv(
        refined_oos_folder / "daily_metrics.csv",
        parse_dates=["date"],
    )

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Left axis: spread
    ax1.plot(
        dm_ref["date"],
        dm_ref["carry_spread"],
        linewidth=1.5,
        label="M1–M2 spread (refined)",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("M1–M2 Spread ($/lb)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # Highlight structure scanner triggers
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

    # Right axis: refined cumulative P&L
    ax2 = ax1.twinx()
    ax2.plot(
        dm_ref["date"],
        dm_ref["cumulative_pnl"],
        linestyle="--",
        linewidth=1.8,
        label="Refined cumulative P&L",
    )
    ax2.set_ylabel("Cumulative P&L ($)")

    # Optionally overlay initial model P&L (dotted)
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

    # Build combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        fontsize=8,
    )

    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
