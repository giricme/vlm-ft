"""Generate publication-quality plots for VLM fine-tuning tutorial and blog."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

UPLOAD = Path("../training_diary")
OUT = Path("./plots")
OUT.mkdir(exist_ok=True)

# Colors
C_S1 = '#2E86AB'        # Blue
C_S2R1 = '#A23B72'      # Magenta
C_S2R2 = '#E8702A'      # Orange
C_S2R3 = '#2CA58D'      # Teal
C_CRASH = '#D32F2F'     # Red
C_LOST = '#FFCDD2'      # Light red
C_GRID = '#E0E0E0'

# Load data
s1t = pd.read_csv(UPLOAD / "robovqa_stage1_full_train_metrics.csv")
s1e = pd.read_csv(UPLOAD / "robovqa_stage1_full_eval_metrics.csv")
s2r1t = pd.read_csv(UPLOAD / "first_robovqa_stage2_full_train_metrics.csv")
s2r1e = pd.read_csv(UPLOAD / "first_robovqa_stage2_full_eval_metrics.csv")
s2r2t = pd.read_csv(UPLOAD / "second_robovqa_stage2_full_train_metrics.csv")
s2r2e = pd.read_csv(UPLOAD / "second_robovqa_stage2_full_eval_metrics.csv")
s2r3t = pd.read_csv(UPLOAD / "robovqa_stage2_full_train_metrics.csv")

for df in [s1t, s1e, s2r1t, s2r1e, s2r2t, s2r2e, s2r3t]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== Data Summary ===")
for name, df in [("S1 train", s1t), ("S2 R1 train", s2r1t), ("S2 R2 train", s2r2t), ("S2 R3 train", s2r3t)]:
    print(f"{name}: steps {df.step.min()}-{df.step.max()} ({len(df)} rows), "
          f"{df.timestamp.min().strftime('%b %d')} to {df.timestamp.max().strftime('%b %d')}")

W = 50  # smoothing window

# ============================================================
# PLOT 1: Full Training Journey - Loss by Date
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(s1t.timestamp, s1t.loss.rolling(W, min_periods=1).mean(),
        color=C_S1, alpha=0.8, lw=1.2, label='Stage 1: Visual Grounding')
ax.plot(s2r1t.timestamp, s2r1t.loss.rolling(W, min_periods=1).mean(),
        color=C_S2R1, alpha=0.8, lw=1.2, label='Stage 2 Run 1 (crashed)')
ax.plot(s2r2t.timestamp, s2r2t.loss.rolling(W, min_periods=1).mean(),
        color=C_S2R2, alpha=0.8, lw=1.2, label='Stage 2 Run 2 (crashed)')
ax.plot(s2r3t.timestamp, s2r3t.loss.rolling(W, min_periods=1).mean(),
        color=C_S2R3, alpha=0.8, lw=1.2, label='Stage 2 Run 3 (current)')

# Crash markers
for t, y_pos, label in [
    (s2r1t.timestamp.max(), 0.17, 'Crash #1\n2,186 steps lost'),
    (s2r2t.timestamp.max(), 0.13, 'Crash #2\n429 steps lost'),
]:
    ax.axvline(t, color=C_CRASH, ls='--', alpha=0.6, lw=1)
    ax.annotate(label, xy=(t, y_pos), fontsize=9, color=C_CRASH, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc=C_LOST, ec=C_CRASH, alpha=0.85))

# Stage boundary
sb = s1t.timestamp.max()
ax.axvline(sb, color='gray', ls=':', alpha=0.5, lw=1.5)
ax.annotate('Stage 1 → 2', xy=(sb, 0.5), fontsize=10, color='gray', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

ax.set_xlabel('Date')
ax.set_ylabel('Training Loss (smoothed)')
ax.set_title('~50 Days of GPU Compute: Training Loss Over Time')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, color=C_GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=30)
fig.tight_layout()
fig.savefig(OUT / '01_training_journey_timeline.png')
plt.close()
print("saved 01_training_journey_timeline.png")

# ============================================================
# PLOT 2: Eval Loss Trajectory (both stages)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(s1e.step, s1e.loss, 'o-', color=C_S1, ms=4, lw=1.5)
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Eval Loss')
ax1.set_title('Stage 1: Eval Loss')
ax1.grid(True, alpha=0.3, color=C_GRID)
ax1.annotate(f'Start: {s1e.loss.iloc[0]:.3f}', xy=(s1e.step.iloc[0], s1e.loss.iloc[0]),
             fontsize=9, ha='left', va='bottom', color='gray')
ax1.annotate(f'Final: {s1e.loss.iloc[-1]:.3f}', xy=(s1e.step.iloc[-1], s1e.loss.iloc[-1]),
             fontsize=10, ha='right', va='top', color=C_S1, fontweight='bold')

# Combine Stage 2 evals, dedup by step keeping latest
s2e_all = pd.concat([s2r1e, s2r2e]).drop_duplicates('step', keep='last').sort_values('step')
ax2.plot(s2e_all.step, s2e_all.loss, 'o-', color=C_S2R1, ms=4, lw=1.5)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Eval Loss')
ax2.set_title('Stage 2: Eval Loss')
ax2.grid(True, alpha=0.3, color=C_GRID)
ax2.annotate(f'Start: {s2e_all.loss.iloc[0]:.4f}', xy=(s2e_all.step.iloc[0], s2e_all.loss.iloc[0]),
             fontsize=9, ha='left', va='bottom', color='gray')
ax2.annotate(f'Latest: {s2e_all.loss.iloc[-1]:.4f}', xy=(s2e_all.step.iloc[-1], s2e_all.loss.iloc[-1]),
             fontsize=10, ha='right', va='top', color=C_S2R1, fontweight='bold')

fig.suptitle('Evaluation Loss: Continuous Improvement Across Both Stages', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / '02_eval_loss_trajectory.png')
plt.close()
print("saved 02_eval_loss_trajectory.png")

# ============================================================
# PLOT 3: Crash & Recovery Timeline (Gantt-style)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))

s2_end = 29835

# Run 1
r1_start, r1_end = s2r1t.step.min(), s2r1t.step.max()
ax.barh(2, r1_end - r1_start, left=r1_start, height=0.55, color=C_S2R1, alpha=0.7)
ax.barh(2, 27186 - 25000, left=25000, height=0.55, color=C_LOST, alpha=0.6, hatch='///')

# Run 2
r2_start, r2_end = s2r2t.step.min(), s2r2t.step.max()
ax.barh(1, r2_end - r2_start, left=r2_start, height=0.55, color=C_S2R2, alpha=0.7)
ax.barh(1, 28929 - 28500, left=28500, height=0.55, color=C_LOST, alpha=0.6, hatch='///')

# Run 3
r3_start, r3_end = s2r3t.step.min(), s2r3t.step.max()
ax.barh(0, r3_end - r3_start, left=r3_start, height=0.55, color=C_S2R3, alpha=0.7)

# Crash & checkpoint markers
ax.plot(27186, 2, 'X', color=C_CRASH, ms=14, zorder=5)
ax.plot(28929, 1, 'X', color=C_CRASH, ms=14, zorder=5)
ax.plot(25000, 2, 's', color='green', ms=9, zorder=5)
ax.plot(28500, 1, 's', color='green', ms=9, zorder=5)

ax.annotate('save_steps: 5000\n2,186 steps lost (~6 days)', xy=(26000, 2.5), fontsize=9, ha='center', color=C_CRASH)
ax.annotate('save_steps: 500\n429 steps lost (~27 hrs)', xy=(28700, 1.5), fontsize=9, ha='center', color=C_CRASH)

ax.axvline(s2_end, color='green', ls='--', alpha=0.4, lw=1)
ax.annotate(f'Target: {s2_end}', xy=(s2_end, -0.55), fontsize=9, color='green', ha='center')

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Run 3\n(current)', 'Run 2\n(save_steps=500)', 'Run 1\n(save_steps=5000)'])
ax.set_xlabel('Training Step')
ax.set_title('Stage 2: The Cost of Checkpoint Frequency')
ax.set_xlim(23000, 30500)
ax.grid(True, axis='x', alpha=0.3, color=C_GRID)

legend_elements = [
    Patch(fc=C_LOST, alpha=0.6, hatch='///', label='Lost progress'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor=C_CRASH, ms=12, label='System crash (OOM)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', ms=9, label='Last checkpoint'),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / '03_crash_recovery_timeline.png')
plt.close()
print("saved 03_crash_recovery_timeline.png")

# ============================================================
# PLOT 4: Step Time & GPU Memory Stability
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for label, color, df in [
    ('Stage 1', C_S1, s1t),
    ('Stage 2 Run 1', C_S2R1, s2r1t),
    ('Stage 2 Run 2', C_S2R2, s2r2t),
    ('Stage 2 Run 3', C_S2R3, s2r3t),
]:
    ax1.scatter(df.timestamp, df.t_step, s=1, alpha=0.3, color=color, label=label)

ax1.set_ylabel('Step Time (seconds)')
ax1.set_title('Hardware Consistency Across ~50 Days of Compute')
ax1.legend(loc='upper right', markerscale=8, framealpha=0.9)
ax1.set_ylim(0, 350)
ax1.grid(True, alpha=0.3, color=C_GRID)
ax1.annotate('Stage 1: ~93s/step', xy=(s1t.timestamp.median(), 110), fontsize=10, color=C_S1, ha='center')
ax1.annotate('Stage 2: ~230s/step', xy=(s2r2t.timestamp.median(), 250), fontsize=10, color=C_S2R2, ha='center')

for label, color, df in [
    ('Stage 1', C_S1, s1t),
    ('Stage 2 Run 1', C_S2R1, s2r1t),
    ('Stage 2 Run 2', C_S2R2, s2r2t),
    ('Stage 2 Run 3', C_S2R3, s2r3t),
]:
    ax2.scatter(df.timestamp, df.gpu_mem_gb, s=1, alpha=0.3, color=color)

ax2.set_xlabel('Date')
ax2.set_ylabel('GPU Memory (GB)')
ax2.set_ylim(11.5, 12.5)
ax2.grid(True, alpha=0.3, color=C_GRID)
ax2.annotate('Stable at ~12.0 GB throughout', xy=(s2r1t.timestamp.median(), 12.2),
             fontsize=10, color='gray', ha='center')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=30)
fig.tight_layout()
fig.savefig(OUT / '04_step_time_gpu_memory.png')
plt.close()
print("saved 04_step_time_gpu_memory.png")

# ============================================================
# PLOT 5: Loss by Step (traditional view)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(s1t.step, s1t.loss.rolling(W, min_periods=1).mean(), color=C_S1, alpha=0.8, lw=1.2, label='Stage 1')
ax.plot(s2r1t.step, s2r1t.loss.rolling(W, min_periods=1).mean(), color=C_S2R1, alpha=0.5, lw=1, label='Stage 2 Run 1 (lost)')
ax.plot(s2r2t.step, s2r2t.loss.rolling(W, min_periods=1).mean(), color=C_S2R2, alpha=0.5, lw=1, label='Stage 2 Run 2 (lost)')
ax.plot(s2r3t.step, s2r3t.loss.rolling(W, min_periods=1).mean(), color=C_S2R3, alpha=0.8, lw=1.2, label='Stage 2 Run 3')

ax.axvline(23594, color='gray', ls=':', alpha=0.5, lw=1.5)
ax.annotate('Stage 1 → Stage 2', xy=(23594, 0.45), fontsize=10, color='gray', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

ax.set_xlabel('Training Step')
ax.set_ylabel('Training Loss (smoothed)')
ax.set_title('Training Loss by Step')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(bottom=0, top=0.7)
ax.grid(True, alpha=0.3, color=C_GRID)
fig.tight_layout()
fig.savefig(OUT / '05_loss_by_step.png')
plt.close()
print("saved 05_loss_by_step.png")

# ============================================================
# PLOT 6: Timing Breakdown
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['Forward', 'Backward', 'Optimizer', 'Transfer', 'Data']
colors_t = ['#2E86AB', '#A23B72', '#C0C0C0', '#D4D4D4', '#E8E8E8']

for ax, df, title in [(ax1, s1t, f'Stage 1: ~{s1t.t_step.median():.0f}s/step'),
                       (ax2, s2r2t, f'Stage 2: ~{s2r2t.t_step.median():.0f}s/step')]:
    vals = [df.t_forward.median(), df.t_backward.median(), df.t_optimizer.median(),
            df.t_transfer.median(), df.t_data.median()]
    bars = ax.barh(labels, vals, color=colors_t, edgecolor='white', lw=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        if v > 0.5:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{v:.1f}s', va='center', fontsize=10)

fig.suptitle('Per-Step Timing Breakdown', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / '06_timing_breakdown.png')
plt.close()
print("saved 06_timing_breakdown.png")

# ============================================================
print(f"\n=== Key Stats ===")
print(f"Stage 1: {(s1t.timestamp.max() - s1t.timestamp.min()).days} days, "
      f"eval {s1e.loss.iloc[0]:.3f} -> {s1e.loss.iloc[-1]:.3f}")
print(f"Stage 2 best eval: {s2r2e.loss.min():.6f}")
print(f"Crash 1: lost {27186-25000} steps | Crash 2: lost {28929-28500} steps")
print(f"GPU mem: S1={s1t.gpu_mem_gb.median():.2f}GB, S2={s2r2t.gpu_mem_gb.median():.2f}GB")
print(f"\nAll plots saved to {OUT}")
