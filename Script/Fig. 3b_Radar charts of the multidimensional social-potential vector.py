import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configuration
csv_path = r"C:\github\Data\data.csv"
out_path = r"C:\github\Data\Result_Fig.3b_radar.png"
cities = ["beijing", "london", "newyork", "mumbai"]
metrics = ["green", "hp", "infra", "build", "cctv"]
ues_color = "#B52C34"
uos_color = "#1E4681"
dpi = 300

ncols = 4
nrows = 1
figsize = (8, 3)
fill_alpha = 0.12
line_width = 1
marker_size = 3

CITY_NAMES = {
    "beijing": "Beijing",
    "london": "London",
    "newyork": "New York",
    "mumbai": "Mumbai"
}

df = pd.read_csv(csv_path)
df['city'] = df['city'].astype(str).str.strip().str.lower()
grouped = df.groupby(['city', 'ue'])[metrics].mean().reset_index()

n_cities = len(cities)
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         subplot_kw=dict(polar=True),
                         figsize=figsize)

axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

print("=" * 40)
print("Drawing radar chart, metrics for each city:")
print("=" * 40)

for ax_idx, city in enumerate(cities):
    ax = axes[ax_idx]

    display_name = CITY_NAMES.get(city, city.capitalize())

    row_ues = grouped[(grouped['city'] == city) & (grouped['ue'] == 1)]
    row_uos = grouped[(grouped['city'] == city) & (grouped['ue'] == 0)]

    vals_ues = row_ues[metrics].iloc[0].tolist() if not row_ues.empty else [np.nan] * N
    vals_uos = row_uos[metrics].iloc[0].tolist() if not row_uos.empty else [np.nan] * N

    # Print metrics to console
    print(f"[{display_name}]")

    print(f"  UES (Red):")
    for m, v in zip(metrics, vals_ues):
        if np.isnan(v):
            print(f"    {m}: NaN")
        else:
            print(f"    {m}: {v:.4f}")

    print(f"  UOS (Blue):")
    for m, v in zip(metrics, vals_uos):
        if np.isnan(v):
            print(f"    {m}: NaN")
        else:
            print(f"    {m}: {v:.4f}")
    print("-" * 40)

    vals_ues_closed = vals_ues + vals_ues[:1]
    vals_uos_closed = vals_uos + vals_uos[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.9, zorder=3)

    ax.spines['polar'].set_color('0.6')
    ax.spines['polar'].set_linewidth(0.8)

    # Plot UOS (blue) then UES (red)
    if not all(np.isnan(vals_uos)):
        ax.fill(angles, vals_uos_closed, color=uos_color, alpha=fill_alpha, zorder=1)
        ax.plot(angles, vals_uos_closed, color=uos_color, linewidth=line_width, zorder=3)
        ax.scatter(angles[:-1], vals_uos, s=marker_size, color=uos_color, zorder=4)

    if not all(np.isnan(vals_ues)):
        ax.fill(angles, vals_ues_closed, color=ues_color, alpha=fill_alpha, zorder=5)
        ax.plot(angles, vals_ues_closed, color=ues_color, linewidth=line_width, zorder=7)
        ax.scatter(angles[:-1], vals_ues, s=marker_size, color=ues_color, zorder=8)

    ax.set_title(display_name, va='bottom', fontsize=12)

# Global legend
legend_elements = [
    Line2D([0], [0], color=ues_color, lw=line_width * 2,
           marker='o', markersize=marker_size * 2.5, markerfacecolor=ues_color,
           label='UES'),
    Line2D([0], [0], color=uos_color, lw=line_width * 2,
           marker='o', markersize=marker_size * 2.5, markerfacecolor=uos_color,
           label='UOS')
]

fig.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.09, 0.1),
           ncol=2, frameon=False, fontsize=11)

fig.subplots_adjust(wspace=0.6, bottom=0.15)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
plt.close(fig)

print(f"\nRadar chart successfully saved to: {out_path}")