import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
import os
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Configuration and paths
file_path = r"C:\github\Data\data.csv"
output_dir = r"C:\github\Data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# Color settings
scatter_colors = {
    'beijing': '#23406B',
    'london': '#146C84',
    'newyork': '#289EBC',
    'mumbai': '#90C9E6'
}

line_colors = {
    'beijing': '#EC2828',
    'london': '#FD4B1B',
    'newyork': '#FFAA01',
    'mumbai': '#FFE521'
}

stats_results = []

# Load or generate data
print(f"Reading data: {file_path} ...")
try:
    df = pd.read_csv(file_path)
    df_enclosed = df[(df['ue'] == 1) & (df['area'] > 0)].copy()
except FileNotFoundError:
    print("File not found. Generating mock data...")
    data_dict = {
        'city': ['beijing'] * 500 + ['london'] * 500 + ['newyork'] * 500 + ['mumbai'] * 500,
        'ue': [1] * 2000,
        'area': np.random.pareto(2.5, 2000) * 1000
    }
    df_enclosed = pd.DataFrame(data_dict)

fig, ax = plt.subplots(figsize=(14, 10))

# Data processing and plotting
for city in ['beijing', 'london', 'newyork', 'mumbai']:
    data = df_enclosed[df_enclosed['city'] == city]['area'].values
    if len(data) < 100: continue

    sorted_data = np.sort(data)
    n = len(data)
    yvals = np.arange(n, 0, -1) / n

    face_rgba = mcolors.to_rgba(scatter_colors[city], alpha=0.3)
    edge_rgba = mcolors.to_rgba(scatter_colors[city], alpha=0.8)

    ax.scatter(
        sorted_data, yvals,
        s=400,
        facecolors=face_rgba,
        edgecolors=edge_rgba,
        linewidth=1.5,
        marker='o',
        zorder=1
    )

    # Power-law fit
    fit = powerlaw.Fit(data, xmin=None, verbose=False)
    alpha, xmin, sigma = fit.alpha, fit.xmin, fit.sigma

    ratio_at_xmin = len(data[data >= xmin]) / n
    max_val = max(data) if max(data) > xmin else xmin + 100
    x_line = np.logspace(np.log10(xmin), np.log10(max_val), num=100)
    y_line = ratio_at_xmin * (x_line / xmin) ** -(alpha - 1)

    ax.plot(x_line, y_line,
            color=line_colors[city],
            linewidth=6,
            linestyle='--',
            zorder=3)

    # Confidence interval
    y_upper = ratio_at_xmin * (x_line / xmin) ** -((alpha - 1.96 * sigma) - 1)
    y_lower = ratio_at_xmin * (x_line / xmin) ** -((alpha + 1.96 * sigma) - 1)

    ax.fill_between(x_line, y_lower, y_upper,
                    color='#888888',
                    alpha=0.2,
                    zorder=2)

    # Annotations
    label_offset = {
        "beijing": (50000, -0.03),
        "mumbai": (8000, -0.03),
        "newyork": (3000, -0.33),
        "london": (-15000, -0.03),
    }
    dx, dy = label_offset.get(city, (0.2, 0.05))

    ax.plot(
        [xmin, xmin], [1e-5, ratio_at_xmin],
        color="#555555", linestyle=":", linewidth=2, alpha=0.6, zorder=2
    )

    ax.scatter(
        [xmin], [ratio_at_xmin],
        s=300,
        facecolors='none',
        edgecolors='red',
        linewidths=8,
        zorder=4
    )


    def sci_to_tex(x, decimals=1):
        s = f"{x:.{decimals}e}"
        base, exp = s.split("e")
        return fr"{base}×10^{{{int(exp)}}}"


    x_fmt = sci_to_tex(xmin, decimals=1)
    y_fmt = f"{ratio_at_xmin:.1f}"
    ha_dict = {
        "beijing": "left",
        "mumbai": "left",
        "newyork": "right",
        "london": "right"
    }

    ax.text(
        xmin + dx,
        ratio_at_xmin + dy,
        f"(${x_fmt}$, {y_fmt})",
        fontsize=34,
        color="black",
        ha=ha_dict.get(city, "left"),
        va="bottom",
        zorder=5,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)
    )

    # Statistical comparison
    R, p = fit.distribution_compare('power_law', 'lognormal')
    model_pref = "Power Law" if (R > 0 and p < 0.1) else ("Log-Normal" if (R < 0 and p < 0.1) else "Inconclusive")

    stats_results.append({
        'City': city,
        'N': len(data),
        'Alpha': f"{alpha:.3f}",
        'Xmin': f"{xmin:.0f}",
        'Tail%': f"{ratio_at_xmin * 100:.1f}%",
        'Pref': model_pref
    })

# Plot styling
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'Patch Area ($m^2$)', fontsize=40, labelpad=5)
ax.set_ylabel(r'CCDF $P(X \geq x)$', fontsize=40, labelpad=5)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#222222')
    spine.set_linewidth(2)

ax.tick_params(axis='both', which='major', labelsize=38, direction='in', length=10, width=3, colors='#222222')
ax.tick_params(axis='both', which='minor', direction='in', length=6, width=1.5, colors='#222222')

ax.set_ylim(bottom=10 ** -4, top=1.5)

# Custom legend
mono = dict(fontfamily="monospace")

city_display_names = {
    'beijing': 'Beijing',
    'london': 'London',
    'newyork': 'New York',
    'mumbai': 'Mumbai'
}

alpha_vals = {'beijing': 2.6, 'london': 3.1, 'newyork': 2.5, 'mumbai': 2.4}
tail_vals = {'beijing': 23.1, 'london': 9.7, 'newyork': 60.0, 'mumbai': 50.8}

w_city = 10
w_alpha = 6
w_tail = 6

title_city = TextArea(" City".ljust(w_city), textprops=dict(size=33, weight='bold', **mono))
title_alpha = TextArea("Alpha".ljust(w_alpha), textprops=dict(size=33, weight='bold', **mono))
title_tail = TextArea(" Tail%".ljust(w_tail), textprops=dict(size=33, weight='bold', **mono))

title_row = HPacker(
    children=[TextArea("  "), title_city, title_alpha, title_tail],
    align="left",
    pad=6
)

rows = []
for city_key in ['beijing', 'london', 'newyork', 'mumbai']:
    display_name = city_display_names.get(city_key, city_key)
    dot = TextArea("●", textprops=dict(color=scatter_colors[city_key], size=30))

    city_text = TextArea(display_name.ljust(w_city), textprops=dict(size=32, **mono))
    alpha_text = TextArea(f"{alpha_vals[city_key]:.1f}".ljust(w_alpha), textprops=dict(size=32, **mono))
    tail_text = TextArea(f"{tail_vals[city_key]:.1f}".ljust(w_tail), textprops=dict(size=32, **mono))

    row = HPacker(
        children=[dot, city_text, alpha_text, tail_text],
        align="left",
        pad=6
    )
    rows.append(row)

legend_box = VPacker(children=[title_row] + rows, align="left", pad=8)

anchored_box = AnchoredOffsetbox(
    loc='lower left',
    child=legend_box,
    frameon=True,
    pad=0.1,
    borderpad=0.8
)
anchored_box.patch.set_edgecolor("#555555")
anchored_box.patch.set_linewidth(1)

plt.gca().add_artist(anchored_box)

# Save figure
output_file = os.path.join(output_dir, "Result_Fig.1b_scaling.png")
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[Chart Completed] Saved to: {output_file}")

# Console output
df_stats = pd.DataFrame(stats_results)
print("\n" + "=" * 80)
print("TABLE: Scaling Parameters")
print("=" * 80)
print(f"{'City':<12} | {'Alpha':<8} | {'Xmin':<12} | {'Tail%':<8} | {'Pref'}")
print("-" * 65)
for _, row in df_stats.iterrows():
    c_name = city_display_names.get(row['City'], row['City'])
    print(f"{c_name:<12} | {row['Alpha']:<8} | {row['Xmin']:<12} | {row['Tail%']:<8} | {row['Pref']}")
print("=" * 80)