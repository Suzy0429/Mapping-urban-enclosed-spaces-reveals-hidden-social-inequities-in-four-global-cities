import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

INPUT_FILE = r"C:\github\Data\data.csv"
OUTPUT_DIR = r"C:\github\Data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TYPE_COLORS = {
    'Residential': (207 / 255, 66 / 255, 78 / 255),
    'Institutional': (250 / 255, 238 / 255, 213 / 255),
    'Commercial': (48 / 255, 130 / 255, 173 / 255),
    'Industrial': (126 / 255, 190 / 255, 214 / 255),
    'Park': (189 / 255, 208 / 255, 197 / 255)
}

TYPE_ORDER = ['Residential', 'Institutional', 'Commercial', 'Industrial', 'Park']
PATTERN_ORDER = ['HΦ-HC', 'HΦ-LC', 'LΦ-HC', 'LΦ-LC']

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE = 19

# Data Processing
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_cols = {'city', 'ue', 'type', 'D', 'C'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_ue1 = df[df['ue'] == 1].copy()
df_ue1['city'] = df_ue1['city'].astype(str).str.strip().str.lower()
df_ue1['type'] = df_ue1['type'].astype(str).str.strip().str.capitalize()

# Plotting
cities = list(df_ue1['city'].unique())
if 'newyork' in cities and 'mumbai' in cities:
    idx_ny = cities.index('newyork')
    idx_mb = cities.index('mumbai')
    cities[idx_ny], cities[idx_mb] = cities[idx_mb], cities[idx_ny]

# Create 2x2 Grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
axes = axes.flatten()

for idx, city in enumerate(cities):
    if idx >= 4:
        print(f"Warning: More than 4 cities detected. Only plotting the first 4 (ignoring {city} and beyond).")
        break

    print(f"Plotting city: {city}")
    sub_df = df_ue1[df_ue1['city'] == city].copy()
    if sub_df.empty:
        continue

    ic_mean = sub_df['C'].mean()
    d_mean = sub_df['D'].mean()

    def get_pattern(row):
        if row['D'] >= d_mean and row['C'] >= ic_mean:
            return 'HΦ-HC'
        elif row['D'] >= d_mean and row['C'] < ic_mean:
            return 'HΦ-LC'
        elif row['D'] < d_mean and row['C'] >= ic_mean:
            return 'LΦ-HC'
        else:
            return 'LΦ-LC'

    sub_df['pattern'] = sub_df.apply(get_pattern, axis=1)

    counts = sub_df.groupby(['pattern', 'type']).size().unstack(fill_value=0)
    counts = counts.reindex(index=PATTERN_ORDER, fill_value=0)

    existing_types = [t for t in TYPE_ORDER if t in counts.columns]
    other_types = [t for t in counts.columns if t not in TYPE_ORDER]
    final_types = existing_types + other_types
    counts = counts[final_types]

    totals = counts.sum(axis=1).values
    city_total = totals.sum()

    ax = axes[idx]
    display_name = "New York" if city == "newyork" else city.capitalize()

    ax.text(0.03, 0.9, display_name, fontsize=FONT_SIZE + 4, fontweight='bold',
            ha='left', va='bottom', transform=ax.transAxes)

    BAR_WIDTH = 0.45
    bottom_heights = np.zeros(len(PATTERN_ORDER))
    labels_data = {i: [] for i in range(len(PATTERN_ORDER))}

    for t_idx, t_name in enumerate(final_types):
        values = counts[t_name].values
        color = TYPE_COLORS.get(t_name, '#999999')

        ax.bar(
            PATTERN_ORDER,
            values,
            bottom=bottom_heights,
            color=color,
            edgecolor='black',
            linewidth=1.2,
            width=BAR_WIDTH,
            zorder=3
        )

        for p_idx, val in enumerate(values):
            if val > 0 and totals[p_idx] > 0:
                pct = int(round(val / totals[p_idx] * 100))
                if pct > 0:
                    y_center = bottom_heights[p_idx] + val / 2
                    labels_data[p_idx].append({
                        'text': f"{pct}%",
                        'y_orig': y_center
                    })

        bottom_heights += values

    max_total = max(totals) if len(totals) > 0 else 100
    min_dist = max_total * 0.18

    for p_idx in range(len(PATTERN_ORDER)):
        bar_labels = labels_data[p_idx]
        if not bar_labels:
            continue

        y_positions = [lbl['y_orig'] for lbl in bar_labels]

        for _ in range(100):
            moved = False
            for i in range(len(y_positions) - 1):
                if y_positions[i + 1] - y_positions[i] < min_dist:
                    overlap = min_dist - (y_positions[i + 1] - y_positions[i])
                    y_positions[i + 1] += overlap * 0.85
                    y_positions[i] -= overlap * 0
                    moved = True

            if y_positions[0] < 0:
                y_positions[0] = 0
            if not moved:
                break

        x_right_edge = p_idx + BAR_WIDTH / 2
        elbow_dx1 = 0.04
        elbow_dx2 = 0.06
        elbow_dx3 = 0.03
        text_pad = 0.015

        for i, lbl in enumerate(bar_labels):
            y_orig = lbl['y_orig']
            y_adj = y_positions[i]
            text_str = lbl['text']

            x0, y0 = x_right_edge, y_orig
            x1, y1 = x0 + elbow_dx1, y_orig
            x2, y2 = x1 + elbow_dx2, y_adj
            x3, y3 = x2 + elbow_dx3, y_adj

            ax.plot([x0, x1, x2, x3], [y0, y1, y2, y3], color='black', lw=1.0, zorder=4)
            ax.text(x3 + text_pad, y3, text_str, va='center', ha='left',
                    fontsize=FONT_SIZE - 2, color='black', zorder=5)

    for p_idx, bar_total in enumerate(totals):
        if city_total > 0 and bar_total > 0:
            pattern_pct = int(round(bar_total / city_total * 100))
            offset = max_total * 0.06

            ax.text(p_idx, bar_total + offset, f"{pattern_pct}%",
                    ha='center', va='bottom', fontsize=FONT_SIZE,
                    fontweight='bold', color='black', zorder=6)

    ax.set_ylim(0, max_total * 1.25)

    # Aesthetics
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_xlim(-0.5, len(PATTERN_ORDER))

# Hide unused subplots
for i in range(len(cities), 4):
    fig.delaxes(axes[i])

# Global Legend and Save
print("Generating and saving 2x2 grid with global legend...")

legend_handles = [
    mpatches.Patch(facecolor=TYPE_COLORS[t_name], edgecolor='black', linewidth=1.2, label=t_name)
    for t_name in TYPE_ORDER
]

fig.legend(handles=legend_handles, loc='lower center', frameon=False,
           ncol=len(TYPE_ORDER), fontsize=FONT_SIZE + 1, bbox_to_anchor=(0.5, -0.05),
           columnspacing=0.9)

plt.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(OUTPUT_DIR, "Result_Fig.3d_scatter_bar.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False)
plt.close(fig)

print(f"\nAll cumulative bar charts successfully merged and saved!")
print(f" -> Saved to: {out_path}")