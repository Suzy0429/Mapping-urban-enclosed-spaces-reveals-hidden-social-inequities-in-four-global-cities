import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

# Configuration
CSV_PATH = r"C:\github\Data\data.csv"
OUT_DIR = r"C:\github\Data"
DPI = 300
FIGSIZE = (15, 10)
FONT_SIZE = 24
Y_LIM = (0.0, 0.45)
BOX_ALPHA = 0.4
LINE_WIDTH = 2

TYPE_ORDER = ['residential', 'institutional', 'commercial', 'industrial', 'park']
PALETTE = {"UES": "#B52C34", "UOS": "#1E4681"}
HUE_ORDER = ["UES", "UOS"]
CITY_NAMES = {
    'beijing': 'Beijing',
    'london': 'London',
    'newyork': 'New York',
    'mumbai': 'Mumbai'
}


def p_to_stars(p):
    if p is None or np.isnan(p):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='ticks')
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
    'axes.linewidth': LINE_WIDTH,
    'xtick.major.width': LINE_WIDTH,
    'ytick.major.width': LINE_WIDTH,
})

# Data Processing
df = pd.read_csv(CSV_PATH)
df = df.copy()
df['city'] = df['city'].astype(str).str.lower()
df['type'] = df['type'].astype(str).str.lower()

if df['ue'].dtype == object:
    df['ue'] = df['ue'].map({'0': 0, '1': 1}).fillna(df['ue'])
df['ue'] = pd.to_numeric(df['ue'], errors='coerce').fillna(0).astype(int)
df['ue_label'] = df['ue'].map({1: 'UES', 0: 'UOS'})

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=FIGSIZE)
axes = axes.flatten()

for idx, city in enumerate(CITY_NAMES.keys()):
    ax = axes[idx]
    df_city = df[df['city'] == city].copy()

    if df_city.empty:
        ax.set_visible(False)
        continue

    city_types = df_city['type'].unique()
    unique_types_city = [t for t in TYPE_ORDER if t in city_types]

    if not unique_types_city:
        ax.set_visible(False)
        continue

    sns.stripplot(
        x='type', y='D', hue='ue_label', data=df_city,
        order=unique_types_city, hue_order=HUE_ORDER, dodge=True,
        jitter=True, size=5, alpha=0.45, palette=PALETTE,
        ax=ax, zorder=1
    )

    box_kwargs = {
        'x': 'type', 'y': 'D', 'hue': 'ue_label', 'data': df_city,
        'order': unique_types_city, 'hue_order': HUE_ORDER, 'palette': PALETTE,
        'showfliers': False, 'linewidth': LINE_WIDTH, 'ax': ax, 'width': 0.8, 'zorder': 2
    }

    try:
        if int(sns.__version__.split('.')[1]) >= 13:
            box_kwargs['gap'] = 0.15
    except:
        pass

    box = sns.boxplot(**box_kwargs)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for obj in list(ax.patches) + list(ax.artists):
        if isinstance(obj, matplotlib.patches.PathPatch) or isinstance(obj, matplotlib.patches.Rectangle):
            fc = obj.get_facecolor()
            if len(fc) >= 3:
                rgba = (*fc[:3], BOX_ALPHA)
                obj.set_facecolor(rgba)
                obj.set_edgecolor('black')
                obj.set_linewidth(LINE_WIDTH)

    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(LINE_WIDTH)

    ax.set_ylim(Y_LIM)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])

    y_min, y_max = Y_LIM
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
    offset = 0.14

    for i, t in enumerate(unique_types_city):
        g1 = df_city[(df_city['type'] == t) & (df_city['ue'] == 1)]['D'].dropna()
        g0 = df_city[(df_city['type'] == t) & (df_city['ue'] == 0)]['D'].dropna()

        if g1.empty or g0.empty:
            continue

        if (len(g1) < 3) or (len(g0) < 3):
            yloc = max(g1.max() if not g1.empty else y_min, g0.max() if not g0.empty else y_min) + 0.04 * y_range
            ax.text(i, yloc, f"n:{len(g1)}/{len(g0)}", ha='center', va='bottom', fontsize=FONT_SIZE - 4)
            continue

        try:
            stat, p = mannwhitneyu(g1, g0, alternative='two-sided')
        except Exception:
            p = np.nan

        stars = p_to_stars(p)
        top = min(max(g1.max(), g0.max()) + 0.03 * y_range, Y_LIM[1] - 0.02 * y_range)
        line_y = top
        tick_height = 0.01 * y_range
        x1 = i - offset
        x2 = i + offset

        ax.plot([x1, x2], [line_y, line_y], lw=LINE_WIDTH, color='black', zorder=3)
        ax.plot([x1, x1], [line_y, line_y - tick_height], lw=LINE_WIDTH, color='black', zorder=3)
        ax.plot([x2, x2], [line_y, line_y - tick_height], lw=LINE_WIDTH, color='black', zorder=3)

        ax.text(i, line_y + 0.01 * y_range, stars, ha='center', va='bottom', fontsize=FONT_SIZE, zorder=4)

    ax.set_xlabel('')
    display_city = CITY_NAMES.get(city, city.title())
    ax.set_ylabel(f'Φ ({display_city})')
    ax.set_xticklabels([str(t).capitalize() for t in unique_types_city], rotation=15, ha='center')
    sns.despine(ax=ax)

legend_handles = [
    Patch(facecolor=mcolors.to_rgba(PALETTE['UES'], BOX_ALPHA), edgecolor='black', linewidth=LINE_WIDTH, label='UES'),
    Patch(facecolor=mcolors.to_rgba(PALETTE['UOS'], BOX_ALPHA), edgecolor='black', linewidth=LINE_WIDTH, label='UOS')
]
fig.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.08, 0.9), ncol=2, frameon=False,
           fontsize=FONT_SIZE)

plt.tight_layout(rect=[0, 0, 1, 0.98])

out_path = os.path.join(OUT_DIR, 'Result_Fig.3a_boxplot.png')
try:
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print(f"Figure successfully saved to: {out_path}")
except Exception as e:
    print(f"Failed to save figure {out_path}: {e}")

plt.close(fig)
print('Plotting completed.')