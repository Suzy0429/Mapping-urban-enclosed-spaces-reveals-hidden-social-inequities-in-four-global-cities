import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import numpy as np
import os

# Configuration and Colors
INPUT_FILE = r"C:\github\Data\data.csv"
OUTPUT_DIR = r"C:\github\Data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rgb_to_norm(rgb_tuple):
    return tuple(c / 255.0 for c in rgb_tuple)

CITY_PALETTE = ['#450357', '#238A8D', '#89D547', '#FAE622']
CITY_ORDER = ['beijing', 'london', 'newyork', 'mumbai']
CITY_NAMES = {
    'beijing': 'Beijing',
    'london': 'London',
    'newyork': 'New York',
    'mumbai': 'Mumbai'
}

BG_Q1 = rgb_to_norm((250, 236, 237))
BG_Q3 = rgb_to_norm((236, 245, 249))

TICK_FS = 22
AXIS_LABEL_FS = 28
QUAD_LABEL_FS = 18

# Custom Area Legend Handler
class AreaMarker:
    pass

class AreaHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center_y = height / 2.0
        r1 = height * 0.6
        r2 = height * 1
        x1 = width * 0.1
        x2 = width * 0.8

        d = x2 - x1
        if d == 0: d = 1
        theta = np.arcsin((r2 - r1) / d)

        x1_t = x1 - r1 * np.sin(theta)
        y1_t = center_y + r1 * np.cos(theta)
        x2_t = x2 - r2 * np.sin(theta)
        y2_t = center_y + r2 * np.cos(theta)

        x1_b = x1 - r1 * np.sin(theta)
        y1_b = center_y - r1 * np.cos(theta)
        x2_b = x2 - r2 * np.sin(theta)
        y2_b = center_y - r2 * np.cos(theta)

        l1 = Line2D([x1_t, x2_t], [y1_t, y2_t], color='black', linestyle='-', linewidth=0.8, transform=trans)
        l2 = Line2D([x1_b, x2_b], [y1_b, y2_b], color='black', linestyle='-', linewidth=0.8, transform=trans)
        c1 = patches.Circle((x1, center_y), r1, fill=False, edgecolor='black', linewidth=1.2, transform=trans, zorder=3)
        c2 = patches.Circle((x2, center_y), r2, fill=False, edgecolor='black', linewidth=1.2, transform=trans, zorder=3)

        return [l1, l2, c1, c2]

# Data Processing
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
df['city'] = df['city'].astype(str).str.strip().str.lower()

df_plot = df[df['ue'] == 1].copy()
global_ic_mean = df_plot['C'].mean()
global_d_mean = df_plot['D'].mean()

# Plotting
fig, ax = plt.subplots(figsize=(10.5, 6))

fixed_x_min, fixed_x_max = 0.0, 1.0
fixed_y_min, fixed_y_max = 0.0, 0.35
ax.set_xlim(fixed_x_min, fixed_x_max)
ax.set_ylim(fixed_y_min, fixed_y_max)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Quadrant backgrounds and annotations
rect_q1 = patches.Rectangle((global_ic_mean, global_d_mean), width=xlim[1] - global_ic_mean,
                            height=ylim[1] - global_d_mean, linewidth=0, facecolor=BG_Q1, zorder=0)
rect_q3 = patches.Rectangle((xlim[0], ylim[0]), width=global_ic_mean - xlim[0], height=global_d_mean - ylim[0],
                            linewidth=0, facecolor=BG_Q3, zorder=0)
ax.add_patch(rect_q1)
ax.add_patch(rect_q3)

bbox_props = dict(boxstyle='round,pad=0.2', facecolor=(1.0, 1.0, 1.0, 0.6), edgecolor=(0.0, 0.0, 0.0, 0.3),
                  linewidth=1.1)
base_kwargs = {'fontsize': QUAD_LABEL_FS, 'color': '#333333', 'fontweight': 'bold', 'zorder': 30}
x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
pad_x, pad_y = x_span * 0.02, y_span * 0.02

ax.text(xlim[1] - pad_x, ylim[1] - pad_y, "HΦ\nHC", ha='right', va='top', bbox=bbox_props, **base_kwargs)
ax.text(xlim[1] - pad_x, ylim[0] + pad_y, "LΦ\nHC", ha='right', va='bottom', bbox=bbox_props, **base_kwargs)
ax.text(xlim[0] + pad_x, ylim[1] - pad_y, "HΦ\nLC", ha='left', va='top', bbox=bbox_props, **base_kwargs)
ax.text(xlim[0] + pad_x, ylim[0] + pad_y, "LΦ\nLC", ha='left', va='bottom', bbox=bbox_props, **base_kwargs)

# Scatter plot
area_min, area_max = df_plot['area'].min(), df_plot['area'].max()

for city in df_plot['city'].unique():
    sub_df = df_plot[df_plot['city'] == city]
    if sub_df.empty: continue

    sizes = 20 + (sub_df['area'] - area_min) / (area_max - area_min) * 2000 if area_max != area_min else [300] * len(sub_df)
    city_color = CITY_PALETTE[CITY_ORDER.index(city)] if city in CITY_ORDER else '#999999'

    ax.scatter(sub_df['C'], sub_df['D'], s=sizes, c=city_color, alpha=1, edgecolor='#555555', linewidth=0.5, zorder=10)

# Global lines and axis labels
ax.axhline(y=global_d_mean, color='#555555', lw=2, zorder=20, dashes=(5, 6))
ax.axvline(x=global_ic_mean, color='#555555', lw=2, zorder=20, dashes=(5, 6))

ax.text(xlim[1] + x_span * 0.01, global_d_mean, 'C', ha='left', va='center', fontsize=AXIS_LABEL_FS, color='#333333',
        zorder=22, clip_on=False)
ax.text(global_ic_mean, ylim[1] + y_span * 0.01, 'Φ', ha='center', va='bottom', fontsize=AXIS_LABEL_FS, color='#333333',
        zorder=22, clip_on=False)

# Ticks formatting
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)

x_ticks = np.arange(fixed_x_min, fixed_x_max + 1e-9, 0.2)
y_ticks = np.arange(fixed_y_min, fixed_y_max + 1e-9, 0.1)
y_labels = [f"{y:.1f}" for y in y_ticks]
if len(y_labels) > 0: y_labels[0] = ''

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{x:.1f}" for x in x_ticks], fontsize=TICK_FS)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=TICK_FS)
ax.tick_params(axis='both', which='major', labelsize=TICK_FS, direction='in')

# Legend Assembly
handles = []
labels = []

for city in CITY_ORDER:
    color = CITY_PALETTE[CITY_ORDER.index(city)]
    name = CITY_NAMES[city]
    handles.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markeredgecolor='#333333', markersize=20))
    labels.append(name)

handles.append(AreaMarker())
labels.append('Area')

ax.legend(handles, labels,
          handler_map={AreaMarker: AreaHandler()},
          loc='center left', bbox_to_anchor=(1, 0.7),
          frameon=False, fontsize=20, labelspacing=1.0,
          handlelength=2)

# Save figure
out_path = os.path.join(OUTPUT_DIR, "Result_Fig.3c_scatter.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Chart successfully saved to: {out_path}")