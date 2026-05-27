import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# 1. Global Configuration
INPUT_DATA_FILE = r"C:\github\Data\data.csv"
INTERMEDIATE_FILE = r"C:\github\Data\Result_rate.csv"
OUTPUT_PLOT_FILE = r"C:\github\Data\Result_Fig.1c_share.png"

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

colors_rgb = {
    'Residential': (207 / 255, 66 / 255, 78 / 255),
    'Institutional': (255 / 255, 232 / 255, 187 / 255),
    'Commercial': (48 / 255, 130 / 255, 173 / 255),
    'Industrial': (126 / 255, 190 / 255, 214 / 255),
    'Park': (189 / 255, 208 / 255, 197 / 255)
}
BG_COLOR_LIGHT = '#F5F5F5'

city_order = ['beijing', 'london', 'newyork', 'mumbai']
city_display_names = {
    'beijing': 'Beijing',
    'london': 'London',
    'newyork': 'New York',
    'mumbai': 'Mumbai'
}
type_order = ['Residential', 'Institutional', 'Commercial', 'Industrial', 'Park']
type_order_plot = type_order[::-1]  # Plot order (top to bottom)

BAR_HEIGHT = 0.5
LABEL_OFFSET_Y = 0.08
FONT_SIZE_VAL = 15
FONT_SIZE_CITY = 22
LABEL_GAP = 0.005
Y_LIM_BOTTOM = -1
Y_LIM_TOP = 4.9

# 2. Data Processing
def calculate_rates():
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"Error: Input file not found {INPUT_DATA_FILE}")
        return None

    try:
        print("Reading data and calculating metrics...")
        df = pd.read_csv(INPUT_DATA_FILE)

        df['area_sq'] = df['area'] ** 2

        group_cols = ['city', 'ue', 'type']
        df_subgroup = df.groupby(group_cols).agg({
            'area': 'sum',
            'area_sq': 'sum',
            'city': 'size'
        }).rename(columns={'city': 'count'}).reset_index()

        total_cols = ['city']
        df_city_total = df.groupby(total_cols).agg({
            'area': 'sum',
            'area_sq': 'sum',
            'city': 'size'
        }).rename(columns={
            'area': 'total_city_area',
            'area_sq': 'total_city_area_sq',
            'city': 'total_city_count'
        }).reset_index()

        result = pd.merge(df_subgroup, df_city_total, on=['city'], how='left')

        result['area_rate'] = result['area'] / result['total_city_area']

        p_area = result['area_rate']
        numerator_area = (1 - 2 * p_area) * result['area_sq'] + (p_area ** 2) * result['total_city_area_sq']
        numerator_area = np.maximum(0, numerator_area)  # Fix negative values
        result['area_rate_se'] = np.sqrt(numerator_area) / result['total_city_area']

        result['count_rate'] = result['count'] / result['total_city_count']

        p_count = result['count_rate']
        n_total = result['total_city_count']
        result['count_rate_se'] = np.sqrt((p_count * (1 - p_count)) / n_total)

        output_cols = [
            'city', 'ue', 'type',
            'area', 'area_rate', 'area_rate_se',
            'count', 'count_rate', 'count_rate_se'
        ]
        final_df = result[output_cols]

        output_dir = os.path.dirname(INTERMEDIATE_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_df.to_csv(INTERMEDIATE_FILE, index=False, encoding='utf-8-sig')

        print(f"Data processing completed! Results saved to: {INTERMEDIATE_FILE}")
        return final_df

    except Exception as e:
        print(f"Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return None


# 3. Visualization
def plot_results(df):
    print("Generating visualization...")

    numeric_cols = ['area_rate', 'area_rate_se', 'count_rate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, city in enumerate(city_order):
        ax = axes[idx]

        if idx in [1, 3]:
            ax.set_facecolor(BG_COLOR_LIGHT)
        else:
            ax.set_facecolor('white')

        ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#C9C9C9', zorder=0, alpha=0.5)
        ax.set_axisbelow(True)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', zorder=1)

        if idx == 0:
            ax.text(0.03, 0.95, "UOS", transform=ax.transAxes,
                    ha='left', va='top', fontsize=18, fontweight='bold', color='#333333', zorder=10)
            ax.text(0.97, 0.95, "UES", transform=ax.transAxes,
                    ha='right', va='top', fontsize=18, fontweight='bold', color='#333333', zorder=10)

        city_data = df[df['city'] == city].copy()

        for y_i, type_name in enumerate(type_order_plot):
            row_open = city_data[(city_data['type'] == type_name) & (city_data['ue'] == 0)]
            row_enclosed = city_data[(city_data['type'] == type_name) & (city_data['ue'] == 1)]
            color = colors_rgb.get(type_name, (0.5, 0.5, 0.5))

            if not row_open.empty:
                val = row_open['area_rate'].values[0]
                err = row_open['area_rate_se'].values[0]
                pop = row_open['count_rate'].values[0]

                ax.barh(y_i, -val, color=color, alpha=0.6, height=BAR_HEIGHT,
                        xerr=err, capsize=4, error_kw={'ecolor': 'black', 'alpha': 1, 'linewidth': 1}, zorder=2)

                ax.scatter(-pop, y_i, color=color, s=80, alpha=1.0,
                           edgecolors='black', linewidths=1.2, zorder=3)

                if val > 0.001:
                    if val > 0.5:
                        lbl_x = -val + LABEL_GAP
                        lbl_ha = 'left'
                    else:
                        lbl_x = -val - LABEL_GAP
                        lbl_ha = 'right'

                    ax.text(lbl_x, y_i - BAR_HEIGHT / 2 - LABEL_OFFSET_Y,
                            f"{val:.1%}", ha=lbl_ha, va='top',
                            fontsize=FONT_SIZE_VAL, color='#555555', fontfamily='Arial')

            if not row_enclosed.empty:
                val = row_enclosed['area_rate'].values[0]
                err = row_enclosed['area_rate_se'].values[0]
                pop = row_enclosed['count_rate'].values[0]

                ax.barh(y_i, val, color=color, alpha=0.6, height=BAR_HEIGHT,
                        xerr=err, capsize=4, error_kw={'ecolor': 'black', 'alpha': 1, 'linewidth': 1}, zorder=2)

                ax.scatter(pop, y_i, color=color, s=100, alpha=1.0,
                           edgecolors='black', linewidths=1.2, zorder=3)

                if val > 0.001:
                    ax.text(val + LABEL_GAP, y_i - BAR_HEIGHT / 2 - LABEL_OFFSET_Y,
                            f"{val:.1%}", ha='left', va='top',
                            fontsize=FONT_SIZE_VAL, color='#555555', fontfamily='Arial')

        display_name = city_display_names.get(city, city)

        ax.text(-0.02, 0.5, display_name, transform=ax.transAxes,
                ha='right', va='center', rotation=90,
                fontsize=FONT_SIZE_CITY, fontfamily='Arial', color='#333333')

        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='y', which='both', left=False, right=False)

        if idx == 0:
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, top=True)
        elif idx == len(city_order) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.tick_params(bottom=True, top=False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, top=False)

    bottom_ax = axes[-1]
    leg_x_rect = 0.22
    leg_x_circ = 0.33
    leg_x_text = 0.38
    leg_y_start = 2.8
    leg_y_step = 0.8

    bottom_ax.text(leg_x_rect, leg_y_start + 0.35, "Area", ha='center', va='bottom', fontsize=16,
                   fontweight='bold', color='#333333')
    bottom_ax.text(leg_x_circ + 0.05, leg_y_start + 0.35, " Count", ha='center', va='bottom', fontsize=16,
                   fontweight='bold', color='#333333')

    for i, type_name in enumerate(type_order_plot[::-1]):
        y_pos = leg_y_start - i * leg_y_step
        color = colors_rgb[type_name]

        rect_width = 0.08
        rect_height = 0.2
        rect = mpatches.Rectangle((leg_x_rect - rect_width / 2, y_pos - rect_height / 2),
                                  rect_width, rect_height,
                                  linewidth=1, facecolor=color, alpha=0.6, zorder=10, clip_on=False)
        bottom_ax.add_patch(rect)

        bottom_ax.scatter(leg_x_circ, y_pos, s=100, color=color, alpha=1.0,
                          edgecolors='black', linewidths=1, zorder=10, clip_on=False)

        bottom_ax.text(leg_x_text, y_pos, type_name, ha='left', va='center',
                       fontsize=18, color='#333333', zorder=10)

    axes[0].set_ylim(Y_LIM_BOTTOM, Y_LIM_TOP)
    axes[0].set_yticks(np.arange(len(type_order_plot)))
    axes[0].set_yticklabels([])

    def percent_abs_formatter(x, pos):
        if x == 0: return ""
        return f"{abs(x):.0%}"

    axes[0].set_xlim(-0.7, 0.7)
    custom_ticks = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    axes[0].set_xticks(custom_ticks)

    axes[0].xaxis.set_ticks_position('top')
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(percent_abs_formatter))
    axes[0].tick_params(axis='x', labelsize=18)

    axes[-1].set_xticks(custom_ticks)
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(percent_abs_formatter))
    axes[-1].tick_params(axis='x', labelsize=18)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.03)

    output_dir = os.path.dirname(OUTPUT_PLOT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(OUTPUT_PLOT_FILE, dpi=600)
    print(f"Chart successfully saved to: {OUTPUT_PLOT_FILE}")
    print("-" * 30)

# 4. Main Entry
if __name__ == "__main__":
    df_results = calculate_rates()

    if df_results is not None:
        plot_results(df_results)