import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

CITY_NAMES = {
    'beijing': 'Beijing',
    'london': 'London',
    'newyork': 'New York',
    'mumbai': 'Mumbai'
}

def plot_distribution_combined():
    input_file = r"C:\github\Data\data.csv"
    output_dir = r"C:\github\Data"
    output_file = os.path.join(output_dir, "Result_Fig.2b_distribution.png")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found {input_file}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_file)
    df['city'] = df['city'].astype(str).str.strip().str.lower()
    df['ue'] = pd.to_numeric(df['ue'], errors='coerce')

    x_min, x_max = 0.0, 0.35
    y_max = 45
    x_axis = np.linspace(x_min, x_max, 300)

    color_enc = '#B52C34'
    color_open = '#1E4681'
    label_fs = 20
    tick_fs = 18
    annot_fs = 14

    target_cities = ['beijing', 'london', 'newyork', 'mumbai']
    valid_cities = [c for c in target_cities if c in df['city'].unique()]

    fig = plt.figure(figsize=(14, 8))
    outer_gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.2)

    for idx, city in enumerate(valid_cities):
        row = idx // 2
        col = idx % 2

        city_data = df[df['city'] == city]

        data_all = city_data['D'].dropna()
        data_enc_all = city_data[city_data['ue'] == 1]['D'].dropna()
        data_open_all = city_data[city_data['ue'] == 0]['D'].dropna()

        if len(data_enc_all) < 2 or len(data_open_all) < 2:
            print(f"Skipping {city} (insufficient samples)")
            continue

        data_min = max(0.0, float(data_all.min()))
        data_max = float(data_all.max())
        if data_max <= data_min:
            data_max = data_min + 1e-6

        nbins = 100
        bins_hist = np.linspace(data_min, data_max, nbins + 1)

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[row, col],
            height_ratios=[4, 2],
            hspace=0
        )

        ax_top = fig.add_subplot(inner_gs[0])
        ax_bottom = fig.add_subplot(inner_gs[1], sharex=ax_top)

        ax_top.hist(
            data_enc_all, bins=bins_hist, density=True, alpha=0.3,
            color=color_enc, edgecolor='gray', linewidth=0.6
        )
        ax_top.hist(
            data_open_all, bins=bins_hist, density=True, alpha=0.3,
            color=color_open, edgecolor='gray', linewidth=0.6
        )

        kde_enc = gaussian_kde(data_enc_all)
        kde_open = gaussian_kde(data_open_all)
        ax_top.plot(x_axis, kde_enc(x_axis), color=color_enc, linewidth=3.5)
        ax_top.plot(x_axis, kde_open(x_axis), color=color_open, linewidth=3.5)

        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(0, y_max)

        plot_data = [data_open_all, data_enc_all]
        flier_props = dict(
            marker='o', markerfacecolor='lightgray',
            markeredgecolor='lightgray', markersize=5
        )

        bp = ax_bottom.boxplot(
            plot_data, vert=False, patch_artist=False,
            widths=0.5, positions=[1, 2],
            flierprops=flier_props
        )

        colors_map = [color_open, color_enc]
        for i, box in enumerate(bp['boxes']):
            plt.setp(box, color=colors_map[i], linewidth=2)
        for i, whisker in enumerate(bp['whiskers']):
            plt.setp(whisker, color=colors_map[i // 2], linewidth=2)
        for i, cap in enumerate(bp['caps']):
            plt.setp(cap, color=colors_map[i // 2], linewidth=2)

        plt.setp(bp['medians'][0], color=color_open, linewidth=4)
        plt.setp(bp['medians'][1], color=color_enc, linewidth=4)

        ax_bottom.set_ylim(0.5, 2.5)
        ax_bottom.set_xlim(x_min, x_max)

        median_open = np.median(data_open_all)
        median_enc = np.median(data_enc_all)

        for a in [ax_top, ax_bottom]:
            a.vlines(median_open, *a.get_ylim(), colors=color_open,
                     linestyles='--', linewidth=2, alpha=0.8)
            a.vlines(median_enc, *a.get_ylim(), colors=color_enc,
                     linestyles='--', linewidth=2, alpha=0.8)

        x_offset = (x_max - x_min) * 0.01
        ny_enc_extra_shift = (x_max - x_min) * 0.01

        if city == 'newyork':
            open_x_init, open_ha_init = median_open + x_offset, 'left'
            enc_x_init, enc_ha_init = median_enc + x_offset - ny_enc_extra_shift, 'left'
        else:
            if median_open > median_enc:
                open_x_init, open_ha_init = median_open + x_offset, 'left'
                enc_x_init, enc_ha_init = median_enc - x_offset, 'right'
            elif median_open < median_enc:
                enc_x_init, enc_ha_init = median_enc + x_offset, 'left'
                open_x_init, open_ha_init = median_open - x_offset, 'right'
            else:
                open_x_init, open_ha_init = median_open - x_offset, 'right'
                enc_x_init, enc_ha_init = median_enc + x_offset, 'left'

        def enforce_bounds(x_val, ha):
            min_x = x_min + (x_max - x_min) * 0.005
            max_x = x_max - (x_max - x_min) * 0.005
            if x_val < min_x: return min_x, 'left'
            if x_val > max_x: return max_x, 'right'
            return x_val, ha

        open_x, open_ha = enforce_bounds(open_x_init, open_ha_init)
        enc_x, enc_ha = enforce_bounds(enc_x_init, enc_ha_init)

        min_sep = (x_max - x_min) * 0.02
        if abs(open_x - enc_x) < min_sep:
            open_x_new = max(x_min + (x_max - x_min) * 0.005, open_x - min_sep / 2)
            enc_x_new = min(x_max - (x_max - x_min) * 0.005, enc_x + min_sep / 2)
            open_x, _ = enforce_bounds(open_x_new, open_ha)
            enc_x, _ = enforce_bounds(enc_x_new, enc_ha)

        ax_top.text(open_x, y_max, f"{median_open:.3f}",
                    color=color_open, fontsize=annot_fs, ha=open_ha, va='top', transform=ax_top.transData)
        ax_top.text(enc_x, y_max, f"{median_enc:.3f}",
                    color=color_enc, fontsize=annot_fs, ha=enc_ha, va='top', transform=ax_top.transData)

        custom_ticks = np.arange(0.0, 0.36, 0.1)
        ax_bottom.set_xticks(custom_ticks)
        ax_bottom.set_xticklabels([f"{x:.1f}" for x in custom_ticks], fontsize=tick_fs)

        for a in [ax_top, ax_bottom]:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['left'].set_linewidth(1.5)
            a.spines['bottom'].set_linewidth(1.5)

        ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_top.tick_params(axis='y', labelsize=tick_fs, width=1.5)

        ax_bottom.set_yticks([1, 2])
        ax_bottom.set_yticklabels(['UOS', 'UES'], fontsize=tick_fs)
        ax_bottom.tick_params(axis='x', direction='out', width=1.5, length=5)
        ax_bottom.tick_params(axis='y', width=1.5)

        display_city = CITY_NAMES.get(city, city.title())
        ax_bottom.set_xlabel(f'Φ ({display_city})', fontsize=label_fs)
        ax_top.set_ylabel('Density', fontsize=label_fs)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All charts rendered successfully: {output_file}")


if __name__ == "__main__":
    plot_distribution_combined()