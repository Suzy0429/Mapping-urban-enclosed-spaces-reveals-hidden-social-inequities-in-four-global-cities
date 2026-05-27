import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.utils import resample
import matplotlib as mpl

try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('default')
    sns.set_style("whitegrid")

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False

plt.rc('font', family='Arial')
plt.rc('axes', titlesize=12, labelsize=10, titleweight='bold')
plt.rc('legend', fontsize=10)

# 1. Load and preprocess inference data
try:
    file_path = r"C:\github\Data\UEthreshold_inference.csv"
    data = pd.read_csv(file_path)
    enclosure_data = data['a'][(data['a'] > 0) & (data['a'] < 1)].copy()
    print(f"Valid data count: {len(enclosure_data)}")
except Exception as e:
    print(f"Data loading error: {e}")
    exit()

# 2. Calculate optimal threshold
try:
    train_df = pd.read_csv(r"C:\github\Data\UEthreshold_train.csv")
    a_values = train_df["FBW"].values
    y_values = train_df["ue"].values

    thresholds = np.unique((a_values[:-1] + a_values[1:]) / 2)
    best_acc = 0
    final_threshold = None
    for t in thresholds:
        pred = (a_values >= t).astype(int)
        acc = (pred == y_values).mean()
        if acc > best_acc:
            best_acc = acc
            final_threshold = t

    print(f"Optimal threshold: {final_threshold:.4f} (Accuracy: {best_acc:.2%})")
except Exception as e:
    print(f"Threshold calculation error: {e}")
    exit()

# 3. Threshold stability test (Bootstrap)
try:
    thresholds = []
    for _ in range(500):
        sample_idx = resample(np.arange(len(a_values)), replace=True, n_samples=len(a_values))
        sample_a = a_values[sample_idx]
        sample_y = y_values[sample_idx]

        sample_thresholds = np.unique((sample_a[:-1] + sample_a[1:]) / 2)
        best_sample_acc = 0
        best_sample_threshold = None
        for t in sample_thresholds:
            pred = (sample_a >= t).astype(int)
            acc = (pred == sample_y).mean()
            if acc > best_sample_acc:
                best_sample_acc = acc
                best_sample_threshold = t

        if best_sample_threshold is not None:
            thresholds.append(best_sample_threshold)

    ci_lower = np.percentile(thresholds, 2.5)
    ci_upper = np.percentile(thresholds, 97.5)
    print(f"Threshold 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
except Exception as e:
    print(f"Stability test error: {e}")
    ci_lower = final_threshold - 0.02
    ci_upper = final_threshold + 0.02

# 4. Visualization
kde = gaussian_kde(enclosure_data)
x_grid = np.linspace(max(0.01, enclosure_data.min()),
                     min(0.99, enclosure_data.max()),
                     1000)
density = kde(x_grid)

plt.figure(figsize=(7, 3.5))
plt.tick_params(axis='both', which='both', direction='out',
                length=6, width=1, colors='black',
                bottom=True, top=False, left=True, right=False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

ax = sns.histplot(enclosure_data, bins=30, kde=False,
                  color='lightgray', edgecolor='black',
                  linewidth=0.5, alpha=0.7, stat='density', label='Data distribution')

plt.plot(x_grid, density, 'b-', linewidth=2.5, label='Kernel Density Estimation (KDE)')

plt.axvline(final_threshold, color='red', linestyle='--',
            linewidth=2.5, label=f'Threshold ({final_threshold:.3f})')

plt.xlabel('Parcel Closure Degree', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

output_img = r"C:\github\Data\Result_UEthreshold.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"\nAnalysis chart saved to: {output_img}")
plt.show()