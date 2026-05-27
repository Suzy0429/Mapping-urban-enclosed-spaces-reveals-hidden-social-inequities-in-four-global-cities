import pandas as pd
import os

INPUT_CSV = r"C:\github\Data\Test_sampling.csv"
OUTPUT_CSV = r"C:\github\Data\Result_Confusion_Matrix_city.csv"


def calculate_confusion(real, pred):
    TP = int(((real == 1) & (pred == 1)).sum())
    TN = int(((real == 0) & (pred == 0)).sum())
    FP = int(((real == 0) & (pred == 1)).sum())
    FN = int(((real == 1) & (pred == 0)).sum())
    return TP, FP, FN, TN


def metrics_from_counts(TP, FP, FN, TN):
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Check required columns
    required_cols = {"city", "id", "status", "result"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Input file missing required columns: {missing}")

    df['city'] = df['city'].astype(str).str.strip().str.lower()

    df['status_num'] = pd.to_numeric(df['status'], errors='coerce')
    df['result_num'] = pd.to_numeric(df['result'], errors='coerce')

    # Drop rows where status/result are not 0 or 1
    valid_mask = df['status_num'].isin([0, 1]) & df['result_num'].isin([0, 1])
    dropped_count = (~valid_mask).sum()
    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} rows with invalid or missing status/result.")

    df = df.loc[valid_mask].copy()
    df['status_num'] = df['status_num'].astype(int)
    df['result_num'] = df['result_num'].astype(int)

    # Target city order priority
    target_cities = ['beijing', 'london', 'newyork', 'mumbai']
    present = [c for c in target_cities if c in df['city'].unique()]
    extras = sorted([c for c in df['city'].unique() if c not in target_cities])
    cities_to_process = present + extras

    results = []
    for city in cities_to_process:
        sub = df[df['city'] == city]
        TP, FP, FN, TN = calculate_confusion(sub['status_num'], sub['result_num'])
        accuracy, precision, recall, f1 = metrics_from_counts(TP, FP, FN, TN)
        results.append({
            "city": city,
            "n_samples": int(TP + FP + FN + TN),
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            # 将这里的 4 修改为 3
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-score": round(f1, 3)
        })

    # Overall metrics
    TP_all, FP_all, FN_all, TN_all = calculate_confusion(df['status_num'], df['result_num'])
    acc_all, prec_all, rec_all, f1_all = metrics_from_counts(TP_all, FP_all, FN_all, TN_all)
    all_row = {
        "city": "all",
        "n_samples": int(TP_all + FP_all + FN_all + TN_all),
        "TP": TP_all,
        "FP": FP_all,
        "FN": FN_all,
        "TN": TN_all,
        # 将这里的 4 修改为 3
        "Accuracy": round(acc_all, 3),
        "Precision": round(prec_all, 3),
        "Recall": round(rec_all, 3),
        "F1-score": round(f1_all, 3)
    }

    result_df = pd.DataFrame([all_row] + results, columns=[
        "city", "n_samples", "TP", "FP", "FN", "TN", "Accuracy", "Precision", "Recall", "F1-score"
    ])

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Confusion matrix and metrics saved to: {OUTPUT_CSV}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()