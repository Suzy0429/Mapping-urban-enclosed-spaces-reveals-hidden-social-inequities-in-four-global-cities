import pandas as pd

def calculate_confusion_matrix(real, pred):
    TP = ((real == 1) & (pred == 1)).sum()
    TN = ((real == 0) & (pred == 0)).sum()
    FP = ((real == 0) & (pred == 1)).sum()
    FN = ((real == 1) & (pred == 0)).sum()
    return TP, FP, FN, TN

def calculate_metrics(TP, FP, FN, TN):
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def main():
    input_file = r"C:\github\Data\Test_model.xlsx"
    output_file = r"C:\github\Data\Result_Confusion_Matrix.csv"

    df = pd.read_excel(input_file)

    models = ["Mapillary", "COCO", "Cityscapes", "ADE20K"]

    results = []
    for model in models:
        TP, FP, FN, TN = calculate_confusion_matrix(df["real"], df[model])
        accuracy, precision, recall, f1 = calculate_metrics(TP, FP, FN, TN)
        results.append({
            "Model": model,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4)
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Metrics calculation complete. Saved to: {output_file}")


if __name__ == "__main__":
    main()