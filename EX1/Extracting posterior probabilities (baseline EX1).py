import pandas as pd
import os
import glob

MODEL = "LApredict"
DATASET = "os" # or qt
confidence = 0.95  # set (1-alpha) threshold

#folder_path = "./LApredict_" + dataset
folder_path = f'{MODEL}/{DATASET}/test_set'
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

results = []

# Totals across all files
totals = {
    "Tot. Correct Predictions": 0,
    "Tot. Correct Clean Predictions": 0,
    "Tot. Correct Fault-prone Predictions": 0,
    "Nr. flagged predictions": 0,
    "Nr. flagged Clean": 0,
    "Nr. flagged Fault-prone": 0,
    "Nr. correctly flagged predictions": 0,
    "Nr. correctly flagged Clean": 0,
    "Nr. correctly flagged Fault-prone": 0,
    "Total Rows": 0
}

for file_path in excel_files:
    df = pd.read_excel(file_path)

    total_rows = len(df)

    # Correct predictions
    correct_all = (df["Predicted label"] == df["True label"])
    correct_clean = correct_all & (df["True label"] == 0)
    correct_fault = correct_all & (df["True label"] == 1)

    # Flagged predictions
    flagged_all = (df["Predicted prob"] >= confidence) | (df["Predicted prob"] <= 1 - confidence)
    flagged_clean = flagged_all & (df["Predicted label"] == 0)
    flagged_fault = flagged_all & (df["Predicted label"] == 1)

    # Correctly flagged predictions
    corr_flagged_all = flagged_all & correct_all
    corr_flagged_clean = flagged_clean & correct_clean
    corr_flagged_fault = flagged_fault & correct_fault

    # Precision values (safe division)
    precision_all = corr_flagged_all.sum() / flagged_all.sum() if flagged_all.sum() > 0 else 0
    precision_clean = corr_flagged_clean.sum() / flagged_clean.sum() if flagged_clean.sum() > 0 else 0
    precision_fault = corr_flagged_fault.sum() / flagged_fault.sum() if flagged_fault.sum() > 0 else 0

    # Store per-file results
    results.append({
        "File": os.path.basename(file_path),
        "Total Rows": total_rows,
        "Tot. Correct Predictions": correct_all.sum(),
        "Tot. Correct Clean Predictions": correct_clean.sum(),
        "Tot. Correct Fault-prone Predictions": correct_fault.sum(),
        "Nr. flagged predictions": flagged_all.sum(),
        "Nr. flagged Clean": flagged_clean.sum(),
        "Nr. flagged Fault-prone": flagged_fault.sum(),
        "Nr. correctly flagged predictions": corr_flagged_all.sum(),
        "Nr. correctly flagged Clean": corr_flagged_clean.sum(),
        "Nr. correctly flagged Fault-prone": corr_flagged_fault.sum(),
        "Precision": precision_all,
        "Precision-clean": precision_clean,
        "Precision-Fault-prone": precision_fault
    })

    # Update totals
    totals["Total Rows"] += total_rows
    totals["Tot. Correct Predictions"] += correct_all.sum()
    totals["Tot. Correct Clean Predictions"] += correct_clean.sum()
    totals["Tot. Correct Fault-prone Predictions"] += correct_fault.sum()
    totals["Nr. flagged predictions"] += flagged_all.sum()
    totals["Nr. flagged Clean"] += flagged_clean.sum()
    totals["Nr. flagged Fault-prone"] += flagged_fault.sum()
    totals["Nr. correctly flagged predictions"] += corr_flagged_all.sum()
    totals["Nr. correctly flagged Clean"] += corr_flagged_clean.sum()
    totals["Nr. correctly flagged Fault-prone"] += corr_flagged_fault.sum()

results.append({
    "File": "TOTAL",
    **totals,
    "Precision": totals["Nr. correctly flagged predictions"] / totals["Nr. flagged predictions"] if totals["Nr. flagged predictions"] > 0 else 0,
    "Precision-clean": totals["Nr. correctly flagged Clean"] / totals["Nr. flagged Clean"] if totals["Nr. flagged Clean"] > 0 else 0,
    "Precision-Fault-prone": totals["Nr. correctly flagged Fault-prone"] / totals["Nr. flagged Fault-prone"] if totals["Nr. flagged Fault-prone"] > 0 else 0
})

summary_df = pd.DataFrame(results)
output_file = os.path.join(MODEL +"_"+ DATASET + "_baseline_effectiveness_results_"+ str(confidence) +".xlsx")
summary_df.to_excel(output_file, index=False)

print(f"Detailed results saved to {output_file}")