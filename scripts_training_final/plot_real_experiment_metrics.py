from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Input CSV
INPUT_CSV = Path("real_experiment_metrics.csv")

# Output plots
LOSS_PLOT = Path("real_experiment_loss_plot.png")
BPC_PLOT = Path("real_experiment_bpc_plot.png")

def main():
    # 1. Read CSV
    df = pd.read_csv(INPUT_CSV)

    print("=== Plot real experiment metrics ===")
    print("Input CSV:", INPUT_CSV)
    print("Number of rows:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    # 2. Convert columns to numeric
    numeric_cols = [
        "step",
        "train_loss",
        "train_bpc",
        "avg_val_loss",
        "avg_val_bpc",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Preview
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    # 4. Validation subset
    val_df = df.dropna(subset=["avg_val_loss", "avg_val_bpc"]).copy()

    print("\nValidation rows:")
    print(val_df[["step", "avg_val_loss", "avg_val_bpc"]].to_string(index=False))

    # 5. Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["train_loss"], marker="o", linewidth=1, label="train loss")
    plt.plot(val_df["step"], val_df["avg_val_loss"], marker="o", linewidth=1, label="avg val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("First real experiment: training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.close()

    # 6. Plot BPC
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["train_bpc"], marker="o", linewidth=1, label="train BPC")
    plt.plot(val_df["step"], val_df["avg_val_bpc"], marker="o", linewidth=1, label="avg val BPC")
    plt.xlabel("Step")
    plt.ylabel("BPC")
    plt.title("First real experiment: training and validation BPC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BPC_PLOT, dpi=150)
    plt.close()

    print("\nPlots saved successfully.")
    print("Loss plot:", LOSS_PLOT)
    print("BPC plot:", BPC_PLOT)

if __name__ == "__main__":
    main()