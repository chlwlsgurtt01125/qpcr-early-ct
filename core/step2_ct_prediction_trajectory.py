# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/cphotonic/qpcr_v2")
SUMMARY = ROOT / "data" / "metrics" / "step2_pred_ct_trajectory_summary.csv"
FIG_DIR = ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(SUMMARY)

    print("[INFO] Loaded columns:")
    print(df.columns.tolist())

    # detect columns safely
    col = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "cutoff":
            col["cutoff"] = c
        elif "mae" in lc:
            col["mae"] = c
        elif "med" in lc:
            col["median"] = c
        elif "p90" in lc or "90" in lc:
            col["p90"] = c
        elif "0.5" in lc:
            col["le_05"] = c
        elif "1.0" in lc:
            col["le_10"] = c

    if "cutoff" not in col or "mae" not in col:
        raise RuntimeError("Required columns not found")

    # ---------------------------
    # Figure 1: error vs cycle
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(df[col["cutoff"]], df[col["mae"]], marker="o", label="MAE")

    if "median" in col:
        plt.plot(df[col["cutoff"]], df[col["median"]], marker="s", label="Median error")

    if "p90" in col:
        plt.plot(df[col["cutoff"]], df[col["p90"]], linestyle="--", label="90 percentile")

    plt.xlabel("Cycle cutoff")
    plt.ylabel("Absolute Ct error")
    plt.title("Ct prediction error vs cycle")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    f1 = FIG_DIR / "ct_error_vs_cycle.png"
    plt.savefig(f1, dpi=300)
    plt.show()
    print(f"[SAVED] {f1}")

    # ---------------------------
    # Figure 2: accuracy fraction
    # ---------------------------
    if "le_05" in col and "le_10" in col:
        plt.figure(figsize=(7, 5))
        plt.plot(
            df[col["cutoff"]],
            df[col["le_05"]] * 100,
            marker="o",
            label="abs error <= 0.5",
        )
        plt.plot(
            df[col["cutoff"]],
            df[col["le_10"]] * 100,
            marker="s",
            label="abs error <= 1.0",
        )

        plt.xlabel("Cycle cutoff")
        plt.ylabel("Samples (%)")
        plt.title("Fraction of accurate Ct prediction")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        f2 = FIG_DIR / "ct_accuracy_fraction_vs_cycle.png"
        plt.savefig(f2, dpi=300)
        plt.show()
        print(f"[SAVED] {f2}")

    print("[DONE] Trajectory analysis finished.")


if __name__ == "__main__":
    main()
