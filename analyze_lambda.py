import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

lambdas = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

results = []

print(f"{'Lambda TA':<10} | {'Fake Cell Score':<20} | {'Max Legit Score':<20} | {'Max Score Gap':<15} | {'Mean Err(m)':<15} | {'Med Err(m)':<15} | {'Min Err(m)':<15} | {'Max Err(m)':<15}")
print("-" * 145)

for l in lambdas:
    # Run detector
    cmd = ["python", "detector/sliding_window_detector.py", "--results-dir", "results", "--lambda-ta", str(l)]
    
    # We suppress output to keep the console clean
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Read detection results
    det_df = pd.read_csv("results/detection_results.csv")

    # Fake cell score
    fake_scores = det_df[det_df["is_fake_gt"] == 1]["score"]
    avg_fake_score = fake_scores.mean() if not fake_scores.empty else float('nan')
    
    # Legit cells: min, max và max score gap
    legit_scores = det_df[det_df["is_fake_gt"] == 0]["score"]
    max_legit_score = legit_scores.max() if not legit_scores.empty else float('nan')
    
    # Score gap: fake score - max legit score (càng cao càng dễ phân biệt)
    score_gap = avg_fake_score - max_legit_score if pd.notna(avg_fake_score) and pd.notna(max_legit_score) else float('nan')

    # Read UE errors
    err_df = pd.read_csv("results/ue_position_errors.csv")
    mean_err = err_df["error_m"].mean()   if not err_df.empty else float('nan')
    med_err  = err_df["error_m"].median() if not err_df.empty else float('nan')
    min_err  = err_df["error_m"].min()    if not err_df.empty else float('nan')
    max_err  = err_df["error_m"].max()    if not err_df.empty else float('nan')
    
    print(f"{l:<10.4f} | {avg_fake_score:<20.4f} | {max_legit_score:<20.4f} | {score_gap:<15.4f} | {mean_err:<15.2f} | {med_err:<15.2f} | {min_err:<15.2f} | {max_err:<15.2f}")

print("\nHoàn tất phân tích.")
