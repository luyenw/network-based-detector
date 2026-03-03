import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

_RESULTS = Path(__file__).parent.parent / "results"

def dec_map_mixture_model(target_ecgi='001-01-0000001'):
    # 1. Load Data
    cell_db = pd.read_csv(_RESULTS / 'cell_database.csv')
    loc_input = pd.read_csv(_RESULTS / 'localization_input.csv')
    
    fbs_gt = cell_db[cell_db['is_fake'] == 1].iloc[0]
    fbs_x, fbs_y = fbs_gt['pos_x'], fbs_gt['pos_y']
    
    lbs_gt = cell_db[(cell_db['is_fake'] == 0) & (cell_db['ecgi'] == target_ecgi)].iloc[0]
    lbs_x, lbs_y = lbs_gt['pos_x'], lbs_gt['pos_y']
    
    print(f"[LOAD] FBS ground truth : ({fbs_x:.1f}, {fbs_y:.1f})")
    print(f"[LOAD] LBS position     : ({lbs_x:.1f}, {lbs_y:.1f})")
    print(f"[LOAD] localization_input.csv : {len(loc_input)} total rows")
    
    # Filter: chỉ giữ neighbour reports (loại bỏ serving cell)
    # và chỉ lấy report cho target ECGI
    df_target = loc_input[
        (loc_input['ecgi'] == target_ecgi) &
        (loc_input['cell_role'] != 'S')
    ].copy()
    df_target = df_target.sort_values('time_sec').reset_index(drop=True)
    
    print(f"[FILTER] After filter (ecgi={target_ecgi}, cell_role!=S): {len(df_target)} samples")
    if len(df_target) == 0:
        print("[ERROR] No samples after filtering! Check localization_input.csv.")
        return
    print(f"[FILTER] RSRP range: [{df_target['rsrp_dbm'].min():.1f}, {df_target['rsrp_dbm'].max():.1f}] dBm")
    print(f"[FILTER] Time range: [{df_target['time_sec'].min():.1f}, {df_target['time_sec'].max():.1f}] s")
    
    # 2. Khởi tạo Không gian và Tham số COST-231
    grid_res = 10
    x_coords = np.arange(-500, 2500 + grid_res, grid_res)
    y_coords = np.arange(-500, 2500 + grid_res, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    prior = np.ones_like(X, dtype=float)
    prior /= np.sum(prior)
    
    THEO_N = 3.76
    THEO_INTERCEPT = 30.7  # dBm
    THEO_SIGMA = 8.0       # dB
    PI = 0.5               # Xác suất hỗn hợp (50% FBS / 50% LBS)
    
    print(f"\n[PARAMS] n={THEO_N}, intercept={THEO_INTERCEPT} dBm, σ={THEO_SIGMA} dB, π={PI}")
    print(f"[GRID]   x=[{x_coords[0]}, {x_coords[-1]}], y=[{y_coords[0]}, {y_coords[-1]}], res={grid_res}m, cells={X.shape}")
    
    rmses = []
    cumulative_samples = []   # tổng samples tích lũy sau mỗi batch
    n_skipped = 0
    total_samples_so_far = 0
    
    # 3. Chạy Bayesian Update theo BATCH (mỗi batch = 1 time window)
    batches = df_target.groupby('time_sec')
    n_batches = len(batches)
    print(f"\n[RUN] Starting BATCH Bayesian update: {n_batches} batches, {len(df_target)} total samples...")
    
    for batch_idx, (t_sec, batch) in enumerate(batches):
        # Tích lũy log-likelihood cho tất cả samples trong batch
        batch_log_likelihood = np.zeros_like(X, dtype=float)
        
        for _, row in batch.iterrows():
            ue_x, ue_y = row['estimated_ue_x'], row['estimated_ue_y']
            rsrp_obs = row['rsrp_dbm']
            
            # Thành phần 1: Kỳ vọng từ LBS thật (Scalar)
            d_lbs = np.sqrt((ue_x - lbs_x)**2 + (ue_y - lbs_y)**2) + 1.0
            rsrp_exp_lbs = THEO_INTERCEPT - 10 * THEO_N * np.log10(d_lbs)
            prob_lbs = norm.pdf(rsrp_obs, loc=rsrp_exp_lbs, scale=THEO_SIGMA)
            
            # Thành phần 2: Kỳ vọng từ FBS giả (Matrix)
            d_fbs_grid = np.sqrt((X - ue_x)**2 + (Y - ue_y)**2) + 1.0
            rsrp_exp_fbs = THEO_INTERCEPT - 10 * THEO_N * np.log10(d_fbs_grid)
            prob_fbs = norm.pdf(rsrp_obs, loc=rsrp_exp_fbs, scale=THEO_SIGMA)
            
            # Mixture Likelihood per sample
            likelihood = PI * prob_fbs + (1 - PI) * prob_lbs
            
            # Cộng log-likelihood (tránh nhân → underflow)
            batch_log_likelihood += np.log(likelihood + 1e-300)
        
        # Batch update: posterior ∝ prior × exp(Σ log L)
        # Shift log-likelihood to avoid overflow in exp()
        batch_log_likelihood -= np.max(batch_log_likelihood)
        batch_likelihood = np.exp(batch_log_likelihood)
        
        posterior = prior * batch_likelihood
        posterior += 1e-50
        post_sum = np.sum(posterior)
        
        if post_sum > 0:
            prior = posterior / post_sum
        else:
            n_skipped += 1
            print(f"  [SKIP] batch {batch_idx} (t={t_sec:.1f}s): posterior collapsed")
            continue
        
        # MAP estimate
        max_idx = np.unravel_index(np.argmax(prior), prior.shape)
        est_y, est_x = y_coords[max_idx[0]], x_coords[max_idx[1]]
        error = np.sqrt((est_x - fbs_x)**2 + (est_y - fbs_y)**2)
        total_samples_so_far += len(batch)
        rmses.append(error)
        cumulative_samples.append(total_samples_so_far)
        
        # Log chi tiết cho 3 batch đầu
        if batch_idx < 3:
            print(f"  [BATCH {batch_idx}] t={t_sec:.1f}s  samples={len(batch)}  "
                  f"MAP=({est_x:.0f},{est_y:.0f})  error={error:.1f}m  "
                  f"posterior_max={np.max(prior):.2e}")
        elif (batch_idx + 1) % 10 == 0:
            print(f"  [PROGRESS] batch {batch_idx+1}/{n_batches}  "
                  f"MAP=({est_x:.0f},{est_y:.0f})  error={error:.1f}m")
    
    # 4. Kết quả
    print(f"\n{'='*60}")
    print(f"[RESULT] Samples processed: {len(rmses)}  skipped: {n_skipped}")
    print(f"[RESULT] FBS ước lượng : ({est_x:.1f}, {est_y:.1f})")
    print(f"[RESULT] FBS thực tế   : ({fbs_x:.1f}, {fbs_y:.1f})")
    print(f"[RESULT] Sai số cuối   : {rmses[-1]:.2f} mét")
    if len(rmses) > 10:
        print(f"[RESULT] Sai số min    : {min(rmses):.2f} mét (tại batch {np.argmin(rmses)})")
        print(f"[RESULT] Sai số trung bình (last 10%): {np.mean(rmses[-(len(rmses)//10):]):.2f} mét")
    
    # 5. Vẽ sơ đồ hội tụ
    batch_indices = np.arange(1, len(rmses) + 1)
    cum_N = np.array(cumulative_samples)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_N, rmses, 'b-', linewidth=1.5, alpha=0.7, label='Localization Error')
    
    # Đường trung bình trượt (moving average, window=5)
    if len(rmses) > 5:
        window = min(5, len(rmses) // 3)
        ma = np.convolve(rmses, np.ones(window)/window, mode='valid')
        ma_x = cum_N[window-1:]  # align with valid convolution output
        plt.plot(ma_x, ma, 'r-', linewidth=2, label=f'Moving Avg (w={window})')
    
    # Theoretical bound: σ_d / √N  (with N = cumulative samples)
    theo_bound = 314.25 / np.sqrt(cum_N)
    plt.plot(cum_N, theo_bound, 'g--', linewidth=2, label='Theoretical Bound')
    
    plt.xlabel('Cumulative Samples (N)', fontsize=12)
    plt.ylabel('Localization Error (m)', fontsize=12)
    plt.title(f'Mixture Model Convergence  [n={THEO_N}, σ={THEO_SIGMA}, π={PI}]', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Y-axis: cap at reasonable value to see convergence
    y_cap = min(max(rmses[:min(5, len(rmses))]) * 1.2, np.percentile(rmses, 95) * 1.5)
    plt.ylim(0, y_cap)
    
    plt.tight_layout()
    plt.savefig(_RESULTS / 'mixture_convergence.png', dpi=150)
    plt.close()
    print(f"\n[PLOT] Saved → {_RESULTS / 'mixture_convergence.png'}")

if __name__ == "__main__":
    dec_map_mixture_model()