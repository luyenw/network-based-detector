import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_data(cell_db_path, detection_log_path, target_ecgi):
    """Load Ground truth and Measurement Reports filtered by target ECGI."""
    cell_db = pd.read_csv(cell_db_path)
    det_log = pd.read_csv(detection_log_path)
    
    # Extract Ground truth FBS location
    fbs_gt = cell_db[cell_db['is_fake'] == 1].iloc[0]
    fbs_x_true, fbs_y_true = fbs_gt['pos_x'], fbs_gt['pos_y']
    
    # Filter and sort detection log
    df_target = det_log[det_log['ecgi'] == target_ecgi].copy()
    df_target = df_target.sort_values(by='timestamp').reset_index(drop=True)
    
    return df_target, fbs_x_true, fbs_y_true

def initialize_grid(min_x, max_x, min_y, max_y, grid_res):
    """Initialize spatial grid and uniform prior probability."""
    x_coords = np.arange(min_x, max_x + grid_res, grid_res)
    y_coords = np.arange(min_y, max_y + grid_res, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    prior = np.ones_like(X, dtype=float)
    prior /= np.sum(prior)
    return X, Y, x_coords, y_coords, prior

def run_dec_map_localization(df_target, X, Y, x_coords, y_coords, prior, fbs_x_true, fbs_y_true):
    """Run the Grid-based Bayesian Update (DEC-MAP) over time."""
    # Path loss parameters — COST-231 urban macro (MUST match sliding_window_detector.py)
    # Detector uses: rsrp = P_tx - (128.1 + 37.6*log10(d/1000))
    #              = (P_tx - 15.3) - 37.6*log10(d)          [d in metres]
    # → intercept = tx_power - 15.3 = 46 - 15.3 = 30.7 dBm
    # → n_pathloss = 3.76  (10*n = 37.6)
    tx_power   = 46.0          # dBm  (same as cell_database)
    n_pathloss = 3.76          # COST-231 urban macro exponent (37.6/10)
    intercept  = tx_power - 15.3   # = 30.7 dBm  (P_tx - 128.1 + 37.6*3)
    sigma_total = 9.1          # dB  combined fading & location noise

    
    rmses = []
    n_samples = []
    
    for _, row in df_target.iterrows():
        w_i = row['anomaly_score_wi']
        
        # Filter: Only process data with a minimal anomaly score to reduce LBS noise
        if w_i < 0.1:
            continue
            
        ue_x, ue_y = row['estimated_ue_x'], row['estimated_ue_y']
        ue_z = row.get('estimated_ue_z', 2.0)   # UE height [m], default 2m
        rsrp_obs = row['rsrp_observed']
        
        # Calculate Expected RSRP over the whole grid
        # Grid represents candidate FBS positions at z=2m (handheld/vehicle-mounted height).
        # UE is also at z=2m → Δz ≈ 0, so 2D distance is sufficient here.
        fbs_z = 2.0   # candidate FBS height [m]
        dist_grid = np.sqrt((X - ue_x)**2 + (Y - ue_y)**2 + (fbs_z - ue_z)**2) + 1.0
        rsrp_exp = intercept - 10 * n_pathloss * np.log10(dist_grid)
        
        # Calculate Physical Consistency Likelihood
        likelihood = norm.pdf(rsrp_obs, loc=rsrp_exp, scale=sigma_total)
        
        # Weighted Update using Anomaly Score w_i
        effective_likelihood = w_i * likelihood + (1 - w_i) * np.max(likelihood)
        posterior_unnorm = prior * effective_likelihood
        
        if np.sum(posterior_unnorm) == 0:
            continue
            
        # Normalize
        prior = posterior_unnorm / np.sum(posterior_unnorm)
        
        # Get MAP (Maximum A Posteriori) Estimate
        max_idx = np.unravel_index(np.argmax(prior), prior.shape)
        est_y, est_x = y_coords[max_idx[0]], x_coords[max_idx[1]]
        
        # Calculate Current RMSE
        current_rmse = np.sqrt((est_x - fbs_x_true)**2 + (est_y - fbs_y_true)**2)
        rmses.append(current_rmse)
        n_samples.append(len(rmses))
        
    return est_x, est_y, rmses, n_samples, prior

def plot_results(X, Y, prior, n_samples, rmses, fbs_x_true, fbs_y_true, est_x, est_y):
    """Plot Convergence Curve and Final Heatmap."""
    # 1. RMSE Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples, rmses, label='Empirical RMSE', linewidth=2)
    
    # Plot Theoretical Bound = (G * sigma_d) / sqrt(N)
    theo_bound = 314.25 / np.sqrt(n_samples)
    plt.plot(n_samples, theo_bound, 'r--', label='Theoretical Bound', linewidth=2)
    
    plt.xlabel('Number of Anomalous Samples (N)', fontsize=12)
    plt.ylabel('Localization Error (m)', fontsize=12)
    plt.title('DEC-MAP Convergence: Empirical vs Theoretical', fontsize=14)
    plt.ylim(0, max(rmses[:10]) * 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig('..\\results\\rmse_convergence.png')
    plt.close()
    
    # 2. Final Posterior Heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, prior, shading='auto', cmap='hot')
    plt.colorbar(label='Probability')
    plt.scatter(fbs_x_true, fbs_y_true, color='blue', marker='*', s=200, label='True FBS Location')
    plt.scatter(est_x, est_y, color='green', marker='o', s=100, label='Estimated Location', facecolors='none', edgecolors='green', linewidth=2)
    plt.title('DEC-MAP Final Posterior Heatmap', fontsize=14)
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.savefig('..\\results\\heatmap.png')
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    TARGET_ECGI = "001-01-0000001"
    
    print("1. Loading and filtering data...")
    df_target, fbs_x, fbs_y = load_data('..\\results\\cell_database.csv', '..\\results\\detection_log.csv', TARGET_ECGI)
    
    print(f"Ground Truth FBS is at: ({fbs_x}, {fbs_y})")
    
    print("2. Initializing Spatial Grid...")
    X, Y, x_coords, y_coords, prior = initialize_grid(-500, 2500, -500, 2500, grid_res=10)
    
    print("3. Running DEC-MAP Localization...")
    est_x, est_y, rmses, n_samples, final_prior = run_dec_map_localization(
        df_target, X, Y, x_coords, y_coords, prior, fbs_x, fbs_y
    )
    
    print(f"4. Generating Plots...")
    plot_results(X, Y, final_prior, n_samples, rmses, fbs_x, fbs_y, est_x, est_y)
    
    print(f"Final Estimated Location: ({est_x}, {est_y})")
    print(f"Final RMSE (Empirical Error): {rmses[-1]:.2f} meters")
    
    theo_error = 314.25 / np.sqrt(n_samples[-1])
    print(f"Theoretical Error Bound: {theo_error:.2f} meters, {n_samples[-1]} samples")