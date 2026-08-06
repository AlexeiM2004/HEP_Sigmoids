### ------------------------------ Imports ------------------------------ ###

import matplotlib.pyplot as plt 
import numpy as np
import torch
import h5py
import vector
import awkward as ak
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import yaml

### ------------------------------ Print Current Timestamp ------------------------------ ###

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Job started at :", formatted_time)

### ------------------------------ Device Usage ------------------------------ ###

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device with number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ Load YAML ------------------------------ ###

with open("config_data.yaml", "r") as f:
    config_data = yaml.safe_load(f)

with open("config_model.yaml", "r") as f:
    config_model = yaml.safe_load(f)

# ------------------------------ Print Configs ------------------------------ #

def print_yaml_config(config, title):
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

print_yaml_config(config_data, "DATA CONFIGURATION")
print_yaml_config(config_model, "MODEL CONFIGURATION")

# ------------------------------ Extract Values ------------------------------ #

# Data loading
evaluation_results = config_data["saving"]["evaluation_results"]
test_file = config_data["data"]["test_file"]

# Analysis from model config
target_names = config_model["analysis"]["target_names"]
obs_fields = config_model["analysis"]["obs_fields"]

# Build plot file names
loss_curve_r2_summary_plots = "loss_curve_r2_summary_plots.png"
distribution_summary_plots = "distribution_summary_plots.png"
resolution_summary_plots = "resolution_summary_plots.png"
scatter_summary_plots = "scatter_summary_plots.png"
spin_observables_plots = "spin_observables_plots.png"

### ------------------------------ Load Data ------------------------------ ###

with h5py.File(evaluation_results, "r") as f:
    Y_pred_unscaled = f["Y_pred"][:]
    Y_test_unscaled = f["Y_test"][:]
    scaler_Y_mean = f["Y_mean"][:]
    scaler_Y_scale = f["Y_scale"][:]

output_dim = Y_pred_unscaled.shape[1]

### ------------------------------ Calculate Metrics ------------------------------ ###

def compute_distribution_metrics(y_pred, y_true, bins=100):
    # Flatten to 1D numpy arrays
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    
    # MAE, MSE, RMSE, R^2
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # SciPy 1D Wasserstein Distance
    wd = wasserstein_distance(y_pred, y_true)
    
    # SciPy KL Divergence
    min_val = min(np.min(y_pred), np.min(y_true))
    max_val = max(np.max(y_pred), np.max(y_true))
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Compute histogram counts
    p_counts, _ = np.histogram(y_pred, bins=bin_edges)
    q_counts, _ = np.histogram(y_true, bins=bin_edges)
    
    # Convert counts to probabilities with epsilon smoothing to prevent log(0)
    eps = 1e-10
    p_prob = (p_counts + eps) / np.sum(p_counts + eps)
    q_prob = (q_counts + eps) / np.sum(q_counts + eps)
    
    # SciPy KL Divergence D_KL(P || Q)
    kl_div = entropy(p_prob, q_prob)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'Wasserstein': wd,
        'KL': kl_div
    }

def mahalanobis_distance(y_pred, y_true):
    # Mean and covariance of the true distribution
    mu_true = np.mean(y_true, axis=0)
    cov_true = np.cov(y_true, rowvar=False)
    # Add small regularization to avoid singular matrix
    cov_true += 1e-6 * np.eye(cov_true.shape[0])
    inv_cov = inv(cov_true)
    
    # Compute distance for each event
    distances = np.array([mahalanobis(p, mu_true, inv_cov) for p in y_pred])
    return distances

def mig_matrix(m_reco,m_truth, bins):
    # import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Define Binning
    # Example: 0 to 3000 GeV with 50 bins
    # bins = np.linspace(0, 500, 10) 

    # 2. Compute Histogram
    # histogram2d(x, y) results in H[i,j] where i is x-bin and j is y-bin
    # To have Reco on X and Truth on Y:
    H, xedges, yedges = np.histogram2d(m_reco, m_truth, bins=bins)

    # 3. Normalize by Truth (the Y-axis bins)
    # In the H matrix from histogram2d, summing over axis 0 (Reco) 
    # gives the total counts for each Truth bin (axis 1).
    truth_bin_counts = np.sum(H, axis=0)
    print(f"Truth bin counts: {truth_bin_counts}")

    # Avoid division by zero for empty truth bins
    truth_bin_counts[truth_bin_counts == 0] = 1

    # Normalize: divide each 'reco' column by the total 'truth' counts for that truth bin
    for i in range(H.shape[1]):
        H[:, i] /= truth_bin_counts[i]
        
    migration_matrix = H
    
    # Print the trace diagonal divided by the total sum
    diagonal_sum = np.trace(migration_matrix)
    total_sum = np.sum(migration_matrix)
    print(f"{diagonal_sum/total_sum:.2f}")

    # 4. Plotting
    plt.figure(figsize=(4, 3))
    # Use pcolormesh. Note: We transpose (.T) because pcolormesh expects 
    # the array shape to match (len(y), len(x))
    plt.pcolormesh(xedges, yedges, migration_matrix.T, cmap='viridis', shading='auto')

    plt.colorbar(label='P(Reco | Truth)')
    plt.title('Migration Matrix (Normalized by Truth Row)')

    # Perfect reconstruction line
    plt.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color='white', linestyle='--', alpha=0.6)
    
    # Print the bin counts in each bin 
    plt.tight_layout()
    return plt

### ------------------------------ Identify Plot Grid ------------------------------ ###

def get_plot_grid(n_plots):
    if n_plots <= 1:
        return 1, 1
    elif n_plots <= 2:
        return 1, 2
    elif n_plots <= 4:
        return 2, 2
    elif n_plots <= 6:
        return 2, 3
    elif n_plots <= 8:
        return 2, 4
    elif n_plots <= 12:
        return 3, 4
    else:
        return 4, 4

n_rows, n_cols = get_plot_grid(output_dim)

### ------------------------------ Display Metrics ------------------------------ ###

# Store target feature names in an array
target_names = ['mttbar']

# Calculate MAE, RMSE, MSE, R2, Wasserstein distance, and KL divergence

print("\n")
print("="*60)
print("TARGET FEATURE METRICS")
print("="*60)

r2_per_dim = []
for i, feature in enumerate(target_names):
    res = compute_distribution_metrics(
        y_pred=Y_pred_unscaled[:, i], 
        y_true=Y_test_unscaled[:, i], 
        bins=100
    )
    r2_per_dim.append(res['R2'])
    print(f"Feature : {feature} | MAE : {res['MAE']:.4f} | RMSE : {res['RMSE']:.4f} | MSE : {res['MSE']:.4f} | R^2 : {res['R2']:.4f} | Wasserstein : {res['Wasserstein']:.4f} | KL Div : {res['KL']:.4f}")

### ------------------------------ Plot Loss Curve and R^2 Plots ------------------------------ #

fig1, axes1 = plt.subplots(1, 1, figsize=(12, 6))
bars = axes1.bar(range(output_dim), r2_per_dim, color='blue', edgecolor='black')
axes1.set_xticks(range(output_dim))
axes1.set_xticklabels(target_names, rotation=90, ha='center')
axes1.set_ylabel('R-Squared')
axes1.set_title('R-Squared per Feature')
axes1.set_ylim([-0.1, 1.0])
axes1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_per_dim):
    axes1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(loss_curve_r2_summary_plots)
plt.close()
### ------------------------------ Plot Target Feature Distribution ------------------------------ #

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes2 = np.ravel(axes2)

for i in range(output_dim):
    ax = axes2[i]
    ax.hist(Y_test_unscaled[:, i], bins=100, density=True, histtype='step',
            label='True', color='blue', linewidth=1.5)
    ax.hist(Y_pred_unscaled[:, i], bins=100, density=True, histtype='step',
            label='Pred', color='red', linewidth=1.5)
    ax.set_title(target_names[i], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(output_dim, n_rows * n_cols):
    axes2[i].set_visible(False)

plt.tight_layout()
plt.savefig(distribution_summary_plots)

### ------------------------------ Plot Target Feature Resolution Plots ------------------------------ #

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes3 = np.ravel(axes3)

for i in range(output_dim):
    ax = axes3[i]
    residuals = Y_pred_unscaled[:, i] - Y_test_unscaled[:, i]
    ax.hist(residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
    ax.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
               label=f'μ={np.mean(residuals):.2f}')
    ax.set_title(target_names[i], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for i in range(output_dim, n_rows * n_cols):
    axes3[i].set_visible(False)

plt.tight_layout()
plt.savefig(resolution_summary_plots)

### ------------------------------ Plot Target Feature Scatter Plots ------------------------------ #

fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes4 = np.ravel(axes4)

for i in range(output_dim):
    ax = axes4[i]
    ax.scatter(Y_test_unscaled[:, i], Y_pred_unscaled[:, i],
               alpha=0.1, s=1, color='blue')
    min_val = min(Y_test_unscaled[:, i].min(), Y_pred_unscaled[:, i].min())
    max_val = max(Y_test_unscaled[:, i].max(), Y_pred_unscaled[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True', fontsize=8)
    ax.set_ylabel('Pred', fontsize=8)
    r2 = r2_per_dim[i]
    ax.set_title(f'{target_names[i]}\nR² = {r2:.3f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

for i in range(output_dim, n_rows * n_cols):
    axes4[i].set_visible(False)

plt.tight_layout()
plt.savefig(scatter_summary_plots)


### ------------------------------ Spin Observable Calculation Functions ------------------------------ #

direct_regression = True

if direct_regression == False:   
    def boost(top, anti_top, leptonP, leptonM, top_mass_true, antitop_mass_true):

        # Build ttbar system
        ttbar = top + anti_top 

        # Boost tops lab --> ttbar_CoM
        top_in_CoM      = top.boostCM_of(ttbar)
        antitop_in_CoM  = anti_top.boostCM_of(ttbar) 

        # Apply on shell correction
        #px, py, pz = top_in_CoM.px, top_in_CoM.py, top_in_CoM.pz
        #p_norm = np.sqrt(px**2 + py**2 + pz**2)
        #E_corrected = np.sqrt(p_norm**2 + top_mass_true**2)
        #top_in_CoM_corrected = vector.zip({
        #    'px': px,
        #    'py': py,
        #    'pz': pz,
        #    'E': E_corrected})

        #apx, apy, apz = antitop_in_CoM.px, antitop_in_CoM.py, antitop_in_CoM.pz
        #ap_norm = np.sqrt(apx**2 + apy**2 + apz**2)
        #anti_E_corrected = np.sqrt(ap_norm**2 + antitop_mass_true**2)
        #antitop_in_CoM_corrected = vector.zip({
        #    'px': apx,
        #    'py': apy,
        #    'pz': apz,
        #    'E': anti_E_corrected})
        
        # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
        leptonP_in_CoM     = leptonP.boostCM_of(ttbar)
        leptonM_in_CoM     = leptonM.boostCM_of(ttbar)
        leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
        leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

        # Compute the unit 3-vector directions
        Kdirection        = top_in_CoM.to_beta3().unit()
        leptonP_direction = leptonP_in_top.to_beta3().unit()
        leptonM_direction = leptonM_in_antitop.to_beta3().unit()

        return Kdirection, leptonP_direction, leptonM_direction, ttbar, top_in_CoM, antitop_in_CoM, leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop


    def helicity_basis_observables(Kdirection, leptonP_direction, leptonM_direction, ttbar, top_in_CoM_corrected, antitop_in_CoM_corrected, leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop):

        # Kinematics
        z     = vector.obj(x=0,y=0,z=1)
        cos_T = z.dot(Kdirection)
        sin_T = (1 - cos_T**2)**0.5
        mask = 1*(cos_T > 0) -1*(cos_T < 0)

        # Helicity basis observables
        cos_K_plus  = Kdirection.dot(leptonP_direction)
        cos_K_minus = -Kdirection.dot(leptonM_direction) 

        Ndirection = z.cross(Kdirection)/sin_T
        cos_N_plus  = mask*Ndirection.dot(leptonP_direction)
        cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 

        Rdirection  =  1/sin_T * (z - cos_T*Kdirection)
        cos_R_plus  = mask*Rdirection.dot(leptonP_direction)
        cos_R_minus = -mask*Rdirection.dot(leptonM_direction) 

        cos_phi     = leptonP_direction.dot(leptonM_direction)

        helicity_observables = ak.zip({
        "cos_K_plus"  :  cos_K_plus,
        "cos_K_minus" :  cos_K_minus,
        "cos_N_plus"  :  cos_N_plus,
        "cos_N_minus" :  cos_N_minus,
        "cos_R_plus"  :  cos_R_plus,
        "cos_R_minus" :  cos_R_minus,
        "cos_phi"     :  cos_phi,
        "cos_theta_star" : cos_T 
        }, depth_limit=1, with_name="Event")

        return helicity_observables

### ------------------------------ Load RECO (pred) Leptons ------------------------------ ###

if direct_regression == False:   
    with h5py.File(test_file, "r") as f:
        R_test = f["R"][:]

    leptonP_reco = vector.Array(
        ak.zip({
            'px': R_test[:, 0],
            'py': R_test[:, 1],
            'pz': R_test[:, 2],
            'E': R_test[:, 3]
        })
    )

    leptonM_reco = vector.Array(
        ak.zip({
            'px': R_test[:, 4],
            'py': R_test[:, 5],
            'pz': R_test[:, 6],
            'E': R_test[:, 7]
        })
    )

### ------------------------------ Load Parton (true) Leptons ------------------------------ ###

if direct_regression == False:   
    with h5py.File(test_file, "r") as f:
        L_test = f["L"][:]

    leptonP_truth = vector.Array(
        ak.zip({
            'px': L_test[:, 0],
            'py': L_test[:, 1],
            'pz': L_test[:, 2],
            'E': L_test[:, 3]
        })
    )
    leptonM_truth = vector.Array(
        ak.zip({
            'px': L_test[:, 4],
            'py': L_test[:, 5],
            'pz': L_test[:, 6],
            'E': L_test[:, 7]
        })
    )

### ------------------------------ Spin Observable Calculation ------------------------------ #

if direct_regression == False:   
    Y_pred_awk = ak.from_numpy(Y_pred_unscaled)
    Y_test_awk = ak.from_numpy(Y_test_unscaled)

    # Predicted vectors
    top_pred = vector.Array(
        ak.zip({
            'px': Y_pred_awk[:, 0],
            'py': Y_pred_awk[:, 1],
            'pz': Y_pred_awk[:, 2],
            'E': Y_pred_awk[:,3]
        })
    )

    antitop_pred = vector.Array(
        ak.zip({
            'px': Y_pred_awk[:, 4],
            'py': Y_pred_awk[:, 5],
            'pz': Y_pred_awk[:, 6],
            'E': Y_pred_awk[:,7]
        })
    )

    # True vectors
    top_true = vector.Array(
        ak.zip({
            'px': Y_test_awk[:, 0],
            'py': Y_test_awk[:, 1],
            'pz': Y_test_awk[:, 2],
            'E': Y_test_awk[:,3]
        })
    )

    antitop_true = vector.Array(
        ak.zip({
            'px': Y_test_awk[:, 4],
            'py': Y_test_awk[:, 5],
            'pz': Y_test_awk[:, 6],
            'E': Y_test_awk[:,7]
        })
    )

### ------------------------------ Calculate Spin Observable Metrics ------------------------------ #

if direct_regression == False:   
    # Calculate Spin Observables
    obs_fields = ['cos_K_plus','cos_K_minus','cos_N_plus','cos_N_minus','cos_R_plus','cos_R_minus','cos_phi','cos_theta_star']

    # Load test masses
    with h5py.File(test_file, "r") as f:
        M_test = f["M"][:]

    dic_of_spin_observables_pred = helicity_basis_observables(*boost(top_pred, antitop_pred, leptonP_reco, leptonM_reco, M_test[:, 0], M_test[:, 1]))
    dic_of_spin_observables_true = helicity_basis_observables(*boost(top_true, antitop_true, leptonP_truth, leptonM_truth, M_test[:, 0], M_test[:, 1]))

    pred_spin_observables = np.column_stack([ak.to_numpy(dic_of_spin_observables_pred[field]) for field in obs_fields])
    true_spin_observables = np.column_stack([ak.to_numpy(dic_of_spin_observables_true[field]) for field in obs_fields])

    # Compute metrics
    print("\n" + "="*60)
    print("SPIN OBSERVABLE METRICS")
    print("="*60)

    r2_spin = []
    for i, feature in enumerate(obs_fields):
        res = compute_distribution_metrics(
            y_pred=pred_spin_observables[:, i], 
            y_true=true_spin_observables[:, i], 
            bins=100
        )
        r2_spin.append(res['R2'])
        print(f"Feature : {feature} | MAE : {res['MAE']:.4f} | RMSE : {res['RMSE']:.4f} | MSE : {res['MSE']:.4f} | R^2 : {res['R2']:.4f} | Wasserstein : {res['Wasserstein']:.4f} | KL Div : {res['KL']:.4f}")

### ------------------------------ Plot Spin Observable Metrics ------------------------------ #

if direct_regression == False:
    # Plot spin observables
    n_rows_spin, n_cols_spin = get_plot_grid(len(obs_fields))
    fig, axes = plt.subplots(n_rows_spin, n_cols_spin, figsize=(n_cols_spin * 4, n_rows_spin * 3))
    axes = axes.flatten()
    
    for i, name in enumerate(obs_fields):
        ax0 = axes[i * 3]
        ax1 = axes[i * 3 + 1]
        ax2 = axes[i * 3 + 2]
        
        # Distribution
        ax0.hist(true_spin_observables[:, i], bins=100, density=True,
                histtype='step', label='True', color='blue', linewidth=1.5)
        ax0.hist(pred_spin_observables[:, i], bins=100, density=True,
                histtype='step', label='Pred', color='red', linewidth=1.5)
        ax0.set_title(f'{name} Distribution', fontsize=10)
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)
        
        # Scatter
        ax1.scatter(true_spin_observables[:, i], pred_spin_observables[:, i],
                    alpha=0.1, s=1, color='blue')
        min_val = min(true_spin_observables[:, i].min(), pred_spin_observables[:, i].min())
        max_val = max(true_spin_observables[:, i].max(), pred_spin_observables[:, i].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('True', fontsize=8)
        ax1.set_ylabel('Pred', fontsize=8)
        ax1.set_title(f'{name}\nR² = {r2_spin[i]:.3f}', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Resolution
        residuals = pred_spin_observables[:, i] - true_spin_observables[:, i]
        ax2.hist(residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
        ax2.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
                    label=f'μ={np.mean(residuals):.3f}')
        ax2.set_title(f'{name} Resolution', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    for i in range(len(obs_fields) * 3, n_rows_spin * n_cols_spin):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(spin_observables_plots)
    plt.close()