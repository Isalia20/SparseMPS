import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import matplotlib.colors as mcolors
import matplotlib

matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

OUTPUT_DIR = 'sparse_vs_dense'
DEVICES = ['cuda', 'mps']


def load_dense_results(device):
    """Load dense results from CSV file."""
    filepath = os.path.join(OUTPUT_DIR, device, 'dense_results.csv')
    dense_results = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            size = int(row['Matrix_Size'])
            failed = row['Failed'].lower() == 'true'
            mean_time = float(row['Mean_Time_ms']) / 1000 if row['Mean_Time_ms'] else None
            std_time = float(row['Std_Time_ms']) / 1000 if row['Std_Time_ms'] else None
            
            dense_results[size] = {
                'mean': mean_time,
                'std': std_time,
                'failed': failed
            }
    
    return dense_results


def load_sparse_results(device):
    """Load sparse results from CSV file."""
    filepath = os.path.join(OUTPUT_DIR, device, 'sparse_results.csv')
    sparse_results = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            size = int(row['Matrix_Size'])
            nnz_ratio = float(row['NNZ_Ratio'])
            failed = row['Failed'].lower() == 'true'
            mean_time = float(row['Mean_Time_ms']) / 1000 if row['Mean_Time_ms'] else None
            std_time = float(row['Std_Time_ms']) / 1000 if row['Std_Time_ms'] else None
            
            if size not in sparse_results:
                sparse_results[size] = {}
            
            sparse_results[size][nnz_ratio] = {
                'mean': mean_time,
                'std': std_time,
                'failed': failed
            }
    
    return sparse_results


def get_matrix_sizes_and_nnz_ratios(dense_results, sparse_results):
    """Extract matrix sizes and nnz ratios from loaded data."""
    matrix_sizes = sorted(dense_results.keys())
    
    nnz_ratios = set()
    for size in sparse_results:
        for nnz in sparse_results[size]:
            nnz_ratios.add(nnz)
    nnz_ratios = sorted(nnz_ratios)
    
    return matrix_sizes, nnz_ratios


def ensure_output_dir(subdir=None):
    path = OUTPUT_DIR if subdir is None else os.path.join(OUTPUT_DIR, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_computation_time_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios, device):
    """Save computation time vs matrix size plot."""
    output_path = ensure_output_dir(device)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    dense_times = []
    dense_sizes = []
    for size in matrix_sizes:
        if not dense_results[size]['failed']:
            dense_times.append(dense_results[size]['mean'] * 1000)
            dense_sizes.append(size)
    
    if dense_times:
        ax1.plot(dense_sizes, dense_times, 'o-', label='Dense', alpha=0.7, 
                 linewidth=2, markersize=8, color='black')
    
    for nnz_ratio in nnz_ratios:
        sparse_times = []
        sizes = []
        for size in matrix_sizes:
            if nnz_ratio in sparse_results.get(size, {}) and not sparse_results[size][nnz_ratio]['failed']:
                sparse_times.append(sparse_results[size][nnz_ratio]['mean'] * 1000)
                sizes.append(size)
        if sparse_times:
            ax1.plot(sizes, sparse_times, 's--', label=f'Sparse {nnz_ratio:.1%}', alpha=0.7)
    
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Computation Time vs Matrix Size ({device.upper()})')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'computation_time_vs_size.eps')
    fig1.savefig(filepath, format='eps', bbox_inches='tight')
    print(f"Saved: {filepath}")


def save_speedup_vs_sparsity_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios, device):
    """Save speedup vs sparsity plot."""
    output_path = ensure_output_dir(device)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for size in matrix_sizes:
        if dense_results[size]['failed']:
            continue
        speedups = []
        nnz_vals = []
        dense_time = dense_results[size]['mean']
        for nnz in nnz_ratios:
            if nnz in sparse_results.get(size, {}) and not sparse_results[size][nnz]['failed']:
                speedup = dense_time / sparse_results[size][nnz]['mean']
                speedups.append(speedup)
                nnz_vals.append(nnz)
        if speedups:
            ax2.plot(nnz_vals, speedups, 'o-', label=f'{size}x{size}', alpha=0.7)
    
    ax2.set_xlabel('NNZ %')
    ax2.set_ylabel('Speedup (Dense/Sparse)')
    ax2.set_title(f'Speedup vs Sparsity Level ({device.upper()})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'speedup_vs_sparsity.eps')
    fig2.savefig(filepath, format='eps', bbox_inches='tight')
    print(f"Saved: {filepath}")


def save_speedup_heatmap(dense_results, sparse_results, matrix_sizes, nnz_ratios, device):
    """Save speedup heatmap."""
    output_path = ensure_output_dir(device)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    speedup_matrix = np.full((len(matrix_sizes), len(nnz_ratios)), np.nan)
    for i, size in enumerate(matrix_sizes):
        if dense_results[size]['failed']:
            continue
        dense_time = dense_results[size]['mean']
        for j, nnz in enumerate(nnz_ratios):
            if nnz in sparse_results.get(size, {}) and not sparse_results[size][nnz]['failed']:
                speedup_matrix[i, j] = dense_time / sparse_results[size][nnz]['mean']

    masked_speedup = np.ma.masked_invalid(speedup_matrix)

    colors_below = plt.cm.Reds_r(np.linspace(0.1, 1.0, 128))
    colors_above = plt.cm.Blues(np.linspace(0.1, 1.0, 128))
    all_colors = np.vstack([colors_below, colors_above])
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('FastRedWhiteBlue', all_colors)

    valid = masked_speedup.compressed()
    if len(valid) > 0:
        max_dev = max(abs(valid.max() - 1), abs(valid.min() - 1))
        max_dev *= 0.25
    else:
        max_dev = 1.0

    vmin = 1.0 - max_dev
    vmax = 1.0 + max_dev

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax3.imshow(masked_speedup, aspect='auto', cmap=custom_cmap, norm=norm, origin='lower')

    ax3.set_xticks(range(len(nnz_ratios)))
    ax3.set_xticklabels([f'{nnz:.1%}' for nnz in nnz_ratios], rotation=45)
    ax3.set_yticks(range(len(matrix_sizes)))
    ax3.set_yticklabels([str(size) for size in matrix_sizes])

    ax3.set_xlabel('NNZ %')
    ax3.set_ylabel('Matrix Size')
    ax3.set_title(f'Speedup Heatmap (Dense/Sparse) - {device.upper()}\nRed: Sparse slower, White: equal (1×), Blue: Sparse faster')

    cbar = plt.colorbar(im, ax=ax3, label='Speedup')
    cbar_ticks = [vmin, 1.0, vmax]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{vmin:.2f}", "1.00", f"{vmax:.2f}"])
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'speedup_heatmap.eps')
    fig3.savefig(filepath, format='eps', bbox_inches='tight')
    print(f"Saved: {filepath}")


def save_sparse_performance_plot(sparse_results, matrix_sizes, device):
    """Save sparse performance plot."""
    output_path = ensure_output_dir(device)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    for nnz_ratio in [0.001, 0.01, 0.1, 0.5]:
        sparse_times = []
        sizes_list = []
        for size in matrix_sizes:
            if nnz_ratio in sparse_results.get(size, {}) and not sparse_results[size][nnz_ratio]['failed']:
                sparse_times.append(sparse_results[size][nnz_ratio]['mean'] * 1000)
                sizes_list.append(size)
        if sparse_times:
            ax4.plot(sizes_list, sparse_times, 'o-', label=f'Sparse {nnz_ratio:.1%}', alpha=0.7)
    
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Sparse Matmul Time (ms)')
    ax4.set_title(f'Sparse Performance ({device.upper()})')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'sparse_performance.eps')
    fig4.savefig(filepath, format='eps', bbox_inches='tight')
    print(f"Saved: {filepath}")


def generate_plots_for_device(device):
    """Generate all plots for a specific device."""
    print(f"\n{'='*80}")
    print(f"Generating plots for {device.upper()}")
    print('='*80)
    
    dense_results = load_dense_results(device)
    sparse_results = load_sparse_results(device)
    matrix_sizes, nnz_ratios = get_matrix_sizes_and_nnz_ratios(dense_results, sparse_results)
    
    print(f"Loaded data for {len(matrix_sizes)} matrix sizes and {len(nnz_ratios)} nnz ratios")
    
    save_computation_time_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios, device)
    save_speedup_vs_sparsity_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios, device)
    save_speedup_heatmap(dense_results, sparse_results, matrix_sizes, nnz_ratios, device)
    save_sparse_performance_plot(sparse_results, matrix_sizes, device)


def print_summary():
    print("\n" + "="*80)
    print("Summary:")
    print(f"  EPS files generated per device (in cuda/ and mps/ folders):")
    print(f"    - computation_time_vs_size.eps")
    print(f"    - speedup_vs_sparsity.eps")
    print(f"    - speedup_heatmap.eps")
    print(f"    - sparse_performance.eps")


def main():
    for device in DEVICES:
        device_dir = os.path.join(OUTPUT_DIR, device)
        if os.path.exists(device_dir):
            generate_plots_for_device(device)
        else:
            print(f"Warning: Directory {device_dir} not found, skipping {device}")
    
    print_summary()
    plt.show()


if __name__ == '__main__':
    main()
