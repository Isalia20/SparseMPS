import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import matplotlib.colors as mcolors
import matplotlib

matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

OUTPUT_DIR = 'sparse_vs_dense'


def create_sparse_matrix(size, nnz_ratio):
    total_elements = size * size
    nnz = int(total_elements * nnz_ratio)
    indices = torch.randint(0, size, (2, nnz))
    values = torch.randn(nnz)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (size, size))
    return sparse_tensor.coalesce()


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def empty_cache(device):
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()


def benchmark_dense_matmul(size, num_runs=10, device='mps' if torch.mps.is_available() else 'cpu'):
    result = {
        'mean': None,
        'std': None,
        'failed': False
    }
    try:
        dense_A = torch.randn(size, size, device=device)
        dense_B = torch.randn(size, size, device=device)
        synchronize(device)
        _ = torch.mm(dense_A, dense_B)
        synchronize(device)
        times_dense = []
        for _ in range(num_runs):
            synchronize(device)
            start = time.time()
            result_dense = torch.mm(dense_A, dense_B)
            synchronize(device)
            times_dense.append(time.time() - start)
        result['mean'] = np.mean(times_dense)
        result['std'] = np.std(times_dense)
        del dense_A, dense_B, result_dense
        empty_cache(device)
    except RuntimeError as e:
        result['failed'] = True
        print(f"      Dense {size}x{size} failed: {str(e)[:80]}...")
        empty_cache(device)
    return result


def benchmark_sparse_matmul(size, nnz_ratio, num_runs=10, device='mps' if torch.mps.is_available() else 'cpu'):
    result = {
        'mean': None,
        'std': None,
        'failed': False
    }
    try:
        sparse_A = create_sparse_matrix(size, nnz_ratio).to(device)
        dense_B = torch.randn(size, size, device=device)
        synchronize(device)
        _ = torch.sparse.mm(sparse_A, dense_B)
        synchronize(device)
        times_sparse = []
        for _ in range(num_runs):
            synchronize(device)
            start = time.time()
            result_sparse = torch.sparse.mm(sparse_A, dense_B)
            synchronize(device)
            times_sparse.append(time.time() - start)
        result['mean'] = np.mean(times_sparse)
        result['std'] = np.std(times_sparse)
        del sparse_A, dense_B, result_sparse
        empty_cache(device)
    except RuntimeError as e:
        result['failed'] = True
        print(f"      Sparse {size}x{size} (nnz={nnz_ratio:.1%}) failed: {str(e)[:80]}...")
        empty_cache(device)
    return result


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_results_to_csv(dense_results, sparse_results, matrix_sizes, nnz_ratios, device):
    ensure_output_dir()
    
    dense_csv_filename = os.path.join(OUTPUT_DIR, 'dense_results.csv')
    with open(dense_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matrix_Size', 'Mean_Time_ms', 'Std_Time_ms', 'Failed', 'Device', 'PyTorch_Version'])
        for size in matrix_sizes:
            result = dense_results[size]
            mean_ms = result['mean'] * 1000 if result['mean'] is not None else ''
            std_ms = result['std'] * 1000 if result['std'] is not None else ''
            writer.writerow([
                size,
                f"{mean_ms:.4f}" if mean_ms != '' else '',
                f"{std_ms:.4f}" if std_ms != '' else '',
                result['failed'],
                device,
                torch.__version__
            ])
    print(f"\nDense results saved to: {dense_csv_filename}")
    
    sparse_csv_filename = os.path.join(OUTPUT_DIR, 'sparse_results.csv')
    with open(sparse_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matrix_Size', 'NNZ_Ratio', 'Mean_Time_ms', 'Std_Time_ms',
                        'Failed', 'Dense_Time_ms', 'Speedup', 'Device', 'PyTorch_Version'])
        for size in matrix_sizes:
            dense_time = dense_results[size]['mean'] if not dense_results[size]['failed'] else None
            dense_time_ms = dense_time * 1000 if dense_time is not None else ''
            for nnz_ratio in nnz_ratios:
                if nnz_ratio in sparse_results[size]:
                    result = sparse_results[size][nnz_ratio]
                    mean_ms = result['mean'] * 1000 if result['mean'] is not None else ''
                    std_ms = result['std'] * 1000 if result['std'] is not None else ''
                    if dense_time is not None and not result['failed']:
                        speedup = dense_time / result['mean']
                        speedup_str = f"{speedup:.4f}"
                    else:
                        speedup_str = ''
                    writer.writerow([
                        size,
                        nnz_ratio,
                        f"{mean_ms:.4f}" if mean_ms != '' else '',
                        f"{std_ms:.4f}" if std_ms != '' else '',
                        result['failed'],
                        f"{dense_time_ms:.4f}" if dense_time_ms != '' else '',
                        speedup_str,
                        device,
                        torch.__version__
                    ])
    print(f"Sparse results saved to: {sparse_csv_filename}")
    
    summary_csv_filename = os.path.join(OUTPUT_DIR, 'benchmark_summary.csv')
    with open(summary_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matrix_Size', 'Operation_Type', 'NNZ_Ratio', 'Mean_Time_ms',
                        'Std_Time_ms', 'Speedup_vs_Dense', 'Failed'])
        for size in matrix_sizes:
            dense_result = dense_results[size]
            dense_mean_ms = dense_result['mean'] * 1000 if dense_result['mean'] is not None else ''
            dense_std_ms = dense_result['std'] * 1000 if dense_result['std'] is not None else ''
            writer.writerow([
                size,
                'Dense',
                1.0,
                f"{dense_mean_ms:.4f}" if dense_mean_ms != '' else '',
                f"{dense_std_ms:.4f}" if dense_std_ms != '' else '',
                '1.0000',
                dense_result['failed']
            ])
            for nnz_ratio in nnz_ratios:
                if nnz_ratio in sparse_results[size]:
                    sparse_result = sparse_results[size][nnz_ratio]
                    sparse_mean_ms = sparse_result['mean'] * 1000 if sparse_result['mean'] is not None else ''
                    sparse_std_ms = sparse_result['std'] * 1000 if sparse_result['std'] is not None else ''
                    if dense_result['mean'] is not None and not sparse_result['failed']:
                        speedup = dense_result['mean'] / sparse_result['mean']
                        speedup_str = f"{speedup:.4f}"
                    else:
                        speedup_str = ''
                    writer.writerow([
                        size,
                        'Sparse',
                        nnz_ratio,
                        f"{sparse_mean_ms:.4f}" if sparse_mean_ms != '' else '',
                        f"{sparse_std_ms:.4f}" if sparse_std_ms != '' else '',
                        speedup_str,
                        sparse_result['failed']
                    ])
    print(f"Summary results saved to: {summary_csv_filename}")
    return dense_csv_filename, sparse_csv_filename, summary_csv_filename


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def print_header(device):
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == 'mps':
        print("Note: Using MPS (Apple Silicon). Memory limits may affect larger matrices.")
    elif device == 'cuda':
        print(f"Note: Using CUDA. GPU: {torch.cuda.get_device_name(0)}")
    print("\n" + "="*80)


def run_dense_benchmarks(matrix_sizes, device):
    dense_results = {}
    print("\nBenchmarking Dense Operations:")
    print("-" * 80)
    for size in matrix_sizes:
        print(f"Matrix Size: {size}x{size}...", end=' ')
        dense_results[size] = benchmark_dense_matmul(size, num_runs=5, device=device)
        if not dense_results[size]['failed']:
            print(f"{dense_results[size]['mean']*1000:.2f}ms")
        else:
            print("FAILED")
    return dense_results


def run_sparse_benchmarks(matrix_sizes, nnz_ratios, dense_results, device):
    sparse_results = {}
    print("\n\nBenchmarking Sparse Operations:")
    print("="*80)
    for size in matrix_sizes:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 80)
        sparse_results[size] = {}
        dense_time = dense_results[size]['mean'] if not dense_results[size]['failed'] else None
        dense_str = f"{dense_time*1000:7.2f}ms" if dense_time is not None else "FAIL"
        for nnz_ratio in nnz_ratios:
            sparse_result = benchmark_sparse_matmul(size, nnz_ratio, num_runs=5, device=device)
            sparse_results[size][nnz_ratio] = sparse_result
            sparse_str = "FAIL" if sparse_result['failed'] else f"{sparse_result['mean']*1000:7.2f}ms"
            if dense_time is not None and not sparse_result['failed']:
                speedup = dense_time / sparse_result['mean']
                speedup_str = f"{speedup:5.2f}x"
            else:
                speedup_str = "N/A"
            print(f"NNZ ratio: {nnz_ratio:5.1%} | "
                  f"Dense: {dense_str:>12} | "
                  f"Sparse: {sparse_str:>12} | "
                  f"Speedup: {speedup_str:>8}")
    return sparse_results


def save_computation_time_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios):
    ensure_output_dir()
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    dense_times = []
    dense_sizes = []
    for size in matrix_sizes:
        if not dense_results[size]['failed']:
            dense_times.append(dense_results[size]['mean']*1000)
            dense_sizes.append(size)
    if dense_times:
        ax1.plot(dense_sizes, dense_times, 'o-', label='Dense', alpha=0.7, linewidth=2, markersize=8, color='black')
    for nnz_ratio in nnz_ratios:
        sparse_times = []
        sizes = []
        for size in matrix_sizes:
            if nnz_ratio in sparse_results[size] and not sparse_results[size][nnz_ratio]['failed']:
                sparse_times.append(sparse_results[size][nnz_ratio]['mean']*1000)
                sizes.append(size)
        if sparse_times:
            ax1.plot(sizes, sparse_times, 's--', label=f'Sparse {nnz_ratio:.1%}', alpha=0.7)
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Computation Time vs Matrix Size')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'computation_time_vs_size.eps')
    fig1.savefig(filepath, format='eps', bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {filepath}")


def save_speedup_vs_sparsity_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios):
    ensure_output_dir()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for size in matrix_sizes:
        if dense_results[size]['failed']:
            continue
        speedups = []
        nnz_vals = []
        dense_time = dense_results[size]['mean']
        for nnz in nnz_ratios:
            if nnz in sparse_results[size] and not sparse_results[size][nnz]['failed']:
                speedup = dense_time / sparse_results[size][nnz]['mean']
                speedups.append(speedup)
                nnz_vals.append(nnz)
        if speedups:
            ax2.plot(nnz_vals, speedups, 'o-', label=f'{size}x{size}', alpha=0.7)
    ax2.set_xlabel('NNZ %')
    ax2.set_ylabel('Speedup (Dense/Sparse)')
    ax2.set_title('Speedup vs Sparsity Level')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'speedup_vs_sparsity.eps')
    fig2.savefig(filepath, format='eps', bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {filepath}")


def save_speedup_heatmap(dense_results, sparse_results, matrix_sizes, nnz_ratios):
    ensure_output_dir()
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    speedup_matrix = np.full((len(matrix_sizes), len(nnz_ratios)), np.nan)
    for i, size in enumerate(matrix_sizes):
        if dense_results[size]['failed']:
            continue
        dense_time = dense_results[size]['mean']
        for j, nnz in enumerate(nnz_ratios):
            if nnz in sparse_results[size] and not sparse_results[size][nnz]['failed']:
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
    ax3.set_title('Speedup Heatmap (Dense/Sparse)\nRed: Sparse slower, White: equal (1×), Blue: Sparse faster')

    cbar = plt.colorbar(im, ax=ax3, label='Speedup')
    cbar_ticks = [vmin, 1.0, vmax]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{vmin:.2f}", "1.00", f"{vmax:.2f}"])
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'speedup_heatmap.eps')
    fig3.savefig(filepath, format='eps', bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {filepath}")


def save_sparse_performance_plot(sparse_results, matrix_sizes):
    ensure_output_dir()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    for nnz_ratio in [0.001, 0.01, 0.1, 0.5]:
        sparse_times = []
        sizes_list = []
        for size in matrix_sizes:
            if nnz_ratio in sparse_results[size] and not sparse_results[size][nnz_ratio]['failed']:
                sparse_times.append(sparse_results[size][nnz_ratio]['mean']*1000)
                sizes_list.append(size)
        if sparse_times:
            ax4.plot(sizes_list, sparse_times, 'o-', label=f'Sparse {nnz_ratio:.1%}', alpha=0.7)
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Sparse Matmul Time (ms)')
    ax4.set_title('Sparse Performance (All Successful Operations)')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'sparse_performance.eps')
    fig4.savefig(filepath, format='eps', bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {filepath}")


def print_summary(device):
    print("\n" + "="*80)
    print("Summary:")
    print(f"- Device used: {device}")
    print(f"- Dense operations benchmarked once per matrix size")
    print(f"- Sparse matmul is generally faster when sparsity is high (low nnz ratio)")
    print(f"- Dense operations may fail due to memory constraints on larger matrices")
    print(f"- Sparse operations can handle much larger matrices within memory limits")
    print(f"- All output files saved to: {OUTPUT_DIR}/")
    print(f"  EPS files:")
    print(f"    - computation_time_vs_size.eps")
    print(f"    - speedup_vs_sparsity.eps")
    print(f"    - speedup_heatmap.eps")
    print(f"    - sparse_performance.eps")
    print(f"  CSV files:")
    print(f"    - dense_results.csv")
    print(f"    - sparse_results.csv")
    print(f"    - benchmark_summary.csv")


def main():
    device = get_device()
    matrix_sizes = [2**i for i in range(7, 15)]
    nnz_ratios = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1]

    print_header(device)

    dense_results = run_dense_benchmarks(matrix_sizes, device)
    sparse_results = run_sparse_benchmarks(matrix_sizes, nnz_ratios, dense_results, device)

    print("\n" + "="*80)
    print("Saving results to CSV files...")
    save_results_to_csv(
        dense_results, sparse_results, matrix_sizes, nnz_ratios, device
    )

    save_computation_time_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios)
    save_speedup_vs_sparsity_plot(dense_results, sparse_results, matrix_sizes, nnz_ratios)
    save_speedup_heatmap(dense_results, sparse_results, matrix_sizes, nnz_ratios)
    save_sparse_performance_plot(sparse_results, matrix_sizes)
    print_summary(device)


if __name__ == '__main__':
    main()