#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle

matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["axes.unicode_minus"] = True

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
fm.fontManager.addfont(FONT_PATH)
GEO_FONT = fm.FontProperties(fname=FONT_PATH).get_name()
matplotlib.rcParams["font.family"] = GEO_FONT

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "sparse_vs_dense")
IMAGES_DIR = os.path.normpath(os.path.join(BASE, "..", "images"))

DEVICES = {"cuda": "CUDA", "mps": "MPS"}

SIZE_MARKERS = ["o", "h", "p", "*", "s", "d", "^", "v"]
SPARSE_PERF_MARKERS = {0.001: "o", 0.01: "s", 0.1: "^", 0.5: "D"}

G = {
    "matrix_size": "მატრიცის ზომა",
    "sparsity_pct": "იშვიათობა %",
    "time_ms": "დრო (მწმ)",
    "dense": "მკვრივი",
    "sparse": "იშვიათი",
    "speedup": "აჩქარება",
    "speedup_ds": "აჩქარება (მკვრივი/იშვიათი)",
    "breakeven": "ტოლობა (1×)",
    "title_comptime": "გამოთვლის დრო მატრიცის ზომის მიხედვით ({d})",
    "title_speedup": "აჩქარება იშვიათობის დონის მიხედვით ({d})",
    "title_heatmap": "აჩქარების სითბური რუკა (მკვრივი/იშვიათი) - {d}",
    "heatmap_sub": "წითელი: იშვიათი უფრო ნელია, თეთრი: ტოლობა (1×), ლურჯი: იშვიათი უფრო სწრაფია",
    "title_sparseperf": "იშვიათი მატრიცის წარმადობა ({d})",
    "sparse_matmul_time": "იშვიათი მატრიცული გამრავლების დრო (მწმ)",
}


def load_dense_results(device):
    fp = os.path.join(DATA_DIR, device, "dense_results.csv")
    out = {}
    with open(fp) as f:
        for row in csv.DictReader(f):
            size = int(row["Matrix_Size"])
            mean = float(row["Mean_Time_ms"]) / 1000 if row["Mean_Time_ms"] else None
            std = float(row["Std_Time_ms"]) / 1000 if row["Std_Time_ms"] else None
            out[size] = {"mean": mean, "std": std, "failed": row["Failed"].lower() == "true"}
    return out


def load_sparse_results(device):
    fp = os.path.join(DATA_DIR, device, "sparse_results.csv")
    out = {}
    with open(fp) as f:
        for row in csv.DictReader(f):
            size = int(row["Matrix_Size"])
            nnz = float(row["NNZ_Ratio"])
            mean = float(row["Mean_Time_ms"]) / 1000 if row["Mean_Time_ms"] else None
            std = float(row["Std_Time_ms"]) / 1000 if row["Std_Time_ms"] else None
            out.setdefault(size, {})[nnz] = {
                "mean": mean, "std": std, "failed": row["Failed"].lower() == "true"}
    return out


def axes(dense, sparse):
    sizes = sorted(dense.keys())
    nnz = sorted({n for s in sparse for n in sparse[s]})
    return sizes, nnz


def save(fig, name):
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def chart_computation_time(dense, sparse, sizes, nnzs, DEV):
    fig, ax = plt.subplots(figsize=(10, 8))
    dt, ds = [], []
    for s in sizes:
        if not dense[s]["failed"]:
            dt.append(dense[s]["mean"] * 1000); ds.append(s)
    if dt:
        ax.plot(ds, dt, "o-", label=G["dense"], alpha=0.7,
                linewidth=2, markersize=8, color="black")
    for nnz in nnzs:
        st, ss = [], []
        for s in sizes:
            if nnz in sparse.get(s, {}) and not sparse[s][nnz]["failed"]:
                st.append(sparse[s][nnz]["mean"] * 1000); ss.append(s)
        if st:
            ax.plot(ss, st, "s--", label=f"{G['sparse']} {nnz:.1%}", alpha=0.7)
    ax.set_xlabel(G["matrix_size"])
    ax.set_ylabel(G["time_ms"])
    ax.set_title(G["title_comptime"].format(d=DEV))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"{DEV}_computation_time_vs_size_geo.pdf")


def chart_speedup_vs_sparsity(dense, sparse, sizes, nnzs, DEV):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, s in enumerate(sizes):
        if dense[s]["failed"]:
            continue
        sp, nv = [], []
        dtime = dense[s]["mean"]
        for nnz in nnzs:
            if nnz in sparse.get(s, {}) and not sparse[s][nnz]["failed"]:
                sp.append(dtime / sparse[s][nnz]["mean"]); nv.append(nnz)
        if sp:
            ax.plot(nv, sp, marker=SIZE_MARKERS[i % len(SIZE_MARKERS)],
                    linestyle="-", label=f"{s}x{s}", alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label=G["breakeven"])
    ax.set_xlabel(G["sparsity_pct"])
    ax.set_ylabel(G["speedup_ds"])
    ax.set_title(G["title_speedup"].format(d=DEV))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"{DEV}_speedup_vs_sparsity_geo.pdf")


def chart_heatmap(dense, sparse, sizes, nnzs, DEV):
    fig, ax = plt.subplots(figsize=(12, 8))
    M = np.full((len(sizes), len(nnzs)), np.nan)
    for i, s in enumerate(sizes):
        if dense[s]["failed"]:
            continue
        dtime = dense[s]["mean"]
        for j, nnz in enumerate(nnzs):
            if nnz in sparse.get(s, {}) and not sparse[s][nnz]["failed"]:
                M[i, j] = dtime / sparse[s][nnz]["mean"]
    masked = np.ma.masked_invalid(M)

    below = plt.cm.Reds_r(np.linspace(0.1, 1.0, 128))
    above = plt.cm.Blues(np.linspace(0.1, 1.0, 128))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "FastRedWhiteBlue", np.vstack([below, above]))

    valid = masked.compressed()
    max_dev = max(abs(valid.max() - 1), abs(valid.min() - 1)) * 0.25 if len(valid) else 1.0
    vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks(range(len(nnzs)))
    ax.set_xticklabels([f"{n:.1%}" for n in nnzs], rotation=45)
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels([str(s) for s in sizes])
    ax.set_xlabel(G["sparsity_pct"])
    ax.set_ylabel(G["matrix_size"])
    ax.set_title(G["title_heatmap"].format(d=DEV) + "\n" + G["heatmap_sub"])

    cbar = plt.colorbar(im, ax=ax, label=G["speedup"])
    cbar.set_ticks([vmin, 1.0, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", "1.00", f"{vmax:.2f}"])
    fig.tight_layout()
    save(fig, f"{DEV}_speedup_heatmap_geo.pdf")


def chart_sparse_performance(dense, sparse, sizes, nnzs, DEV):
    fig, ax = plt.subplots(figsize=(10, 8))
    for nnz in [0.001, 0.01, 0.1, 0.5]:
        st, ss = [], []
        for s in sizes:
            if nnz in sparse.get(s, {}) and not sparse[s][nnz]["failed"]:
                st.append(sparse[s][nnz]["mean"] * 1000); ss.append(s)
        if st:
            ax.plot(ss, st, marker=SPARSE_PERF_MARKERS[nnz], linestyle="-",
                    label=f"{G['sparse']} {nnz:.1%}", alpha=0.7)
    ax.set_xlabel(G["matrix_size"])
    ax.set_ylabel(G["sparse_matmul_time"])
    ax.set_title(G["title_sparseperf"].format(d=DEV))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"{DEV}_sparse_performance_geo.pdf")


def chart_schematic():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.axis("off")

    cell = 5.0
    pad = 0.45
    yc = 32.0

    ZERO_F, ZERO_E = (0.945, 0.945, 0.945), (0.62, 0.62, 0.62)
    NZ_F, NZ_E = (0.80, 0.16, 0.16), (0.60, 0.10, 0.10)
    BLUE_E, PURP_E = (0.34, 0.45, 0.62), (0.42, 0.20, 0.52)

    def rbox(cx, cy, face, edge, lw=1.6):
        ax.add_patch(FancyBboxPatch(
            (cx - cell / 2 + pad, cy - cell / 2 + pad), cell - 2 * pad, cell - 2 * pad,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=lw, edgecolor=edge, facecolor=face, mutation_aspect=1))

    def bracket(xl, xr, yb, yt, ext=1.4, lw=3.0):
        for x, d in ((xl, ext), (xr, -ext)):
            ax.plot([x, x], [yb, yt], color="black", lw=lw, solid_capstyle="round")
            ax.plot([x, x + d], [yt, yt], color="black", lw=lw, solid_capstyle="round")
            ax.plot([x, x + d], [yb, yb], color="black", lw=lw, solid_capstyle="round")

    def grid(left, nrows, ncols, fill_fn):
        w, h = ncols * cell, nrows * cell
        top = yc + h / 2
        for i in range(nrows):
            for j in range(ncols):
                cx = left + cell / 2 + j * cell
                cy = top - cell / 2 - i * cell
                fill_fn(i, j, cx, cy)
        return left, left + w, yc - h / 2, yc + h / 2

    def title(x, y, s):
        ax.text(x, y, s, ha="center", va="center", fontsize=15, fontweight="bold")

    def dim(x, y, s):
        ax.text(x, y, s, ha="center", va="center", fontsize=12, color=(0.3, 0.3, 0.3))

    nz = {(1, 5), (1, 6), (2, 1), (2, 5), (3, 0), (3, 3), (3, 5), (4, 1)}

    def sparse_fill(i, j, cx, cy):
        if (i, j) in nz:
            shade = rng.uniform(0.0, 0.18)
            face = (NZ_F[0] + shade, NZ_F[1] + shade * 0.6, NZ_F[2] + shade * 0.6)
            rbox(cx, cy, face, NZ_E)
            ax.add_patch(Circle((cx, cy), 0.62, facecolor="white",
                                edgecolor="none", zorder=5))
        else:
            rbox(cx, cy, ZERO_F, ZERO_E, lw=1.2)

    sl, sr, sb, st_ = grid(6, 5, 7, sparse_fill)
    bracket(sl - 1.4, sr + 1.4, sb - 1.2, st_ + 1.2)
    title((sl + sr) / 2, st_ + 6.5, "იშვიათი მატრიცა")
    dim((sl + sr) / 2, sb - 4.5, "5×7")

    def dense_fill(i, j, cx, cy):
        rbox(cx, cy, plt.cm.Blues(rng.uniform(0.40, 0.82)), BLUE_E)

    dl, dr, db, dt_ = grid(54, 7, 4, dense_fill)
    bracket(dl - 1.4, dr + 1.4, db - 1.2, dt_ + 1.2)
    title((dl + dr) / 2, dt_ + 3.0, "მკვრივი მატრიცა")
    dim((dl + dr) / 2, db - 4.5, "7×4")

    def result_fill(i, j, cx, cy):
        shade = rng.uniform(0.12, 0.24) if i == 0 else rng.uniform(0.55, 0.90)
        rbox(cx, cy, plt.cm.Purples(shade), PURP_E)

    rl, rr, rb, rt_ = grid(86, 5, 4, result_fill)
    bracket(rl - 1.4, rr + 1.4, rb - 1.2, rt_ + 1.2)
    title((rl + rr) / 2, rt_ + 6.5, "შედეგის მატრიცა")
    dim((rl + rr) / 2, rb - 4.5, "5×4")

    ax.text((sr + dl) / 2 + 0.7, yc, "×", ha="center", va="center", fontsize=34)
    ax.text((dr + rl) / 2 + 0.7, yc, "=", ha="center", va="center", fontsize=34)

    ax.text(56, 58, "იშვიათი × მკვრივი მატრიცული გამრავლება",
            ha="center", va="center", fontsize=20, fontweight="bold")

    ly = 4.0
    sw = 3.2
    items = [
        (ZERO_F, ZERO_E, False, "ნულოვანი ელემენტი"),
        (NZ_F, NZ_E, True, "არანულოვანი ელემენტი"),
        (plt.cm.Blues(0.65), BLUE_E, False, "მკვრივი ელემენტი"),
        (plt.cm.Purples(0.75), PURP_E, False, "შედეგის ელემენტი"),
    ]
    xs = [4, 32, 60, 88]
    for x0, (face, edge, dot, label) in zip(xs, items):
        ax.add_patch(FancyBboxPatch(
            (x0, ly - sw / 2), sw, sw,
            boxstyle="round,pad=0,rounding_size=0.7",
            linewidth=1.6, edgecolor=edge, facecolor=face, mutation_aspect=1))
        if dot:
            ax.add_patch(Circle((x0 + sw / 2, ly), 0.45, facecolor="white",
                                edgecolor="none", zorder=5))
        ax.text(x0 + sw + 1.5, ly, label, ha="left", va="center", fontsize=13)

    fig.tight_layout()
    save(fig, "sparse_dense_matmul_geo.pdf")


def main():
    for low, DEV in DEVICES.items():
        dense = load_dense_results(low)
        sparse = load_sparse_results(low)
        sizes, nnzs = axes(dense, sparse)
        print(f"\n[{DEV}] {len(sizes)} sizes, {len(nnzs)} nnz ratios")
        chart_computation_time(dense, sparse, sizes, nnzs, DEV)
        chart_speedup_vs_sparsity(dense, sparse, sizes, nnzs, DEV)
        chart_heatmap(dense, sparse, sizes, nnzs, DEV)
        chart_sparse_performance(dense, sparse, sizes, nnzs, DEV)
    chart_schematic()
    print("\nFont used:", GEO_FONT)
    print("Done. Output dir:", IMAGES_DIR)


if __name__ == "__main__":
    main()
