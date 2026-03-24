"""
Utilities for VulnSage: visualisation, statistical tests, LaTeX tables,
logging, and reproducibility helpers.

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import json, logging, os, random, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ──────────────────── reproducibility ───────────────────────────────────────

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_system_info() -> Dict[str, Any]:
    info = {"python": sys.version, "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "platform": sys.platform}
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())]
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import slither
        info["slither"] = "available"
    except ImportError:
        info["slither"] = "not installed"
    return info


def setup_logging(config):
    level = getattr(logging, config.logging_cfg.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    log_file = Path(config.paths.logs_dir) / "vulnsage.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


# ──────────────────── statistical tests ─────────────────────────────────────

def paired_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05
                 ) -> Dict[str, Any]:
    stat, p = sp_stats.ttest_rel(a, b)
    return {"statistic": float(stat), "p_value": float(p),
            "significant": p < alpha, "mean_diff": float(np.mean(a - b))}


def wilcoxon_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05
                  ) -> Dict[str, Any]:
    try:
        stat, p = sp_stats.wilcoxon(a, b)
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    return {"statistic": float(stat), "p_value": float(p),
            "significant": p < alpha}


def bonferroni_correction(p_values: List[float], alpha: float = 0.05
                          ) -> Dict[str, Any]:
    n = len(p_values)
    corrected_alpha = alpha / n if n else alpha
    return {"corrected_alpha": corrected_alpha,
            "significant": [p < corrected_alpha for p in p_values],
            "n_significant": sum(p < corrected_alpha for p in p_values)}


def compute_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return float(diff / pooled_std) if pooled_std > 0 else 0.0


def confidence_interval(data: np.ndarray, confidence: float = 0.95
                        ) -> Tuple[float, float, float]:
    n = len(data)
    mean = float(np.mean(data))
    se = float(sp_stats.sem(data))
    margin = se * sp_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin


# ──────────────────── visualisation ─────────────────────────────────────────

def _configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                             "axes.titlesize": 12, "figure.dpi": 150})
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            pass
        return plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return None


def plot_training_curves(history: Dict[str, List[float]], save_path: Path):
    plt = _configure_matplotlib()
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pairs = [("loss", "Loss"), ("accuracy", "Accuracy"),
             ("f1_score", "F1-Score"), ("auroc", "AUROC")]
    for ax, (key, title) in zip(axes.flat, pairs):
        tr = history.get(f"train_{key}", [])
        va = history.get(f"val_{key}", [])
        if tr:
            ax.plot(range(1, len(tr)+1), tr, label="Train", lw=2)
        if va:
            ax.plot(range(1, len(va)+1), va, label="Val", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves -> {save_path}")


def plot_ablation_bars(ablation: Dict[str, Any], save_path: Path):
    plt = _configure_matplotlib()
    if plt is None:
        return
    deltas = ablation.get("deltas", {})
    if not deltas:
        return
    names = list(deltas.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score", "auroc"]
    x = np.arange(len(names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        vals = [deltas[n].get(m, 0) * 100 for n in names]
        ax.bar(x + (i - 2) * w, vals, w, label=m.replace("_", " ").title())
    ax.set_xlabel("Ablated Component"); ax.set_ylabel("Performance Drop (%)")
    ax.set_title("Ablation Study: Component Contribution (RQ2)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Ablation bars -> {save_path}")


def plot_roc_curves(results_by_dataset: Dict[str, Any], save_path: Path):
    """FIX: Plot actual ROC curves from stored FPR/TPR data."""
    plt = _configure_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for i, (ds_name, res) in enumerate(results_by_dataset.items()):
        roc_data = res.get("roc_data")
        m = res.get("metrics", {})
        auroc = m.get("auroc", 0)
        color = colors[i % len(colors)]

        if roc_data and roc_data.get("fpr") and roc_data.get("tpr"):
            fpr = np.array(roc_data["fpr"])
            tpr = np.array(roc_data["tpr"])
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{ds_name.upper()} (AUROC={auroc:.3f})")
        else:
            ax.text(0.5, 0.3 + i * 0.08, f"{ds_name}: AUROC={auroc:.3f}",
                    fontsize=10, ha="center", color=color)

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Across Datasets")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves -> {save_path}")


def plot_evidence_attention_heatmap(explainability_results: Dict[str, Any],
                                     save_path: Path):
    """Plot evidence type attention heatmap for RQ3 case studies."""
    plt = _configure_matplotlib()
    if plt is None:
        return
    agg = explainability_results.get("aggregate_evidence_attention", {})
    if not agg:
        return
    types = list(agg.keys())
    means = [agg[t]["mean"] for t in types]
    stds = [agg[t].get("std", 0) for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, means, yerr=stds, capsize=5,
                  color=["#2196F3", "#FF9800", "#F44336", "#4CAF50"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Evidence Type")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Cross-Evidence Attention Distribution (RQ3)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Evidence attention heatmap -> {save_path}")


# ──────────────────── LaTeX table generation ────────────────────────────────

def generate_main_results_table(results: Dict[str, Any], dataset: str) -> str:
    """Generate LaTeX table for RQ1 (Table 1)."""
    m = results.get("metrics", {})
    ci = results.get("confidence_intervals", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance on " + dataset.upper() + r" dataset.}",
        r"\label{tab:results_" + dataset + r"}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Accuracy & Precision & Recall & F1-Score \\",
        r"\midrule",
    ]
    # Baselines from literature
    baselines = {
        "esc": [("Slither", "---", "---", "---", "---"),
                ("Mythril", "---", "---", "---", "---"),
                ("DR-GCN", "80.08", "71.83", "79.90", "75.65"),
                ("TMP", "78.12", "77.36", "76.41", "76.52"),
                ("AME", "81.66", "80.52", "79.36", "79.94"),
                ("CGE", "82.14", "79.08", "80.22", "79.50"),
                ("SMS", "86.81", "84.31", "84.29", "84.29"),
                ("EFEVD", "89.53", "87.72", "92.82", "91.18")],
        "sms": [("TMP", "76.45", "76.04", "75.30", "75.67"),
                ("AME", "81.06", "79.62", "78.45", "79.03"),
                ("SMS/DMT", "83.85", "79.46", "77.48", "78.46")],
        "dappscan": [("Slither", "---", "---", "---", "---"),
                     ("Mythril", "---", "---", "---", "---"),
                     ("Securify", "---", "---", "---", "---")],
    }
    for entry in baselines.get(dataset, []):
        name = entry[0]
        vals = entry[1:]
        lines.append(f"  {name} & {' & '.join(vals)} \\\\")

    lines.append(r"\midrule")
    acc = f"{m.get('accuracy',0)*100:.2f}"
    pre = f"{m.get('precision',0)*100:.2f}"
    rec = f"{m.get('recall',0)*100:.2f}"
    f1 = f"{m.get('f1_score',0)*100:.2f}"
    lines.append(rf"  \textbf{{VulnSage}} & \textbf{{{acc}}} & \textbf{{{pre}}} & \textbf{{{rec}}} & \textbf{{{f1}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_ablation_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ2 (Table 3)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study results.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccl}",
        r"\toprule",
        r"Variant & Acc & Prec & Rec & F1 & $\Delta$F1 \\",
        r"\midrule",
    ]
    full = results.get("full_model", {})
    full_f1 = full.get("f1_score", 0)
    lines.append(rf"  VulnSage (full) & {full.get('accuracy',0)*100:.2f} & "
                 rf"{full.get('precision',0)*100:.2f} & {full.get('recall',0)*100:.2f} & "
                 rf"{full_f1*100:.2f} & --- \\")
    lines.append(r"\midrule")
    for name in ("no_cfg", "no_ast", "no_taint", "no_callgraph",
                 "no_llm", "no_program_analysis", "no_cross_evidence"):
        m = results.get(name, {})
        if not m:
            continue
        delta = (full_f1 - m.get("f1_score", 0)) * 100
        label = name.replace("_", " ").replace("no ", "w/o ")
        lines.append(rf"  {label} & {m.get('accuracy',0)*100:.2f} & "
                     rf"{m.get('precision',0)*100:.2f} & {m.get('recall',0)*100:.2f} & "
                     rf"{m.get('f1_score',0)*100:.2f} & -{delta:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_cross_dataset_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for RQ5 (Table 6)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset transfer results (F1-score).}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Train $\rightarrow$ Test & ESC & SMS & DAppSCAN \\",
        r"\midrule",
    ]
    datasets = ["esc", "sms", "dappscan"]
    for train_ds in datasets:
        row = [train_ds.upper()]
        for test_ds in datasets:
            if train_ds == test_ds:
                row.append("---")
            else:
                key = f"train_{train_ds}__test_{test_ds}"
                m = results.get(key, {})
                f1 = m.get("f1_score", 0) * 100
                row.append(f"{f1:.2f}")
        lines.append("  " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_efficiency_table(results: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Efficiency comparison.}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Time/contract (ms) & Evidence (ms) & GPU Mem (MB) \\",
        r"\midrule",
        rf"  VulnSage & {results.get('per_contract_ms',0):.1f} & "
        rf"{results.get('evidence_extraction_avg_ms',0):.1f} & "
        rf"{results.get('gpu_memory_mb',0):.0f} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ──────────────────── figure generation orchestrator ────────────────────────

def generate_all_figures(config):
    """Generate all publication-quality figures from saved metrics."""
    fig_dir = Path(config.paths.figures_dir)
    met_dir = Path(config.paths.metrics_dir)

    # Training curves
    for ds in config.data.active_datasets:
        fp = met_dir / f"rq1_{ds}.json"
        if fp.exists():
            with open(fp) as f:
                r = json.load(f)
            if "training_history" in r:
                plot_training_curves(r["training_history"],
                                     fig_dir / f"training_{ds}.pdf")

    # Ablation bars
    fp = met_dir / "ablation_results.json"
    if fp.exists():
        with open(fp) as f:
            r = json.load(f)
        plot_ablation_bars(r, fig_dir / "ablation.pdf")

    # ROC curves
    roc_results = {}
    for ds in config.data.active_datasets:
        fp = met_dir / f"rq1_{ds}.json"
        if fp.exists():
            with open(fp) as f:
                roc_results[ds] = json.load(f)
    if roc_results:
        plot_roc_curves(roc_results, fig_dir / "roc_curves.pdf")

    # Evidence attention heatmap
    fp = met_dir / "explainability.json"
    if fp.exists():
        with open(fp) as f:
            r = json.load(f)
        plot_evidence_attention_heatmap(r, fig_dir / "evidence_attention.pdf")

    logger.info(f"All figures saved to {fig_dir}")
