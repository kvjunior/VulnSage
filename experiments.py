"""
Experiment Orchestration for VulnSage (ASE 2026).

Maps directly to the paper's research questions:

* **RQ1** (S5.2) -- CrossDatasetExperiment   : effectiveness on 3 datasets
* **RQ2** (S5.3) -- AblationExperiment       : component contribution
* **RQ3** (S5.4) -- ExplainabilityExperiment : interpretability analysis
* **RQ4** (S5.5) -- EfficiencyExperiment     : runtime & resource comparison
* **RQ5** (S5.6) -- TransferExperiment       : cross-dataset generalisation

Run::

    python -m src.experiments --mode all --config configs/default.yaml --gpu 0,1,2,3

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import argparse, json, logging, os, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config, get_default_config, create_ablation_configs
from .analyzer import ProgramAnalysisPipeline
from .data import (create_dataloaders, ContractLevelSplitter, UnifiedDataset,
                   vulnsage_collate, _load_samples)
from .model import VulnSage, create_model
from .engine import Trainer, Evaluator, MetricsAccumulator
from .utils import (set_all_seeds, get_system_info, setup_logging,
                    plot_training_curves, plot_ablation_bars,
                    generate_main_results_table, generate_ablation_table,
                    compute_effect_size, paired_ttest, bonferroni_correction)

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# ────────────────────── helpers ─────────────────────────────────────────────

def _device(args) -> torch.device:
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_eval(config: Config, pipeline, dataset: str, device: torch.device,
                tag: str = "") -> Dict[str, Any]:
    """Train one model and evaluate -- reusable building block."""
    loaders = create_dataloaders(config, pipeline, dataset_name=dataset)
    model = create_model(config)

    trainer = Trainer(model, config, loaders["train"], loaders["val"], device)
    history = trainer.train()
    trainer.load_checkpoint("best_model.pth")

    evaluator = Evaluator(model, config, loaders["test"], device)
    results = evaluator.evaluate()
    results["training_history"] = history
    results["dataset"] = dataset

    fname = f"{tag}_{dataset}.json" if tag else f"{dataset}_results.json"
    out = Path(config.paths.metrics_dir) / fname
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ────────────────────── RQ1: Effectiveness ──────────────────────────────────

class CrossDatasetExperiment:
    """RQ1 (S5.2): How effective is VulnSage across three datasets?"""

    def __init__(self, config: Config, pipeline, device):
        self.config = config
        self.pipeline = pipeline
        self.device = device

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ1: Cross-Dataset Effectiveness")
        logger.info("=" * 60)
        all_results = {}
        for ds in self.config.data.active_datasets:
            logger.info(f"\n-- Dataset: {ds} --")
            r = _train_eval(self.config, self.pipeline, ds, self.device, tag="rq1")
            all_results[ds] = r
        return all_results


# ────────────────────── RQ1 supplement: Cross-Validation ────────────────────

class CrossValidationExperiment:
    """
    RQ1 supplement: k-fold CV for statistical rigor.

    FIX: Now supports all three datasets (ESC, SMS, DAppSCAN), not just ESC.
    """

    def __init__(self, config, pipeline, device, n_folds: int = 5):
        self.config = config
        self.pipeline = pipeline
        self.device = device
        self.n_folds = n_folds

    def run(self, dataset: str = "esc") -> Dict[str, Any]:
        logger.info(f"Running {self.n_folds}-fold CV on {dataset}")
        from torch.utils.data import DataLoader

        # FIX: Load samples for ANY dataset, not just ESC
        samples = _load_samples(self.config, dataset)

        splitter = ContractLevelSplitter(self.config.reproducibility.seed)
        folds = splitter.create_cv_folds(samples, self.n_folds)
        fold_metrics = []

        for fi, (tr_idx, va_idx) in enumerate(folds):
            logger.info(f"  Fold {fi+1}/{self.n_folds}")
            set_all_seeds(self.config.reproducibility.seed + fi)

            tr_samples = [samples[i] for i in tr_idx]
            va_samples = [samples[i] for i in va_idx]

            tr_ds = UnifiedDataset(tr_samples, self.pipeline, dataset)
            va_ds = UnifiedDataset(va_samples, self.pipeline, dataset)
            if self.pipeline:
                tr_ds.precompute_evidence()
            tr_loader = DataLoader(tr_ds, batch_size=self.config.training.batch_size,
                                   shuffle=True, collate_fn=vulnsage_collate)
            va_loader = DataLoader(va_ds, batch_size=self.config.training.batch_size,
                                   collate_fn=vulnsage_collate)

            model = create_model(self.config)
            trainer = Trainer(model, self.config, tr_loader, va_loader, self.device)
            trainer.train()
            trainer.load_checkpoint("best_model.pth")

            ev = Evaluator(model, self.config, va_loader, self.device)
            r = ev.evaluate()
            fold_metrics.append(r["metrics"])

        # Aggregate
        agg = {}
        for key in fold_metrics[0]:
            vals = [fm[key] for fm in fold_metrics]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "values": vals}

        result = {"n_folds": self.n_folds, "dataset": dataset,
                  "fold_metrics": fold_metrics, "aggregate": agg}
        out = Path(self.config.paths.metrics_dir) / f"cv_{dataset}_results.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result


# ────────────────────── RQ2: Ablation ───────────────────────────────────────

class AblationExperiment:
    """RQ2 (S5.3): Which components contribute most?"""

    def __init__(self, config, pipeline, device):
        self.config = config
        self.pipeline = pipeline
        self.device = device

    def run(self, dataset: str = "esc") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ2: Ablation Study")
        logger.info("=" * 60)

        full = _train_eval(self.config, self.pipeline, dataset, self.device, tag="ablation_full")

        variants = create_ablation_configs(self.config)
        ablation_results = {"full_model": full["metrics"]}
        for name, vcfg in variants.items():
            logger.info(f"\n-- Ablation: {name} --")
            pipe = ProgramAnalysisPipeline(vcfg.analyzer)
            r = _train_eval(vcfg, pipe, dataset, self.device, tag=f"ablation_{name}")
            ablation_results[name] = r["metrics"]

        # Deltas
        deltas = {}
        for name, m in ablation_results.items():
            if name == "full_model":
                continue
            deltas[name] = {k: full["metrics"].get(k, 0) - m.get(k, 0)
                           for k in ("accuracy", "precision", "recall", "f1_score", "auroc")}
        ablation_results["deltas"] = deltas

        out = Path(self.config.paths.metrics_dir) / "ablation_results.json"
        with open(out, "w") as f:
            json.dump(ablation_results, f, indent=2, default=str)
        return ablation_results


# ────────────────────── RQ3: Explainability ─────────────────────────────────

class ExplainabilityExperiment:
    """
    RQ3 (S5.4): Can VulnSage explain its detections?

    FIX: Now performs real explainability analysis:
    1. Extracts cross-evidence attention weights showing which evidence
       types the model relies on most.
    2. Extracts fusion gate values showing evidence vs LLM balance.
    3. Generates LLM-based vulnerability explanations for sampled contracts.
    4. Analyses per-type attention patterns for case studies.
    """

    def __init__(self, config, pipeline, device):
        self.config = config
        self.pipeline = pipeline
        self.device = device

    def run(self, dataset: str = "esc", n_samples: int = 20) -> Dict[str, Any]:
        logger.info("RQ3: Explainability Analysis")
        loaders = create_dataloaders(self.config, self.pipeline, dataset)
        model = create_model(self.config)

        # Load best checkpoint
        ckpt_path = Path(self.config.paths.checkpoint_dir) / "best_model.pth"
        base = model.module if hasattr(model, "module") else model
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            base.load_state_dict(ckpt["model"])
        else:
            logger.warning("No checkpoint found; using untrained model for explainability")

        model.eval()
        ds = loaders["test"].dataset
        indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)

        explanations = []
        attention_stats = {"per_type_mean": defaultdict(list),
                           "gate_mean": []}

        for idx in indices:
            sample = ds[idx]
            # Build a single-item batch
            batch = vulnsage_collate([sample])

            with torch.no_grad():
                outputs = base.forward(batch)
                preds_d = base.get_predictions(outputs)

            # 1. Cross-evidence attention weights
            attn_weights = base.get_evidence_attention_weights()
            evidence_importance = {}
            type_names = ["CFG", "AST", "Taint", "CallGraph"]
            if attn_weights is not None:
                # Average over heads: (1, n_heads, n_types, n_types) -> (n_types, n_types)
                avg_attn = attn_weights[0].mean(dim=0).cpu().numpy()
                # Per-type importance: how much attention each type receives
                incoming_attn = avg_attn.sum(axis=0)  # sum over query types
                for t_idx, t_name in enumerate(type_names):
                    evidence_importance[t_name] = float(incoming_attn[t_idx])
                    attention_stats["per_type_mean"][t_name].append(float(incoming_attn[t_idx]))

            # 2. Fusion gate values
            gate_val = None
            try:
                with torch.no_grad():
                    h_ev = outputs["h_evidence"]
                    h_llm = outputs["h_llm"]
                    g = base.fusion.gate(torch.cat([h_ev, h_llm], dim=-1))
                    gate_val = float(g.mean().cpu())
                    attention_stats["gate_mean"].append(gate_val)
            except Exception:
                pass

            # 3. LLM explanation (if available)
            llm_explanation = ""
            if hasattr(base.llm_module, 'generate_explanation') and base.llm_module._is_loaded:
                try:
                    llm_explanation = base.llm_module.generate_explanation(
                        sample["source_code"][:2000],
                        sample["evidence_text"][:1000])
                except Exception as exc:
                    llm_explanation = f"Explanation generation failed: {exc}"

            exp = {
                "contract_id": sample["contract_id"],
                "true_label": sample["label"],
                "predicted_label": int(preds_d["predictions"][0].cpu()),
                "prediction_confidence": float(preds_d["probabilities"][0].max().cpu()),
                "vulnerability_type": sample["vulnerability_type"],
                "evidence_attention": evidence_importance,
                "fusion_gate_value": gate_val,
                "evidence_summary": sample["evidence_text"][:500],
                "source_snippet": sample["source_code"][:300],
                "llm_explanation": llm_explanation[:500],
            }
            explanations.append(exp)

        # Aggregate attention statistics
        agg_attention = {}
        for t_name in type_names:
            vals = attention_stats["per_type_mean"].get(t_name, [])
            if vals:
                agg_attention[t_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        gate_vals = attention_stats["gate_mean"]
        agg_gate = {
            "mean": float(np.mean(gate_vals)) if gate_vals else 0.0,
            "std": float(np.std(gate_vals)) if gate_vals else 0.0,
            "interpretation": ("Higher values = model relies more on program analysis evidence; "
                               "lower = more on LLM reasoning"),
        }

        result = {
            "n_samples": len(explanations),
            "aggregate_evidence_attention": agg_attention,
            "aggregate_fusion_gate": agg_gate,
            "explanations": explanations,
        }

        out = Path(self.config.paths.metrics_dir) / "explainability.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Explainability results saved ({len(explanations)} samples)")
        return result


# ────────────────────── RQ4: Efficiency ─────────────────────────────────────

class EfficiencyExperiment:
    """
    RQ4 (S5.5): Runtime and resource comparison.

    FIX: Now loads the best checkpoint before measuring inference time,
    so measurements reflect the actual trained model.
    """

    def __init__(self, config, pipeline, device):
        self.config = config
        self.pipeline = pipeline
        self.device = device

    def run(self, dataset: str = "esc") -> Dict[str, Any]:
        logger.info("RQ4: Efficiency Analysis")
        loaders = create_dataloaders(self.config, self.pipeline, dataset)
        model = create_model(self.config)

        # FIX: Load best checkpoint before measuring
        ckpt_path = Path(self.config.paths.checkpoint_dir) / "best_model.pth"
        if ckpt_path.exists():
            base = model.module if hasattr(model, "module") else model
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            base.load_state_dict(ckpt["model"])
            logger.info("Loaded best checkpoint for efficiency measurement")
        else:
            logger.warning("No checkpoint found; measuring untrained model efficiency")

        # Measure evidence extraction time
        ev_times = []
        test_ds = loaders["test"].dataset
        for i in range(min(50, len(test_ds))):
            sample = test_ds.samples[i]
            if sample.get("source_code") and self.pipeline:
                t0 = time.time()
                self.pipeline.analyze(sample["source_code"], sample.get("contract_id", ""))
                ev_times.append(time.time() - t0)

        # Warmup GPU
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measure inference time
        times, n_contracts = [], 0
        with torch.no_grad():
            for batch in loaders["test"]:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = model(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
                n_contracts += len(batch["labels"])

        total_time = sum(times)
        result = {
            "n_contracts": n_contracts,
            "total_inference_s": total_time,
            "per_contract_ms": 1000 * total_time / max(1, n_contracts),
            "throughput_per_s": n_contracts / max(0.001, total_time),
            "evidence_extraction_avg_ms": 1000 * np.mean(ev_times) if ev_times else 0,
            "evidence_extraction_std_ms": 1000 * np.std(ev_times) if ev_times else 0,
            "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
            "system_info": get_system_info(),
        }
        out = Path(self.config.paths.metrics_dir) / "efficiency_results.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result


# ────────────────────── RQ5: Transfer ───────────────────────────────────────

class TransferExperiment:
    """RQ5 (S5.6): Cross-dataset generalisation."""

    def __init__(self, config, pipeline, device):
        self.config = config
        self.pipeline = pipeline
        self.device = device

    def run(self) -> Dict[str, Any]:
        logger.info("RQ5: Cross-Dataset Transfer")
        datasets = self.config.data.active_datasets
        results = {}
        for train_ds in datasets:
            for test_ds in datasets:
                if train_ds == test_ds:
                    continue
                key = f"train_{train_ds}__test_{test_ds}"
                logger.info(f"  {key}")
                train_loaders = create_dataloaders(self.config, self.pipeline, train_ds)
                model = create_model(self.config)
                trainer = Trainer(model, self.config, train_loaders["train"],
                                  train_loaders["val"], self.device)
                trainer.train()
                trainer.load_checkpoint("best_model.pth")

                test_loaders = create_dataloaders(self.config, self.pipeline, test_ds)
                ev = Evaluator(model, self.config, test_loaders["test"], self.device)
                r = ev.evaluate()
                results[key] = r["metrics"]

        out = Path(self.config.paths.metrics_dir) / "transfer_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results


# ────────────────────── evidence extraction ─────────────────────────────────

def extract_all_evidence(config: Config):
    """Pre-compute and cache evidence for all datasets."""
    logger.info("Pre-computing program analysis evidence for all datasets...")
    pipeline = ProgramAnalysisPipeline(config.analyzer)
    for ds in config.data.active_datasets:
        logger.info(f"  Extracting evidence for {ds}...")
        loaders = create_dataloaders(config, pipeline, ds)
        for split_name, loader in loaders.items():
            if hasattr(loader.dataset, "precompute_evidence"):
                loader.dataset.precompute_evidence()
    logger.info("Evidence extraction complete.")


# ────────────────────── CLI entry point ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VulnSage Experiments (ASE 2026)")
    p.add_argument("--mode", choices=["train", "evaluate", "cross_validation",
                   "ablation", "explainability", "efficiency", "transfer",
                   "extract_evidence", "generate_figures", "all"], default="train")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = get_default_config()
    if args.seed is not None:
        config.reproducibility.seed = args.seed
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.dataset:
        config.data.active_datasets = [args.dataset]

    config.setup()
    setup_logging(config)
    device = _device(args)

    pipeline = ProgramAnalysisPipeline(config.analyzer)

    print(f"\n{'='*60}")
    print(f"VulnSage -- mode={args.mode}")
    print(f"{'='*60}\n")

    t0 = time.time()

    if args.mode == "extract_evidence":
        extract_all_evidence(config)

    elif args.mode == "train":
        ds = config.data.active_datasets[0]
        _train_eval(config, pipeline, ds, device)

    elif args.mode == "cross_validation":
        ds = args.dataset or config.data.active_datasets[0]
        CrossValidationExperiment(config, pipeline, device, args.folds).run(ds)

    elif args.mode == "ablation":
        AblationExperiment(config, pipeline, device).run()

    elif args.mode == "explainability":
        ExplainabilityExperiment(config, pipeline, device).run()

    elif args.mode == "efficiency":
        EfficiencyExperiment(config, pipeline, device).run()

    elif args.mode == "transfer":
        TransferExperiment(config, pipeline, device).run()

    elif args.mode == "generate_figures":
        from .utils import generate_all_figures
        generate_all_figures(config)

    elif args.mode == "all":
        CrossDatasetExperiment(config, pipeline, device).run()
        for ds in config.data.active_datasets:
            CrossValidationExperiment(config, pipeline, device, args.folds).run(ds)
        AblationExperiment(config, pipeline, device).run()
        ExplainabilityExperiment(config, pipeline, device).run()
        EfficiencyExperiment(config, pipeline, device).run()
        if len(config.data.active_datasets) > 1:
            TransferExperiment(config, pipeline, device).run()

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/3600:.2f}h.  Results -> {config.paths.metrics_dir}")


if __name__ == "__main__":
    main()
