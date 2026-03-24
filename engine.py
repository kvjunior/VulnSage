"""
Training and Evaluation Engine for VulnSage.

Implements the complete training loop with mixed-precision, gradient
accumulation across 4x RTX 3090, early stopping, and comprehensive
evaluation including bootstrap confidence intervals, per-vulnerability-type
breakdown, and per-SWC-type analysis (for DAppSCAN).

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import json, logging, time, warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# FIX: Use the non-deprecated autocast import
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE_TYPE = True  # New-style API uses device_type param
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE_TYPE = False


# ──────────────────────── metrics accumulator ───────────────────────────────

class MetricsAccumulator:
    """Collect predictions across batches and compute comprehensive metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds, self.targets, self.probs, self.losses = [], [], [], []
        self.vuln_types: List[str] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               probs: torch.Tensor, loss: float,
               vuln_types: Optional[List[str]] = None):
        self.preds.extend(preds.cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
        self.probs.extend(probs.detach().cpu().numpy())
        self.losses.append(loss)
        if vuln_types:
            self.vuln_types.extend(vuln_types)

    def compute(self) -> Dict[str, float]:
        y, yh = np.array(self.targets), np.array(self.preds)
        yp = np.array(self.probs)
        yp1 = yp[:, 1] if yp.ndim == 2 and yp.shape[1] == 2 else yp.ravel()
        m: Dict[str, float] = {
            "loss": float(np.mean(self.losses)),
            "accuracy": float(accuracy_score(y, yh)),
            "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh, zero_division=0)),
            "f1_score": float(f1_score(y, yh, zero_division=0)),
        }
        try:
            if len(np.unique(y)) > 1:
                m["auroc"] = float(roc_auc_score(y, yp1))
                m["auprc"] = float(average_precision_score(y, yp1))
            else:
                m["auroc"] = m["auprc"] = 0.0
        except Exception:
            m["auroc"] = m["auprc"] = 0.0
        cm = confusion_matrix(y, yh)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            m.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                      "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0})
        return m

    def bootstrap_ci(self, n_boot: int = 1000, ci: float = 0.95,
                     seed: int = 42) -> Dict[str, Tuple[float, float]]:
        rng = np.random.RandomState(seed)
        y, yh = np.array(self.targets), np.array(self.preds)
        n = len(y)
        results: Dict[str, List[float]] = defaultdict(list)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            results["accuracy"].append(accuracy_score(y[idx], yh[idx]))
            results["precision"].append(precision_score(y[idx], yh[idx], zero_division=0))
            results["recall"].append(recall_score(y[idx], yh[idx], zero_division=0))
            results["f1_score"].append(f1_score(y[idx], yh[idx], zero_division=0))
        alpha = 1 - ci
        cis = {}
        for k, vals in results.items():
            arr = np.array(vals)
            cis[k] = (float(np.percentile(arr, 100*alpha/2)),
                      float(np.percentile(arr, 100*(1-alpha/2))))
        return cis

    def per_type_metrics(self) -> Dict[str, Dict[str, float]]:
        if not self.vuln_types:
            return {}
        y, yh = np.array(self.targets), np.array(self.preds)
        vt = np.array(self.vuln_types)
        out = {}
        for t in sorted(set(vt)):
            mask = vt == t
            if mask.sum() == 0:
                continue
            yt, yht = y[mask], yh[mask]
            out[t] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(yt, yht)),
                "precision": float(precision_score(yt, yht, zero_division=0)),
                "recall": float(recall_score(yt, yht, zero_division=0)),
                "f1_score": float(f1_score(yt, yht, zero_division=0)),
            }
        return out

    def get_roc_data(self) -> Optional[Dict[str, Any]]:
        """Return FPR, TPR, thresholds for ROC curve plotting."""
        y = np.array(self.targets)
        yp = np.array(self.probs)
        yp1 = yp[:, 1] if yp.ndim == 2 and yp.shape[1] == 2 else yp.ravel()
        if len(np.unique(y)) < 2:
            return None
        fpr, tpr, thresholds = roc_curve(y, yp1)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()}

    def get_pr_data(self) -> Optional[Dict[str, Any]]:
        """Return precision, recall, thresholds for PR curve plotting."""
        y = np.array(self.targets)
        yp = np.array(self.probs)
        yp1 = yp[:, 1] if yp.ndim == 2 and yp.shape[1] == 2 else yp.ravel()
        if len(np.unique(y)) < 2:
            return None
        prec, rec, thresholds = precision_recall_curve(y, yp1)
        return {"precision": prec.tolist(), "recall": rec.tolist(),
                "thresholds": thresholds.tolist()}


# ──────────────────────── early stopping ────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.delta = delta
        self.better = (lambda a, b: a > b + delta) if mode == "max" else (lambda a, b: a < b - delta)
        self.counter = 0
        self.best: Optional[float] = None
        self.triggered = False

    def __call__(self, score: float) -> bool:
        if self.best is None or self.better(score, self.best):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def state_dict(self):
        return {"counter": self.counter, "best": self.best, "triggered": self.triggered}

    def load_state_dict(self, d):
        self.counter, self.best, self.triggered = d["counter"], d["best"], d["triggered"]


# ──────────────────────── trainer ───────────────────────────────────────────

class Trainer:
    """
    Full training pipeline with mixed-precision, gradient accumulation,
    separate LR for LLM vs evidence encoder, and early stopping.
    """

    def __init__(self, model: nn.Module, config, train_loader, val_loader,
                 device: torch.device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.base_model = model.module if hasattr(model, "module") else model

        if hasattr(self.base_model, "llm_module"):
            self.base_model.llm_module.load_llm(device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        use_amp = config.training.use_mixed_precision and torch.cuda.is_available()
        if use_amp:
            if _AMP_DEVICE_TYPE:
                self.scaler = GradScaler("cuda")
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        self.accum_steps = max(1, config.training.gradient_accumulation_steps)
        self.early_stop = EarlyStopping(
            config.training.early_stopping_patience,
            config.training.early_stopping_min_delta,
            config.training.early_stopping_mode)

        self.train_acc = MetricsAccumulator()
        self.val_acc = MetricsAccumulator()
        self.epoch = 0
        self.best_score = 0.0
        self.history: Dict[str, List[float]] = defaultdict(list)

    def _build_optimizer(self):
        llm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "llm" in name or "lora" in name.lower():
                llm_params.append(p)
            else:
                other_params.append(p)
        groups = [
            {"params": other_params, "lr": self.config.training.learning_rate},
            {"params": llm_params, "lr": self.config.training.llm_learning_rate},
        ]
        cfg = self.config.training
        if cfg.optimizer == "adamw":
            return AdamW(groups, betas=(cfg.adam_beta1, cfg.adam_beta2),
                         weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adam":
            return Adam(groups, betas=(cfg.adam_beta1, cfg.adam_beta2),
                        weight_decay=cfg.weight_decay)
        return SGD(groups, lr=cfg.learning_rate, momentum=0.9,
                   weight_decay=cfg.weight_decay, nesterov=True)

    def _build_scheduler(self):
        cfg = self.config.training
        total_steps = cfg.num_epochs
        warmup = max(1, int(total_steps * cfg.lr_warmup_ratio))
        warmup_sched = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer,
                                         T_max=max(1, total_steps - warmup),
                                         eta_min=cfg.min_learning_rate)
        return SequentialLR(self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

    def _autocast_context(self):
        """Return the appropriate autocast context manager."""
        if self.scaler is None:
            import contextlib
            return contextlib.nullcontext()
        if _AMP_DEVICE_TYPE:
            return autocast("cuda")
        else:
            return autocast()

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.train_acc.reset()
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}", leave=True)
        for step, batch in enumerate(pbar):
            use_amp = self.scaler is not None
            if use_amp:
                with self._autocast_context():
                    out = self.model(batch)
                    loss, ld = self.base_model.compute_loss(batch, out)
                loss = loss / self.accum_steps
                self.scaler.scale(loss).backward()
            else:
                out = self.model(batch)
                loss, ld = self.base_model.compute_loss(batch, out)
                loss = loss / self.accum_steps
                loss.backward()

            if (step + 1) % self.accum_steps == 0:
                if use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.training.gradient_clip_value)
                if use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            preds_d = self.base_model.get_predictions(out)
            self.train_acc.update(preds_d["predictions"], batch["labels"],
                                  preds_d["probabilities"], ld["total_loss"],
                                  batch.get("vulnerability_type"))
            pbar.set_postfix(loss=f'{ld["total_loss"]:.4f}')
        return self.train_acc.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.val_acc.reset()
        for batch in self.val_loader:
            out = self.model(batch)
            loss, ld = self.base_model.compute_loss(batch, out)
            preds_d = self.base_model.get_predictions(out)
            self.val_acc.update(preds_d["predictions"], batch["labels"],
                                preds_d["probabilities"], ld["total_loss"],
                                batch.get("vulnerability_type"))
        return self.val_acc.compute()

    def train(self) -> Dict[str, List[float]]:
        logger.info(f"Training for up to {self.config.training.num_epochs} epochs")
        t0 = time.time()
        for ep in range(self.config.training.num_epochs):
            self.epoch = ep
            tm = self.train_epoch()
            vm = self.validate()
            self.scheduler.step()

            for k, v in tm.items():
                self.history[f"train_{k}"].append(v)
            for k, v in vm.items():
                self.history[f"val_{k}"].append(v)

            metric = vm.get(self.config.training.early_stopping_metric, vm.get("f1_score", 0))
            if metric > self.best_score:
                self.best_score = metric
                self.save_checkpoint("best_model.pth")

            logger.info(f"Ep {ep+1}  train_f1={tm['f1_score']:.4f}  "
                        f"val_f1={vm['f1_score']:.4f}  val_auroc={vm.get('auroc',0):.4f}")

            if self.early_stop(metric):
                logger.info(f"Early stopping at epoch {ep+1}")
                break

        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed/3600:.2f}h, best={self.best_score:.4f}")
        return dict(self.history)

    def save_checkpoint(self, name: str):
        path = Path(self.config.paths.checkpoint_dir) / name
        torch.save({
            "epoch": self.epoch, "model": self.base_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "early_stop": self.early_stop.state_dict(),
            "history": dict(self.history),
        }, path)

    def load_checkpoint(self, name: str):
        path = Path(self.config.paths.checkpoint_dir) / name
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.base_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_score = ckpt["best_score"]
        self.history = defaultdict(list, ckpt.get("history", {}))
        logger.info(f"Loaded checkpoint {name} (best={self.best_score:.4f})")


# ──────────────────────── evaluator ─────────────────────────────────────────

class Evaluator:
    """Comprehensive evaluation with CIs, per-type breakdown, and curves."""

    def __init__(self, model: nn.Module, config, test_loader, device):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.device = device
        self.base_model = model.module if hasattr(model, "module") else model
        self.acc = MetricsAccumulator()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        self.acc.reset()
        for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
            out = self.model(batch)
            loss, ld = self.base_model.compute_loss(batch, out)
            preds_d = self.base_model.get_predictions(out)
            self.acc.update(preds_d["predictions"], batch["labels"],
                            preds_d["probabilities"], ld["total_loss"],
                            batch.get("vulnerability_type"))

        metrics = self.acc.compute()
        cis = self.acc.bootstrap_ci()
        per_type = self.acc.per_type_metrics()
        roc_data = self.acc.get_roc_data()
        pr_data = self.acc.get_pr_data()

        self._print_results(metrics, cis, per_type)
        return {"metrics": metrics, "confidence_intervals": cis,
                "per_type": per_type, "roc_data": roc_data, "pr_data": pr_data}

    def _print_results(self, m, ci, pt):
        print(f"\n{'='*60}\nEVALUATION RESULTS\n{'='*60}")
        for k in ("accuracy", "precision", "recall", "f1_score", "auroc"):
            lo, hi = ci.get(k, (0, 0))
            print(f"  {k:12s}: {m.get(k,0):.4f}  [{lo:.4f}, {hi:.4f}]")
        if pt:
            print(f"\nPer-type breakdown:")
            for t, tm in pt.items():
                print(f"  {t:25s}  n={tm['n']:5d}  F1={tm['f1_score']:.4f}")
        print("=" * 60)

    def save_results(self, name: str = "eval_results.json"):
        results = self.evaluate()
        path = Path(self.config.paths.metrics_dir) / name
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
        return results
