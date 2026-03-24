"""
Multi-Dataset Pipeline for VulnSage.

Unified data-loading interface for three heterogeneous benchmarks:

* **Dataset 1 -- ESC** (Zhuang et al., IJCAI 2020): 9 742 contracts,
  contract-level, 4 vulnerability types.
* **Dataset 2 -- SMS** (Qian et al., WWW 2023): 42 910 contracts / 514 880
  functions, function-level, 4 types.
* **Dataset 3 -- DAppSCAN** (Zheng et al., IEEE TSE 2024): 682 DApps /
  6 665 compiled contracts, file-level, 37 SWC types.

All loaders share ``UnifiedDataset`` which integrates pre-computed
``AnalysisEvidence`` objects from the program-analysis pipeline.

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import json, logging, os, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from .analyzer import AnalysisEvidence, ProgramAnalysisPipeline

logger = logging.getLogger(__name__)

# ─────────────────────── dataset-specific loaders ───────────────────────────


class Dataset1_ESC:
    """
    ESC / EFEVD dataset (Zhuang et al., IJCAI 2020; Jiang et al., IJCAI 2024).

    Files: graph_feature.txt, graph_index.txt, graph_edge.txt
    9 742 contract samples -- 4 types: reentrancy, timestamp, overflow, safe.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.contracts: Optional[List[str]] = None
        self.source_codes: Optional[List[str]] = None

    def load(self) -> "Dataset1_ESC":
        logger.info("Loading Dataset 1 (ESC)...")
        feat_path = self.data_dir / "graph_feature.txt"
        idx_path = self.data_dir / "graph_index.txt"

        if not feat_path.exists():
            raise FileNotFoundError(f"ESC feature file missing: {feat_path}")
        if not idx_path.exists():
            raise FileNotFoundError(f"ESC index file missing: {idx_path}")

        rows = []
        with open(feat_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([float(x) for x in line.split()])
        self.features = np.array(rows, dtype=np.float32)

        contracts, labels = [], []
        with open(idx_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    contracts.append(parts[0])
                    labels.append(int(parts[1]))
        self.labels = np.array(labels, dtype=np.int64)
        self.contracts = contracts

        src_dir = self.data_dir / "source_code"
        if src_dir.is_dir():
            self.source_codes = []
            for cid in contracts:
                fp = src_dir / f"{cid}.sol"
                self.source_codes.append(fp.read_text() if fp.exists() else "")
        else:
            self.source_codes = [""] * len(contracts)

        logger.info(f"  ESC loaded: {len(self.labels)} instances, "
                    f"{self.features.shape[1]} features")
        return self


class Dataset2_SMS:
    """
    SMS dataset (Qian et al., WWW 2023).

    42 910 contracts / 514 880 functions, 4 vulnerability types.
    Organised per-vulnerability with source code and bytecode.

    FIX: Label assignment now uses explicit label files or directory
    structure (safe/ vs vulnerable/) rather than fragile substring matching.
    """

    VULN_TYPES = ["reentrancy", "timestamp", "overflow", "delegatecall"]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: Dict[str, List[Dict[str, Any]]] = {}

    def load(self) -> "Dataset2_SMS":
        logger.info("Loading Dataset 2 (SMS)...")
        for vtype in self.VULN_TYPES:
            vdir = self.data_dir / vtype
            if not vdir.is_dir():
                vdir = self.data_dir / "source_code" / vtype
            if not vdir.is_dir():
                logger.warning(f"  SMS sub-dir not found: {vdir}")
                continue
            samples = []
            for sol_file in sorted(vdir.rglob("*.sol")):
                try:
                    code = sol_file.read_text(errors="replace")
                    # FIX: Robust label assignment using directory structure
                    lbl = self._determine_label(sol_file, vtype)
                    samples.append({
                        "source_code": code,
                        "contract_id": sol_file.stem,
                        "label": lbl,
                        "vulnerability_type": vtype,
                        "file_path": str(sol_file),
                    })
                except Exception as exc:
                    logger.debug(f"  skip {sol_file}: {exc}")
            self.samples[vtype] = samples
            vuln_count = sum(1 for s in samples if s["label"] == 1)
            logger.info(f"  SMS/{vtype}: {len(samples)} samples "
                        f"({vuln_count} vulnerable, {len(samples)-vuln_count} safe)")
        return self

    @staticmethod
    def _determine_label(sol_file: Path, vtype: str) -> int:
        """
        Determine label from directory structure.

        FIX: Instead of fragile substring matching on 'vulnerability' or 'vuln',
        we use the established SMS directory convention:
          - Files under 'safe/' or 'benign/' directories are label 0
          - Files under 'vulnerable/' or the vulnerability type name are label 1
          - If a label file (.json or .txt) exists alongside, use that
        """
        # Check for explicit label file
        label_file = sol_file.with_suffix(".json")
        if label_file.exists():
            try:
                with open(label_file) as f:
                    meta = json.load(f)
                return int(meta.get("label", meta.get("vulnerable", 0)))
            except Exception:
                pass

        # Use directory naming convention
        path_parts = [p.lower() for p in sol_file.parts]
        # Explicit safe directories
        if any(p in ("safe", "benign", "non_vulnerable", "nonvulnerable", "negative")
               for p in path_parts):
            return 0
        # Explicit vulnerable directories
        if any(p in ("vulnerable", "positive", "malicious") for p in path_parts):
            return 1
        # If inside a vuln-type directory with sub-dirs safe/vuln
        for i, p in enumerate(path_parts):
            if p == vtype and i + 1 < len(path_parts):
                next_dir = path_parts[i + 1]
                if next_dir in ("safe", "benign", "negative"):
                    return 0
                if next_dir in ("vulnerable", "positive"):
                    return 1
        # Default: if the file is directly under the vulnerability type dir,
        # it's assumed vulnerable per SMS convention
        if vtype in path_parts:
            return 1
        return 0


class Dataset3_DAppSCAN:
    """
    DAppSCAN dataset (Zheng et al., IEEE TSE 2024).

    DAPPSCAN-SOURCE : 39 904 Solidity files, 1 618 SWC weaknesses, 682 DApps.
    DAPPSCAN-BYTECODE : 6 665 compiled contracts, 888 SWC weaknesses.
    """

    def __init__(self, source_dir: str, bytecode_dir: str):
        self.source_dir = Path(source_dir)
        self.bytecode_dir = Path(bytecode_dir)
        self.samples: List[Dict[str, Any]] = []
        self.swc_counts: Dict[str, int] = defaultdict(int)

    def load(self) -> "Dataset3_DAppSCAN":
        logger.info("Loading Dataset 3 (DAppSCAN)...")
        json_files = sorted(self.source_dir.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    report = json.load(f)
                sol_path = report.get("filePath", "")
                swcs = report.get("SWCs", [])
                if not swcs:
                    continue
                full_sol = self.source_dir / sol_path
                code = full_sol.read_text(errors="replace") if full_sol.exists() else ""
                for swc in swcs:
                    cat = swc.get("category", "Unknown")
                    fn = swc.get("function", "N/A")
                    ln = swc.get("lineNumber", "")
                    swc_id = re.search(r'SWC-(\d+)', cat)
                    swc_num = int(swc_id.group(1)) if swc_id else -1
                    self.samples.append({
                        "source_code": code,
                        "contract_id": Path(sol_path).stem,
                        "label": 1,
                        "vulnerability_type": cat,
                        "swc_id": swc_num,
                        "function": fn,
                        "line_number": ln,
                        "dapp_path": str(jf.parent),
                    })
                    self.swc_counts[cat] += 1
            except Exception as exc:
                logger.debug(f"  skip {jf}: {exc}")

        # Safe files (no SWC reports)
        reported_files = {s["dapp_path"] + "/" + s["contract_id"] for s in self.samples}
        for sol in sorted(self.source_dir.rglob("*.sol"))[:2000]:
            key = str(sol.parent) + "/" + sol.stem
            if key not in reported_files:
                try:
                    code = sol.read_text(errors="replace")
                    self.samples.append({
                        "source_code": code,
                        "contract_id": sol.stem,
                        "label": 0,
                        "vulnerability_type": "safe",
                        "swc_id": 0,
                    })
                except Exception:
                    pass

        logger.info(f"  DAppSCAN loaded: {len(self.samples)} samples, "
                    f"{sum(1 for s in self.samples if s['label']==1)} vulnerable")
        return self


# ─────────────────────── unified dataset ────────────────────────────────────


class UnifiedDataset(Dataset):
    """
    PyTorch Dataset wrapping any of the three benchmarks.

    Each item returns::
        {
            "source_code":      str,
            "evidence_text":    str,
            "evidence_quality": Tensor,
            "label":            int,
            "vulnerability_type": str,
            "contract_id":      str,
            "dataset_source":   str,
        }
    """

    def __init__(self, samples: List[Dict[str, Any]],
                 pipeline: Optional[ProgramAnalysisPipeline] = None,
                 dataset_source: str = ""):
        self.samples = samples
        self.pipeline = pipeline
        self.dataset_source = dataset_source
        self._evidence_cache: Dict[int, AnalysisEvidence] = {}

    def precompute_evidence(self) -> None:
        """Run program analysis on all samples (uses disk cache internally)."""
        if self.pipeline is None:
            return
        logger.info(f"Pre-computing evidence for {len(self.samples)} samples...")
        for i, s in enumerate(self.samples):
            if s.get("source_code"):
                ev = self.pipeline.analyze(s["source_code"], s.get("contract_id", str(i)))
                self._evidence_cache[i] = ev
            if (i + 1) % 500 == 0:
                logger.info(f"  evidence: {i+1}/{len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        if idx in self._evidence_cache:
            ev = self._evidence_cache[idx]
            ev_text = self.pipeline.format_evidence_prompt(ev) if self.pipeline else ""
            ev_quality = self.pipeline.evidence_quality(ev) if self.pipeline else [0.]*4
        else:
            ev_text = ""
            ev_quality = [0.0] * 4

        return {
            "source_code": s.get("source_code", "")[:16000],
            "evidence_text": ev_text,
            "evidence_quality": torch.tensor(ev_quality, dtype=torch.float32),
            "label": int(s.get("label", 0)),
            "vulnerability_type": s.get("vulnerability_type", "unknown"),
            "contract_id": s.get("contract_id", ""),
            "dataset_source": self.dataset_source,
        }


# ─────────────────────── splitting ──────────────────────────────────────────


class ContractLevelSplitter:
    """Stratified contract-level splitting to prevent data leakage."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def split(self, samples: List[Dict], train_r: float, val_r: float,
              test_r: float) -> Tuple[List[int], List[int], List[int]]:
        contract_map: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            contract_map[s.get("contract_id", str(i))].append(i)

        contracts = list(contract_map.keys())
        labels = []
        for c in contracts:
            idxs = contract_map[c]
            lbl = max(samples[i]["label"] for i in idxs)
            labels.append(lbl)
        labels_arr = np.array(labels)

        n = len(contracts)
        perm = np.random.permutation(n)

        test_idx, trainval_idx = self._stratified_split(perm, labels_arr, test_r)
        if val_r > 0:
            adjusted_val_r = val_r / (1 - test_r) if test_r < 1 else 0.5
            train_idx, val_idx = self._stratified_split(trainval_idx, labels_arr[trainval_idx], adjusted_val_r)
        else:
            train_idx, val_idx = trainval_idx, np.array([], dtype=int)

        to_sample_idx = lambda cidxs: [i for ci in cidxs for i in contract_map[contracts[ci]]]
        return to_sample_idx(train_idx), to_sample_idx(val_idx), to_sample_idx(test_idx)

    @staticmethod
    def _stratified_split(indices, labels, ratio):
        if len(indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        c0 = [i for i in indices if labels[i] == 0]
        c1 = [i for i in indices if labels[i] == 1]
        np.random.shuffle(c0)
        np.random.shuffle(c1)
        n0 = max(1, int(len(c0) * ratio)) if c0 else 0
        n1 = max(1, int(len(c1) * ratio)) if c1 else 0
        split_a = list(c0[n0:]) + list(c1[n1:])
        split_b = list(c0[:n0]) + list(c1[:n1])
        return np.array(split_a, dtype=int), np.array(split_b, dtype=int)

    def create_cv_folds(self, samples: List[Dict], n_folds: int = 5
                        ) -> List[Tuple[List[int], List[int]]]:
        contract_map: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            contract_map[s.get("contract_id", str(i))].append(i)
        contracts = list(contract_map.keys())
        labels = np.array([max(samples[i]["label"] for i in contract_map[c]) for c in contracts])

        # Ensure at least n_folds samples per class
        if min(np.sum(labels == 0), np.sum(labels == 1)) < n_folds:
            logger.warning(f"Too few samples for {n_folds}-fold CV; reducing folds")
            n_folds = min(n_folds, min(np.sum(labels == 0), np.sum(labels == 1)))
            n_folds = max(2, n_folds)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        folds = []
        for train_ci, val_ci in skf.split(contracts, labels):
            train_idx = [i for ci in train_ci for i in contract_map[contracts[ci]]]
            val_idx = [i for ci in val_ci for i in contract_map[contracts[ci]]]
            folds.append((train_idx, val_idx))
        return folds


# ─────────────────────── collation ──────────────────────────────────────────

def vulnsage_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate for variable-length text + tensor fields."""
    return {
        "source_code": [b["source_code"] for b in batch],
        "evidence_text": [b["evidence_text"] for b in batch],
        "evidence_quality": torch.stack([b["evidence_quality"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "vulnerability_type": [b["vulnerability_type"] for b in batch],
        "contract_id": [b["contract_id"] for b in batch],
        "dataset_source": [b["dataset_source"] for b in batch],
    }


# ─────────────────────── factory ────────────────────────────────────────────

def _load_samples(config, dataset_name: str) -> List[Dict[str, Any]]:
    """Load raw samples for a given dataset (reusable for CV)."""
    if dataset_name == "esc":
        ds = Dataset1_ESC(config.data.esc_data_dir).load()
        return [{"source_code": ds.source_codes[i] if ds.source_codes else "",
                 "contract_id": ds.contracts[i],
                 "label": int(ds.labels[i]),
                 "vulnerability_type": "binary"} for i in range(len(ds.labels))]
    elif dataset_name == "sms":
        ds2 = Dataset2_SMS(config.data.sms_data_dir).load()
        samples = []
        for vtype, slist in ds2.samples.items():
            samples.extend(slist)
        return samples
    elif dataset_name == "dappscan":
        ds3 = Dataset3_DAppSCAN(config.data.dappscan_source_dir,
                                config.data.dappscan_bytecode_dir).load()
        return ds3.samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders(config, pipeline: Optional[ProgramAnalysisPipeline] = None,
                       dataset_name: Optional[str] = None
                       ) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoader objects for the requested dataset."""
    splitter = ContractLevelSplitter(config.reproducibility.seed)
    loaders: Dict[str, DataLoader] = {}

    target = dataset_name or config.data.active_datasets[0]
    samples = _load_samples(config, target)

    if target == "esc":
        tr, va, te = splitter.split(samples, config.data.esc_train_split,
                                     config.data.esc_val_split, config.data.esc_test_split)
    elif target == "sms":
        tr, va, te = splitter.split(samples, config.data.sms_train_split,
                                     0.0, config.data.sms_test_split)
        # Carve 10% of train for val
        va = tr[:len(tr) // 10]
        tr = tr[len(tr) // 10:]
    elif target == "dappscan":
        tr, va, te = splitter.split(samples, config.data.dappscan_train_split,
                                     config.data.dappscan_val_split,
                                     config.data.dappscan_test_split)
    else:
        raise ValueError(f"Unknown dataset: {target}")

    for split_name, indices in [("train", tr), ("val", va), ("test", te)]:
        split_samples = [samples[i] for i in indices]
        uds = UnifiedDataset(split_samples, pipeline, dataset_source=target)
        if pipeline and split_name == "train":
            uds.precompute_evidence()
        loaders[split_name] = DataLoader(
            uds, batch_size=config.training.batch_size,
            shuffle=(split_name == "train"),
            num_workers=min(config.data.num_workers, 4),
            collate_fn=vulnsage_collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == "train"))

    for k, v in loaders.items():
        logger.info(f"  {k}: {len(v.dataset)} samples, {len(v)} batches")
    return loaders
