"""
Configuration Management for VulnSage.

Hierarchical, type-safe configuration system with comprehensive validation,
YAML/JSON serialisation, and systematic ablation-config generation.

Mathematical context
--------------------
The configuration parameterises the full VulnSage pipeline::

    ŷ = σ(W_c · Fusion(EvidenceEnc(e), LLM_LoRA(x, prompt(e))) + b_c)

where  e = {e_cfg, e_ast, e_taint, e_cg}  are program-analysis evidence
vectors extracted from Solidity source code  x.

Author : [Anonymous for double-blind review]
Target : IEEE/ACM ASE 2026
"""

import json, logging, os, random, warnings, yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised on invalid or inconsistent configuration."""


# ───────────────────────────── section dataclasses ──────────────────────────


@dataclass
class ModelConfig:
    """Architecture hyper-parameters."""
    evidence_encoder_dim: int = 768
    num_evidence_types: int = 4
    cross_evidence_heads: int = 8
    cross_evidence_layers: int = 3
    evidence_feedforward_dim: int = 2048
    llm_model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    load_in_4bit: bool = True
    fusion_dim: int = 1024
    classifier_hidden_dim: int = 512
    num_classes: int = 2
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_evidence_quality: bool = True

    def validate(self):
        if self.evidence_encoder_dim <= 0:
            raise ConfigError("evidence_encoder_dim must be > 0")
        if self.evidence_encoder_dim % self.cross_evidence_heads != 0:
            raise ConfigError("evidence_encoder_dim must be divisible by cross_evidence_heads")
        if not 0 <= self.dropout_rate <= 1:
            raise ConfigError("dropout_rate must be in [0, 1]")
        if self.activation not in ("relu", "gelu", "elu", "tanh"):
            raise ConfigError(f"Unknown activation: {self.activation}")
        if self.lora_rank <= 0:
            raise ConfigError("lora_rank must be > 0")


@dataclass
class TrainingConfig:
    """Optimisation and scheduling."""
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 2e-5
    llm_learning_rate: float = 5e-6
    min_learning_rate: float = 1e-7
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    gradient_clip_value: float = 1.0
    lr_scheduler: str = "cosine_warmup"
    lr_warmup_ratio: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_metric: str = "f1_score"
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 1e-4
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    label_smoothing: float = 0.05
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    def validate(self):
        if self.num_epochs <= 0:
            raise ConfigError("num_epochs must be > 0")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be > 0")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ConfigError(f"Unknown optimizer: {self.optimizer}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("gradient_accumulation_steps must be >= 1")


@dataclass
class DataConfig:
    """Multi-dataset pipeline (3 heterogeneous benchmarks)."""
    active_datasets: List[str] = field(default_factory=lambda: ["esc", "sms", "dappscan"])
    esc_data_dir: str = "data/dataset1_esc/raw"
    esc_train_split: float = 0.7
    esc_val_split: float = 0.15
    esc_test_split: float = 0.15
    sms_data_dir: str = "data/dataset2_sms/raw"
    sms_train_split: float = 0.8
    sms_test_split: float = 0.2
    sms_num_runs: int = 5
    dappscan_source_dir: str = "data/dataset3_dappscan/source"
    dappscan_bytecode_dir: str = "data/dataset3_dappscan/bytecode"
    dappscan_train_split: float = 0.7
    dappscan_val_split: float = 0.15
    dappscan_test_split: float = 0.15
    contract_level_split: bool = True
    stratified_split: bool = True
    num_cv_folds: int = 5
    num_workers: int = 8
    max_source_length: int = 4096

    def validate(self):
        valid = {"esc", "sms", "dappscan"}
        for d in self.active_datasets:
            if d not in valid:
                raise ConfigError(f"Unknown dataset '{d}'; valid = {valid}")
        if not self.contract_level_split:
            warnings.warn("contract_level_split=False may cause data leakage")
        for ds_name in ["esc", "sms", "dappscan"]:
            splits = []
            if ds_name == "esc":
                splits = [self.esc_train_split, self.esc_val_split, self.esc_test_split]
            elif ds_name == "sms":
                splits = [self.sms_train_split, self.sms_test_split]
            elif ds_name == "dappscan":
                splits = [self.dappscan_train_split, self.dappscan_val_split, self.dappscan_test_split]
            if any(s < 0 or s > 1 for s in splits):
                raise ConfigError(f"Split ratios for {ds_name} must be in [0, 1]")


@dataclass
class AnalyzerConfig:
    """Program-analysis evidence extraction."""
    extract_cfg: bool = True
    extract_ast: bool = True
    extract_taint: bool = True
    extract_callgraph: bool = True
    use_slither: bool = True
    slither_path: Optional[str] = None
    solc_path: Optional[str] = None
    max_analysis_timeout: int = 120
    evidence_cache_dir: str = "data/evidence_cache"
    parallel_workers: int = 32
    max_evidence_tokens: int = 1024
    evidence_template: str = "structured"

    def validate(self):
        if self.evidence_template not in ("structured", "natural_language"):
            raise ConfigError("evidence_template must be 'structured' or 'natural_language'")
        if self.max_analysis_timeout <= 0:
            raise ConfigError("max_analysis_timeout must be > 0")


@dataclass
class PathConfig:
    project_root: str = "."
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    logs_dir: str = "results/logs"
    metrics_dir: str = "results/metrics"

    def __post_init__(self):
        root = Path(self.project_root).resolve()
        for a in ("results_dir", "checkpoint_dir", "figures_dir",
                   "tables_dir", "logs_dir", "metrics_dir"):
            setattr(self, a, root / getattr(self, a))

    def create_directories(self):
        for a in ("results_dir", "checkpoint_dir", "figures_dir",
                   "tables_dir", "logs_dir", "metrics_dir"):
            Path(getattr(self, a)).mkdir(parents=True, exist_ok=True)


@dataclass
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        logger.info(f"Seed={self.seed}, deterministic={self.deterministic}")


@dataclass
class ExperimentConfig:
    experiment_name: str = "vulnsage_ase2026"
    run_cross_validation: bool = True
    run_ablation: bool = True
    run_explainability: bool = True
    run_efficiency: bool = True
    run_transfer: bool = True
    save_best_only: bool = True
    ablation_components: List[str] = field(default_factory=lambda: [
        "cfg_evidence", "ast_evidence", "taint_evidence",
        "callgraph_evidence", "llm_reasoning", "program_analysis",
        "cross_evidence_attention"])
    # ASE 2026 Mandatory Data Availability Statement
    data_availability_doi: str = ""
    data_availability_statement: str = (
        "All datasets used are publicly available: "
        "ESC (Zhuang et al., IJCAI 2020), "
        "SMS (Qian et al., WWW 2023), "
        "DAppSCAN (Zheng et al., IEEE TSE 2024). "
        "Our code, trained models, and reproduction scripts are "
        "available at [DOI placeholder — to be replaced with Zenodo DOI]. "
        "Evidence cache and experiment logs are included in the artifact."
    )

    def validate(self):
        if not self.experiment_name.strip():
            raise ConfigError("experiment_name must not be empty")


@dataclass
class LoggingConfig:
    log_level: str = "info"
    log_interval: int = 10
    use_tensorboard: bool = True
    save_metrics_json: bool = True
    save_metrics_csv: bool = True
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "auroc", "auprc"])

    def validate(self):
        if self.log_level not in ("debug", "info", "warning", "error", "critical"):
            raise ConfigError(f"Invalid log_level: {self.log_level}")


# ───────────────────────────── master config ────────────────────────────────


class Config:
    """Unified configuration for the VulnSage framework."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        d = cfg or {}
        self.model = ModelConfig(**d.get("model", {}))
        self.training = TrainingConfig(**d.get("training", {}))
        self.data = DataConfig(**d.get("data", {}))
        self.analyzer = AnalyzerConfig(**d.get("analyzer", {}))
        self.paths = PathConfig(**d.get("paths", {}))
        self.reproducibility = ReproducibilityConfig(**d.get("reproducibility", {}))
        self.experiment = ExperimentConfig(**d.get("experiment", {}))
        self.logging_cfg = LoggingConfig(**d.get("logging", {}))
        self.creation_time = datetime.now().isoformat()

    def validate(self):
        for sec in (self.model, self.training, self.data, self.analyzer,
                    self.experiment, self.logging_cfg):
            sec.validate()

    def setup(self):
        self.validate()
        self.paths.create_directories()
        self.reproducibility.set_seed()
        self._print_summary()

    def _print_summary(self):
        dev = "CPU"
        if torch.cuda.is_available():
            dev = f"CUDA x{torch.cuda.device_count()}"
        print(f"\n{'=' * 70}\nVULNSAGE — {self.experiment.experiment_name}\n{'=' * 70}")
        print(f"  LLM       : {self.model.llm_model_name}")
        print(f"  LoRA      : r={self.model.lora_rank}, a={self.model.lora_alpha}, 4-bit={self.model.load_in_4bit}")
        print(f"  Evidence  : CFG={self.analyzer.extract_cfg} AST={self.analyzer.extract_ast} "
              f"Taint={self.analyzer.extract_taint} CG={self.analyzer.extract_callgraph}")
        print(f"  Slither   : {self.analyzer.use_slither}")
        print(f"  Datasets  : {self.data.active_datasets}")
        print(f"  Device    : {dev}   Seed: {self.reproducibility.seed}")
        print(f"{'=' * 70}\n")

    # serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model), "training": asdict(self.training),
            "data": asdict(self.data), "analyzer": asdict(self.analyzer),
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
            "reproducibility": asdict(self.reproducibility),
            "experiment": asdict(self.experiment),
            "logging": asdict(self.logging_cfg),
        }

    def save(self, fp: Union[str, Path]):
        fp = Path(fp); fp.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        with open(fp, "w") as f:
            if fp.suffix == ".json":
                json.dump(d, f, indent=2, default=str)
            else:
                yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, fp: Union[str, Path]) -> "Config":
        fp = Path(fp)
        if not fp.exists():
            raise ConfigError(f"Config not found: {fp}")
        with open(fp) as f:
            d = json.load(f) if fp.suffix == ".json" else yaml.safe_load(f)
        d.pop("metadata", None)
        return cls(d)

    def __repr__(self):
        return f"Config('{self.experiment.experiment_name}')"


# ───────────────────────── factory helpers ──────────────────────────────────

def get_default_config() -> Config:
    return Config()


def create_ablation_configs(base: Config) -> Dict[str, Config]:
    """Generate 7 ablation variants for RQ2 (§5.3)."""
    variants: Dict[str, Config] = {}
    specs = [
        ("no_cfg",              {"analyzer": {"extract_cfg": False}}),
        ("no_ast",              {"analyzer": {"extract_ast": False}}),
        ("no_taint",            {"analyzer": {"extract_taint": False}}),
        ("no_callgraph",        {"analyzer": {"extract_callgraph": False}}),
        ("no_llm",              {"model": {"llm_model_name": "none", "use_lora": False}}),
        ("no_program_analysis", {"analyzer": {"extract_cfg": False, "extract_ast": False,
                                              "extract_taint": False, "extract_callgraph": False}}),
        ("no_cross_evidence",   {"model": {"cross_evidence_layers": 0}}),
    ]
    for name, overrides in specs:
        c = Config(base.to_dict())
        c.experiment.experiment_name = f"ablation_{name}"
        for section, params in overrides.items():
            obj = getattr(c, section)
            for k, v in params.items():
                setattr(obj, k, v)
        variants[name] = c
    return variants
