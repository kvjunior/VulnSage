"""
VulnSage: Program Analysis-Guided LLM Reasoning for
Smart Contract Vulnerability Detection.

Target: IEEE/ACM ASE 2026 (CCF-A)
"""

__version__ = "1.0.0"

from .config import Config, get_default_config, create_ablation_configs
from .analyzer import ProgramAnalysisPipeline, AnalysisEvidence
from .data import create_dataloaders, UnifiedDataset
from .model import VulnSage, create_model
from .engine import Trainer, Evaluator, MetricsAccumulator
