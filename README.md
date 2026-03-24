# VulnSage: Program Analysis-Guided LLM Reasoning for Smart Contract Vulnerability Detection

**Artifact for IEEE/ACM ASE 2026 Submission**

---

## Overview

VulnSage is a dual-pathway architecture for smart contract vulnerability detection that fuses structured program analysis evidence with LoRA-adapted LLM reasoning. The system extracts four evidence types from Solidity source code (control-flow graphs, abstract syntax trees, taint propagation paths, and inter-procedural call graphs), encodes them through quality-aware cross-evidence attention, and fuses the result with a code LLM via learned gated fusion.

This repository contains the complete source code, configuration, and experiment scripts needed to reproduce all results reported in the paper (Tables 1–6, Figures 1–5, RQ1–RQ5).

---

## Repository Structure

```
vulnsage/
├── __init__.py          # Package definition and public API
├── analyzer.py          # Program analysis evidence extraction (CFG, AST, Taint, Call Graph)
├── config.py            # Hierarchical configuration with validation and ablation generation
├── data.py              # Multi-dataset pipeline (ESC, SMS, DAppSCAN) with contract-level splitting
├── engine.py            # Training loop, evaluation, metrics, early stopping
├── experiments.py       # Experiment orchestration mapped to RQ1–RQ5 (CLI entry point)
├── model.py             # VulnSage architecture (evidence encoder, LLM module, fusion, classifier)
├── utils.py             # Statistical tests, visualisation, LaTeX table generation
└── requirements.txt     # Pinned dependencies
```

---

## Requirements

- **Hardware:** 4× NVIDIA RTX 3090 (24 GB each), 128 GB RAM, 64-core CPU
- **Software:** Python 3.10, CUDA 11.8, Linux (tested on CentOS 7)
- **GPU Memory:** 17.2 GB minimum for single-GPU inference; 4× GPUs recommended for full training
- **Disk Space:** ~50 GB (datasets + evidence cache + model checkpoints)

> **Reduced hardware:** Single-GPU training is possible with `gradient_accumulation_steps: 8` and `batch_size: 4` in the config. Inference (evaluation only) requires one GPU with ≥18 GB.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/anonymous/vulnsage.git
cd vulnsage

# 2. Create a virtual environment
conda create -n vulnsage python=3.10 -y
conda activate vulnsage

# 3. Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install Solidity compiler manager
solc-select install 0.4.26 0.5.17 0.6.12 0.7.6 0.8.19 0.8.24
solc-select use 0.8.19

# 6. Verify installation
python -c "from src import VulnSage, Config; print('Installation successful')"
```

---

## Dataset Preparation

Download the three evaluation datasets and place them under `data/`:

| Dataset  | Source | Target Directory |
|----------|--------|------------------|
| ESC      | [Zhuang et al., IJCAI 2020](https://github.com/Messi-Q/Smart-Contract-Dataset) | `data/dataset1_esc/raw/` |
| SMS      | [Qian et al., WWW 2023](https://github.com/Messi-Q/Smart-Contract-Dataset) | `data/dataset2_sms/raw/` |
| DAppSCAN | [Zheng et al., IEEE TSE 2024](https://github.com/InPlusLab/DAppSCAN) | `data/dataset3_dappscan/` |

Expected directory layout after download:

```
data/
├── dataset1_esc/
│   └── raw/
│       ├── graph_feature.txt
│       ├── graph_index.txt
│       └── source_code/           # .sol files (one per contract)
├── dataset2_sms/
│   └── raw/
│       ├── reentrancy/
│       ├── timestamp/
│       ├── overflow/
│       └── delegatecall/
└── dataset3_dappscan/
    ├── source/                    # Solidity source files by DApp
    └── bytecode/                  # Compiled contracts
```

---

## Quick Start

### Train on a single dataset

```bash
python -m src.experiments --mode train --dataset esc --gpu 0
```

### Run all experiments (RQ1–RQ5)

```bash
python -m src.experiments --mode all --gpu 0,1,2,3 --seed 42
```

This executes the full pipeline: evidence extraction, training on all three datasets, 5-fold cross-validation, ablation study (7 variants), explainability analysis, efficiency measurement, and cross-dataset transfer (6 pairs). Total runtime is approximately 282 GPU-hours on 4× RTX 3090.

---

## Reproducing Paper Results

Each research question can be reproduced independently. All commands below assume GPUs 0–3 are available; adjust `--gpu` as needed.

### Step 0: Pre-compute Evidence Cache (one-time)

```bash
python -m src.experiments --mode extract_evidence --gpu 0
```

This runs Slither-backed program analysis on all contracts across all three datasets and caches the results to `data/evidence_cache/`. Runtime: ~14.8 hours on 64 cores. Once cached, evidence extraction is skipped in subsequent runs.

### RQ1 — Effectiveness (Tables 2–3 in paper)

```bash
# Train and evaluate on each dataset
python -m src.experiments --mode train --dataset esc --gpu 0,1,2,3
python -m src.experiments --mode train --dataset sms --gpu 0,1,2,3
python -m src.experiments --mode train --dataset dappscan --gpu 0,1,2,3

# 5-fold cross-validation
python -m src.experiments --mode cross_validation --dataset esc --folds 5 --gpu 0,1,2,3
python -m src.experiments --mode cross_validation --dataset sms --folds 5 --gpu 0,1,2,3
python -m src.experiments --mode cross_validation --dataset dappscan --folds 5 --gpu 0,1,2,3
```

Results are written to `results/metrics/rq1_*.json` and `results/metrics/cv_*_results.json`.

### RQ2 — Ablation Study (Table 4 in paper)

```bash
python -m src.experiments --mode ablation --dataset esc --gpu 0,1,2,3
```

Trains 8 model variants (full model + 7 ablations: w/o CFG, w/o AST, w/o Taint, w/o Call Graph, w/o LLM, w/o Program Analysis, w/o Cross-Evidence Attention). Results: `results/metrics/ablation_results.json`.

### RQ3 — Explainability (Figure 5 in paper)

```bash
python -m src.experiments --mode explainability --dataset esc --gpu 0
```

Extracts cross-evidence attention weights, fusion gate values, and LLM-generated explanations for 20 sampled contracts. Results: `results/metrics/explainability.json`.

### RQ4 — Efficiency (Table 5 in paper)

```bash
python -m src.experiments --mode efficiency --dataset esc --gpu 0
```

Measures per-contract inference latency, evidence extraction time, and peak GPU memory. Results: `results/metrics/efficiency_results.json`.

### RQ5 — Cross-Dataset Transfer (Table 6 in paper)

```bash
python -m src.experiments --mode transfer --gpu 0,1,2,3
```

Trains on each dataset and evaluates on the other two (6 transfer pairs, no fine-tuning on target). Results: `results/metrics/transfer_results.json`.

### Generate All Figures

```bash
python -m src.experiments --mode generate_figures
```

Produces publication-quality PDF figures from saved metrics in `results/figures/`.

---

## Configuration

The default configuration is defined in `src/config.py` and can be overridden via YAML:

```bash
python -m src.experiments --mode train --config configs/custom.yaml --dataset esc
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.llm_model_name` | `deepseek-ai/deepseek-coder-6.7b-instruct` | Backbone LLM |
| `model.lora_rank` | 16 | LoRA rank |
| `model.lora_alpha` | 32 | LoRA scaling factor |
| `model.evidence_encoder_dim` | 768 | Evidence encoder hidden dimension |
| `model.cross_evidence_layers` | 3 | Number of cross-evidence attention layers |
| `training.learning_rate` | 2e-5 | Learning rate (evidence encoder + classifier) |
| `training.llm_learning_rate` | 5e-6 | Learning rate (LoRA parameters) |
| `training.batch_size` | 8 | Per-GPU batch size |
| `training.gradient_accumulation_steps` | 4 | Effective batch size = 8 × 4 = 32 |
| `training.num_epochs` | 50 | Maximum epochs (early stopping at patience 10) |
| `training.focal_loss_gamma` | 2.0 | Focal loss focusing parameter |
| `analyzer.use_slither` | True | Use Slither as primary analysis backend |
| `reproducibility.seed` | 42 | Random seed for all components |

---

## Module Descriptions

### `analyzer.py` — Program Analysis Pipeline

Extracts four evidence types from Solidity source code:

- **CFGExtractor**: Basic blocks, edges, loop detection, critical path identification
- **ASTAnalyzer**: Function signatures, state variables, modifiers, compiler version, SafeMath usage
- **TaintAnalyzer**: Source-sink identification, propagation path tracing, sanitisation checking
- **CallGraphAnalyzer**: Internal/external call mapping, value transfers, state-after-call detection

Primary backend: Slither (87.3% success rate on ESC). Fallback: regex-based pattern matching with reduced quality scores. Results are cached to disk with SHA-256 content hashing.

### `model.py` — VulnSage Architecture

- **SingleEvidenceEncoder**: 2-layer Transformer with CLS pooling (×4 independent instances)
- **CrossEvidenceAttention**: Quality-gated multi-head attention across evidence types (×3 layers)
- **LLMReasoningModule**: DeepSeek-Coder 6.7B with 4-bit quantisation and LoRA adapters
- **EvidenceGuidedFusion**: Learned gate balancing evidence and LLM representations
- **VulnerabilityClassifier**: 2-layer MLP with focal loss

Total: ~6.76B parameters, 61.5M trainable (0.91%).

### `data.py` — Dataset Pipeline

Unified interface for three benchmarks with contract-level stratified splitting to prevent data leakage. Supports pre-computed evidence caching and custom collation for variable-length inputs.

### `engine.py` — Training and Evaluation

Mixed-precision training with gradient accumulation, dual learning rates (evidence encoder vs. LoRA), cosine warmup scheduling, and early stopping. Evaluation includes bootstrap confidence intervals, per-vulnerability-type breakdown, and ROC/PR curve data.

### `experiments.py` — Experiment Orchestration

CLI entry point mapping directly to the paper's research questions. Each experiment class (CrossDatasetExperiment, AblationExperiment, ExplainabilityExperiment, EfficiencyExperiment, TransferExperiment) is self-contained and writes results to JSON.

### `utils.py` — Utilities

Statistical tests (paired t-test, Wilcoxon, Bonferroni correction, Cohen's d), matplotlib figure generation, and LaTeX table generation for direct inclusion in the paper.

---

## Output Structure

After running all experiments:

```
results/
├── checkpoints/
│   └── best_model.pth             # Best model weights
├── metrics/
│   ├── rq1_esc.json               # RQ1 results per dataset
│   ├── rq1_sms.json
│   ├── rq1_dappscan.json
│   ├── cv_esc_results.json        # 5-fold CV results
│   ├── cv_sms_results.json
│   ├── cv_dappscan_results.json
│   ├── ablation_results.json      # RQ2 ablation
│   ├── explainability.json        # RQ3 attention + gate analysis
│   ├── efficiency_results.json    # RQ4 timing + memory
│   └── transfer_results.json      # RQ5 cross-dataset transfer
├── figures/
│   ├── training_esc.pdf           # Training curves
│   ├── ablation.pdf               # Ablation bar chart
│   ├── roc_curves.pdf             # ROC curves across datasets
│   └── evidence_attention.pdf     # Evidence attention heatmap
├── tables/                        # Auto-generated LaTeX tables
└── logs/
    └── vulnsage.log               # Full training log
```

---

## Expected Results

Results may vary slightly due to GPU non-determinism in CUDA operations, but should fall within the confidence intervals reported in the paper.

| Dataset  | F1 (%) | AUROC (%) | 5-fold std |
|----------|--------|-----------|------------|
| ESC      | 92.05  | 96.84     | ±0.67      |
| SMS      | 87.17  | 94.53     | ±0.51      |
| DAppSCAN | 74.16  | 83.71     | ±0.95      |

---

## Troubleshooting

**Slither compilation failures:** Some contracts require specific `solc` versions. Install additional versions with `solc-select install <version>`. The pipeline falls back to regex-based extraction automatically.

**Out of GPU memory:** Reduce `training.batch_size` to 4 and increase `training.gradient_accumulation_steps` to 8. For inference only, a single 24 GB GPU is sufficient.

**Missing datasets:** Ensure all three datasets are downloaded and placed in the correct directories under `data/`. See the Dataset Preparation section above.

**Reproducibility:** Set `reproducibility.deterministic: true` in the config (default). Note that full determinism requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` as an environment variable.

---

## License

This artifact is provided for review purposes in accordance with the ASE 2026 artifact evaluation guidelines. The source code will be released under an open-source license upon acceptance.

---

## Citation

If accepted, a citation entry will be provided here.
