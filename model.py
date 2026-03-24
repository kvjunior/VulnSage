"""
VulnSage Neural Architecture.

The architecture processes Solidity source code and structured program-analysis
evidence through two complementary pathways that are fused for prediction::

    Source Code --> Program Analysis --> Evidence Encoder  -+
                                                            |-> Fusion -> y_hat
    Source Code --> LLM (LoRA) --> Code Representation  ---+

Core modules
------------
* ``EvidenceEncoder``          -- per-type encoding + cross-evidence attention
* ``CrossEvidenceAttention``   -- novel multi-head cross-attention across 4
                                 evidence types with quality-aware weighting
* ``LLMReasoningModule``       -- LoRA fine-tuned LLM conditioned on evidence
* ``EvidenceFusion``           -- gated fusion of evidence + LLM representations
* ``VulnSage``                 -- top-level model integrating all components

Mathematical formulation (section 3 of paper)
---------------------------------------------
    h_i = TransformerEnc_i(tokenise(NL(evidence_i)))     i in {cfg, ast, taint, cg}
    H   = CrossEvAttn(h_cfg, h_ast, h_taint, h_cg; q)   q = quality scores
    h_l = LLM_LoRA(x + prompt(e))
    h   = Gate(H, h_l)
    y_hat = sigma(MLP(h))

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import math, logging, warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ────────────────── evidence encoder components ─────────────────────────────


class SingleEvidenceEncoder(nn.Module):
    """Lightweight Transformer encoder for one evidence type."""

    def __init__(self, d_model: int = 768, nhead: int = 8, n_layers: int = 2,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        (batch, d_model)  -- CLS pooled representation
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])


class CrossEvidenceAttention(nn.Module):
    """
    Multi-head cross-attention across evidence types with quality gating.

    Each evidence type  h_i  attends to every other type, weighted by
    a learned quality gate  g_i  in [0, 1]:

        alpha_{ij} = softmax(Q_i K_j^T / sqrt(d_k)) * g_j
        h_i'       = sum_j alpha_{ij} V_j

    FIX: Residual connection now uses mean of ALL evidence types (not just
    the first), preventing bias toward CFG evidence.
    """

    def __init__(self, d_model: int, n_heads: int, n_types: int = 4,
                 dropout: float = 0.1, use_quality: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_types = n_types
        self.use_quality = use_quality
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if use_quality:
            self.quality_gate = nn.Sequential(
                nn.Linear(n_types, d_model),
                nn.Sigmoid())

    def forward(self, evidence_list: List[torch.Tensor],
                quality: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        evidence_list : list of (batch, d_model), length = n_types
        quality       : (batch, n_types)  -- per-evidence quality scores

        Returns
        -------
        (batch, d_model)
        """
        B = evidence_list[0].size(0)
        H = torch.stack(evidence_list, dim=1)  # (B, n_types, d_model)

        Q = self.W_q(H).view(B, self.n_types, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(H).view(B, self.n_types, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(H).view(B, self.n_types, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.use_quality and quality is not None:
            q_mask = quality.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, n_types)
            scores = scores * q_mask

        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, self.n_types, self.d_model)
        ctx = self.W_o(ctx)

        fused = ctx.mean(dim=1)  # pool across evidence types

        # FIX: residual from MEAN of all types, not just the first
        residual = torch.stack(evidence_list, dim=1).mean(dim=1)  # (B, d_model)
        fused = self.norm(fused + residual)

        # Store attention weights for explainability
        self._last_attn_weights = attn.detach()

        return fused


class EvidenceEncoder(nn.Module):
    """
    Encode all four evidence types and fuse via cross-evidence attention.

    Pipeline::
        e_i text -> tokenise -> SingleEvidenceEncoder_i -> h_i
        [h_cfg, h_ast, h_taint, h_cg] -> CrossEvidenceAttention -> h_e
    """

    def __init__(self, config):
        super().__init__()
        d = config.model.evidence_encoder_dim
        heads = config.model.cross_evidence_heads
        n_layers = config.model.cross_evidence_layers
        ff = config.model.evidence_feedforward_dim
        drop = config.model.dropout_rate
        n_types = config.model.num_evidence_types

        self.type_encoders = nn.ModuleList([
            SingleEvidenceEncoder(d, heads, 2, ff, drop)
            for _ in range(n_types)])

        self.text_embed = nn.Sequential(
            nn.Linear(256, d),
            nn.GELU(), nn.Dropout(drop))

        self.cross_attn_layers = nn.ModuleList([
            CrossEvidenceAttention(d, heads, n_types, drop,
                                   use_quality=config.model.use_evidence_quality)
            for _ in range(max(1, n_layers))])

        self.n_types = n_types
        self.d_model = d

    def _text_to_tensor(self, texts: List[str], max_len: int = 512) -> torch.Tensor:
        """Simple character-level encoding as fallback when tokeniser is absent."""
        B = len(texts)
        T = torch.zeros(B, max_len, 256)
        for i, txt in enumerate(texts):
            for j, ch in enumerate(txt[:max_len]):
                c = ord(ch) if ord(ch) < 256 else 0
                T[i, j, c] = 1.0
        return T

    def forward(self, evidence_texts: List[List[str]],
                quality: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        evidence_texts : list of 4 lists of strings (one per evidence type),
                         each inner list has batch_size strings
        quality        : (batch, 4)

        Returns
        -------
        (batch, d_model)
        """
        dev = device or next(self.parameters()).device
        type_reprs = []
        for t_idx in range(min(self.n_types, len(evidence_texts))):
            texts = evidence_texts[t_idx]
            emb = self._text_to_tensor(texts).to(dev)
            emb = self.text_embed(emb)
            h = self.type_encoders[t_idx](emb)
            type_reprs.append(h)

        while len(type_reprs) < self.n_types:
            type_reprs.append(torch.zeros_like(type_reprs[0]))

        fused = type_reprs[0]
        for layer in self.cross_attn_layers:
            fused = layer(type_reprs, quality)
        return fused


# ────────────────── LLM reasoning module ────────────────────────────────────


class LLMReasoningModule(nn.Module):
    """
    LoRA-tuned LLM that reasons about vulnerabilities given evidence.

    Supports two modes:
    * **available** -- load real HuggingFace model with LoRA adapters
    * **fallback**  -- lightweight MLP projection (for ablation / testing)

    FIX: Projection layer is now registered as a proper nn.Linear in __init__
    so it is included in the optimizer's parameter groups.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = config.model.llm_model_name
        self.d_out = config.model.evidence_encoder_dim
        self.use_lora = config.model.use_lora
        self._llm = None
        self._tokenizer = None
        self._is_loaded = False

        # FIX: Register projection as a proper submodule so it's always
        # part of the optimizer's parameter groups.
        # Common LLM hidden sizes: 4096 (7B), 2048 (3B), 768 (small)
        self.llm_proj = nn.Linear(4096, self.d_out)

        if self.model_name == "none":
            self.fallback = nn.Sequential(
                nn.Linear(config.model.evidence_encoder_dim,
                          config.model.evidence_encoder_dim),
                nn.GELU())
        else:
            self.fallback = None

    def load_llm(self, device: torch.device) -> None:
        """Lazy-load the LLM with LoRA (called once before training)."""
        if self._is_loaded or self.model_name == "none":
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info(f"Loading LLM: {self.model_name}")
            load_kw: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
            try:
                from transformers import BitsAndBytesConfig
                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4")
            except ImportError:
                logger.warning("bitsandbytes not available -- loading in fp16")
                load_kw["torch_dtype"] = torch.float16

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kw)

            if self.use_lora:
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                    lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
                self._llm = get_peft_model(self._llm, lora_cfg)
                trainable = sum(p.numel() for p in self._llm.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self._llm.parameters())
                logger.info(f"LoRA: {trainable:,} trainable / {total:,} total "
                            f"({100*trainable/total:.2f}%)")

            # FIX: Reinitialise projection to match actual LLM hidden dim
            llm_hidden = self._llm.config.hidden_size
            if llm_hidden != self.llm_proj.in_features:
                self.llm_proj = nn.Linear(llm_hidden, self.d_out).to(device)

            self._is_loaded = True
        except Exception as exc:
            logger.error(f"LLM load failed: {exc}. Using fallback MLP.")
            self.fallback = nn.Sequential(
                nn.Linear(self.d_out, self.d_out), nn.GELU())
            self.model_name = "none"

    def forward(self, source_codes: List[str],
                evidence_prompts: List[str]) -> torch.Tensor:
        """
        Returns
        -------
        (batch, d_out)  -- LLM hidden-state representation
        """
        device = next(self.parameters()).device

        if self.model_name == "none" or self._llm is None:
            B = len(source_codes)
            dummy = torch.zeros(B, self.d_out, device=device)
            if self.fallback is not None:
                dummy = self.fallback(dummy)
            return dummy

        prompts = []
        for src, ev in zip(source_codes, evidence_prompts):
            prompt = (f"Analyse the following smart contract for vulnerabilities.\n\n"
                      f"## Evidence from program analysis:\n{ev}\n\n"
                      f"## Source code:\n{src[:3000]}\n\n"
                      f"Is this contract vulnerable? Explain your reasoning.")
            prompts.append(prompt)

        enc = self._tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=2048).to(device)
        with torch.no_grad() if not self.training else torch.enable_grad():
            out = self._llm(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # (B, seq, dim)
        pooled = hidden[:, -1, :]       # last-token pooling

        # FIX: Use the properly registered projection layer
        pooled = self.llm_proj(pooled)
        return pooled

    def generate_explanation(self, source_code: str, evidence_prompt: str,
                             max_new_tokens: int = 256) -> str:
        """Generate a natural-language vulnerability explanation (for RQ3)."""
        if self._llm is None or self._tokenizer is None:
            return "LLM not loaded — cannot generate explanation."

        device = next(self.parameters()).device
        prompt = (f"You are a smart contract security auditor.\n\n"
                  f"## Program analysis evidence:\n{evidence_prompt}\n\n"
                  f"## Source code:\n{source_code[:2500]}\n\n"
                  f"Based on the evidence above, provide a detailed vulnerability "
                  f"analysis. Identify specific vulnerabilities, explain the attack "
                  f"vector, and reference the evidence that supports your finding.")

        enc = self._tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            output = self._llm.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.3, top_p=0.9)
        decoded = self._tokenizer.decode(output[0][enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        return decoded.strip()


# ────────────────── fusion & prediction ─────────────────────────────────────


class EvidenceGuidedFusion(nn.Module):
    """
    Gated fusion of evidence encoder output and LLM representation.

        g = sigma(W_g [h_e; h_l] + b_g)
        h = g * h_e + (1 - g) * h_l
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_evidence: torch.Tensor, h_llm: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([h_evidence, h_llm], dim=-1))
        h = g * h_evidence + (1 - g) * h_llm
        return self.norm(self.dropout(h))


class VulnerabilityClassifier(nn.Module):
    """Two-layer MLP classifier with focal-loss support."""

    def __init__(self, d_model: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


# ────────────────── top-level model ─────────────────────────────────────────


class VulnSage(nn.Module):
    """
    VulnSage: Program Analysis-Guided LLM Reasoning for
    Smart Contract Vulnerability Detection.

    This is the complete model described in section 3 of the paper.
    """

    # Evidence markers used in structured prompts
    EVIDENCE_MARKERS = ["[CFG]", "[AST]", "[TAINT]", "[CALL GRAPH]"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.model.evidence_encoder_dim

        # 1 Evidence pathway
        self.evidence_encoder = EvidenceEncoder(config)

        # 2 LLM pathway
        self.llm_module = LLMReasoningModule(config)

        # 3 Fusion
        self.fusion = EvidenceGuidedFusion(d, config.model.dropout_rate)

        # 4 Classifier
        self.classifier = VulnerabilityClassifier(
            d, config.model.classifier_hidden_dim,
            config.model.num_classes, config.model.dropout_rate)

        self._init_weights()
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward pass ───────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        ev_quality = batch["evidence_quality"].to(device)

        ev_texts = self._split_evidence(batch["evidence_text"])

        # 1 evidence pathway
        h_evidence = self.evidence_encoder(ev_texts, ev_quality, device)

        # 2 LLM pathway
        h_llm = self.llm_module(batch["source_code"], batch["evidence_text"])
        h_llm = h_llm.to(device)

        # 3 fusion
        h = self.fusion(h_evidence, h_llm)

        # 4 classification
        logits = self.classifier(h)

        return {"logits": logits, "h_evidence": h_evidence,
                "h_llm": h_llm, "h_fused": h}

    # ── loss computation ───────────────────────────────────────────────────

    def compute_loss(self, batch: Dict[str, Any],
                     outputs: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = outputs["logits"].device
        labels = batch["labels"].to(device)
        logits = outputs["logits"]

        gamma = self.config.training.focal_loss_gamma
        alpha = self.config.training.focal_loss_alpha

        ce = F.cross_entropy(logits, labels, reduction="none",
                             label_smoothing=self.config.training.label_smoothing)
        pt = torch.exp(-ce)
        focal = alpha * ((1 - pt) ** gamma) * ce
        loss = focal.mean()

        return loss, {"total_loss": loss.item(), "focal_loss": loss.item()}

    # ── predictions ────────────────────────────────────────────────────────

    def get_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        probs = F.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        return {"predictions": preds, "probabilities": probs}

    # ── explainability: attention weight extraction ─────────────────────────

    def get_evidence_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Extract cross-evidence attention weights for explainability (RQ3).

        Returns the last computed attention weights from the final
        cross-evidence attention layer, shape (batch, n_heads, n_types, n_types).
        """
        for layer in reversed(list(self.evidence_encoder.cross_attn_layers)):
            if hasattr(layer, '_last_attn_weights'):
                return layer._last_attn_weights
        return None

    def get_fusion_gate_values(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Extract gated fusion weights showing evidence vs LLM balance (RQ3).

        Returns gate values in [0, 1] where higher = more evidence weight.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            outputs = self.forward(batch)
            h_ev = outputs["h_evidence"]
            h_llm = outputs["h_llm"]
            g = self.fusion.gate(torch.cat([h_ev, h_llm], dim=-1))
        return g  # (batch, d_model)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_evidence(evidence_texts: List[str]) -> List[List[str]]:
        """
        Split concatenated evidence text into per-type lists.

        FIX: Uses the canonical markers from EvidenceStructurer.TEMPLATE
        and handles missing markers gracefully by assigning empty strings.
        """
        n_types = 4
        markers = ["[CFG]", "[AST]", "[TAINT]", "[CALL GRAPH]"]
        per_type: List[List[str]] = [[] for _ in range(n_types)]

        for text in evidence_texts:
            sections = [""] * n_types
            for t_idx, marker in enumerate(markers):
                start = text.find(marker)
                if start < 0:
                    continue
                end = len(text)
                for next_marker in markers[t_idx + 1:]:
                    npos = text.find(next_marker)
                    if npos > start:
                        end = npos
                        break
                sections[t_idx] = text[start:end].strip()
            # If no markers found at all, give full text to each type
            if all(s == "" for s in sections):
                sections = [text] * n_types
            for t_idx in range(n_types):
                per_type[t_idx].append(sections[t_idx])
        return per_type

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "architecture": "VulnSage",
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "size_mb": self.total_params * 4 / 1024 / 1024,
            "evidence_encoder_dim": self.config.model.evidence_encoder_dim,
            "llm": self.config.model.llm_model_name,
            "lora_rank": self.config.model.lora_rank,
            "cross_evidence_layers": self.config.model.cross_evidence_layers,
            "use_quality_gating": self.config.model.use_evidence_quality,
        }


# ────────────────── factory ─────────────────────────────────────────────────

def create_model(config) -> VulnSage:
    """Factory: create VulnSage, move to GPU, optionally wrap in DataParallel."""
    model = VulnSage(config)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            logger.info(f"DataParallel across {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
    summary = (model.module if hasattr(model, "module") else model).get_model_summary()
    logger.info(f"VulnSage created: {summary['trainable_parameters']:,} trainable params")
    return model
