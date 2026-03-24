"""
Program Analysis Evidence Extraction for VulnSage.

This module implements the core *program-analysis* side of the paper title:
it wraps multiple static-analysis tools to produce **structured evidence**
that guides LLM reasoning about smart-contract vulnerabilities.

Four evidence extractors operate on Solidity source code:

1. **CFGExtractor**      — control-flow graph (basic blocks, critical paths)
2. **ASTAnalyzer**       — abstract syntax tree (functions, state vars, modifiers)
3. **TaintAnalyzer**     — taint tracking     (sources -> sinks, unvalidated paths)
4. **CallGraphAnalyzer** — call graph         (internal/external calls, callbacks)

Each extractor has two modes:
  * **Slither-backed** (preferred) — runs the real Slither static-analysis
    framework for sound, tool-backed evidence.
  * **Regex fallback** — lightweight pattern matching when Slither is
    unavailable (e.g. compilation failure, missing solc version).

Mathematical context
--------------------
Evidence vector for contract *x*::

    e(x) = [e_cfg(x);  e_ast(x);  e_taint(x);  e_cg(x)]  in  R^{4d}

where each  e_i = TextEncoder(NL(Extractor_i(x))) .

Author : [Anonymous]
Target : IEEE/ACM ASE 2026
"""

from __future__ import annotations

import hashlib, json, logging, os, re, subprocess, tempfile, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ───────────────────────── evidence dataclasses ─────────────────────────────


@dataclass
class BasicBlock:
    block_id: int
    instructions: List[str]
    is_entry: bool = False
    is_exit: bool = False


@dataclass
class CFGEvidence:
    """Control-Flow Graph evidence."""
    basic_blocks: List[BasicBlock] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    critical_paths: List[List[int]] = field(default_factory=list)
    has_loops: bool = False
    num_blocks: int = 0
    num_edges: int = 0
    quality_score: float = 1.0
    extraction_method: str = "regex"  # "slither" or "regex"

    def to_natural_language(self) -> str:
        lines = [f"Control-Flow Graph: {self.num_blocks} blocks, {self.num_edges} edges."]
        if self.has_loops:
            lines.append("Contains loop structures (potential gas issues / infinite loops).")
        if self.critical_paths:
            lines.append(f"Found {len(self.critical_paths)} critical execution path(s) "
                         "involving external calls or state modifications.")
        for i, path in enumerate(self.critical_paths[:3]):
            lines.append(f"  Path {i+1}: blocks {' -> '.join(map(str, path))}")
        return "\n".join(lines)


@dataclass
class FunctionSig:
    name: str
    visibility: str = "public"
    modifiers: List[str] = field(default_factory=list)
    has_payable: bool = False
    has_external_call: bool = False
    state_mutability: str = "nonpayable"


@dataclass
class StateVar:
    name: str
    var_type: str
    visibility: str = "internal"
    is_mapping: bool = False


@dataclass
class ASTEvidence:
    """Abstract Syntax Tree evidence."""
    functions: List[FunctionSig] = field(default_factory=list)
    state_variables: List[StateVar] = field(default_factory=list)
    modifiers_used: Dict[str, List[str]] = field(default_factory=dict)
    inheritance_chain: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    uses_safemath: bool = False
    compiler_version: str = ""
    quality_score: float = 1.0
    extraction_method: str = "regex"

    def to_natural_language(self) -> str:
        lines = [f"AST Analysis: {len(self.functions)} functions, "
                 f"{len(self.state_variables)} state variables."]
        if self.compiler_version:
            lines.append(f"Compiler: {self.compiler_version}")
        if self.inheritance_chain:
            lines.append(f"Inherits: {' -> '.join(self.inheritance_chain)}")
        payable_fns = [f.name for f in self.functions if f.has_payable]
        if payable_fns:
            lines.append(f"Payable functions: {', '.join(payable_fns)}")
        ext_call_fns = [f.name for f in self.functions if f.has_external_call]
        if ext_call_fns:
            lines.append(f"Functions with external calls: {', '.join(ext_call_fns)}")
        unprotected = [f.name for f in self.functions
                       if f.visibility in ("public", "external") and not f.modifiers]
        if unprotected:
            lines.append(f"Public functions without modifiers: {', '.join(unprotected[:5])}")
        if not self.uses_safemath and self.compiler_version and self.compiler_version < "0.8":
            lines.append("WARNING: No SafeMath usage with compiler < 0.8 (overflow risk).")
        return "\n".join(lines)


@dataclass
class TaintSource:
    name: str
    source_type: str  # "user_input", "block_attr", "storage"
    location: str = ""


@dataclass
class TaintSink:
    name: str
    sink_type: str  # "external_call", "selfdestruct", "state_write", "delegatecall"
    location: str = ""


@dataclass
class TaintPath:
    source: TaintSource
    sink: TaintSink
    intermediaries: List[str] = field(default_factory=list)
    has_sanitization: bool = False


@dataclass
class TaintEvidence:
    """Taint-tracking evidence."""
    sources: List[TaintSource] = field(default_factory=list)
    sinks: List[TaintSink] = field(default_factory=list)
    taint_paths: List[TaintPath] = field(default_factory=list)
    unvalidated_paths: List[TaintPath] = field(default_factory=list)
    quality_score: float = 1.0
    extraction_method: str = "regex"

    def to_natural_language(self) -> str:
        lines = [f"Taint Analysis: {len(self.sources)} sources, {len(self.sinks)} sinks, "
                 f"{len(self.taint_paths)} paths."]
        if self.unvalidated_paths:
            lines.append(f"WARNING: {len(self.unvalidated_paths)} UNVALIDATED taint path(s):")
            for i, p in enumerate(self.unvalidated_paths[:3]):
                chain = " -> ".join([p.source.name] + p.intermediaries + [p.sink.name])
                lines.append(f"  [{i+1}] {chain}  (no sanitisation)")
        return "\n".join(lines)


@dataclass
class CallInfo:
    caller: str
    callee: str
    call_type: str  # "internal", "external", "delegatecall", "library"
    has_value_transfer: bool = False
    state_change_before: bool = False
    state_change_after: bool = False


@dataclass
class CallGraphEvidence:
    """Inter-function / inter-contract call graph evidence."""
    internal_calls: List[CallInfo] = field(default_factory=list)
    external_calls: List[CallInfo] = field(default_factory=list)
    callback_risks: List[str] = field(default_factory=list)
    state_changes_after_external: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    extraction_method: str = "regex"

    def to_natural_language(self) -> str:
        lines = [f"Call Graph: {len(self.internal_calls)} internal, "
                 f"{len(self.external_calls)} external call(s)."]
        if self.callback_risks:
            lines.append(f"WARNING: Callback / reentrancy risk in: {', '.join(self.callback_risks)}")
        if self.state_changes_after_external:
            lines.append(f"WARNING: State change AFTER external call in: "
                         f"{', '.join(self.state_changes_after_external)}")
        for c in self.external_calls[:3]:
            flag = " [VALUE]" if c.has_value_transfer else ""
            lines.append(f"  {c.caller} -> {c.callee} ({c.call_type}){flag}")
        return "\n".join(lines)


@dataclass
class AnalysisEvidence:
    """Aggregated evidence from all four extractors."""
    cfg: Optional[CFGEvidence] = None
    ast: Optional[ASTEvidence] = None
    taint: Optional[TaintEvidence] = None
    callgraph: Optional[CallGraphEvidence] = None
    source_code: str = ""
    contract_id: str = ""
    analysis_time_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def overall_quality(self) -> float:
        scores = [e.quality_score for e in (self.cfg, self.ast, self.taint, self.callgraph) if e]
        return float(sum(scores) / len(scores)) if scores else 0.0


# ───────────────────────── Slither integration ────────────────────────────


class SlitherRunner:
    """
    Wrapper around the Slither static analysis framework.

    Runs Slither on a Solidity source file and extracts structured evidence
    from its internal representations (CFG, AST, data dependencies, call graph).
    Falls back gracefully when Slither is unavailable or compilation fails.
    """

    def __init__(self, slither_path: Optional[str] = None,
                 solc_path: Optional[str] = None,
                 timeout: int = 120):
        self.slither_path = slither_path
        self.solc_path = solc_path
        self.timeout = timeout
        self._slither_available = self._check_slither()

    def _check_slither(self) -> bool:
        """Check if Slither is importable."""
        try:
            from slither.slither import Slither
            return True
        except ImportError:
            logger.warning("Slither not installed. Using regex fallback for analysis.")
            return False

    @property
    def available(self) -> bool:
        return self._slither_available

    def analyze(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Run Slither on source code, return structured analysis results.

        Returns None if Slither fails (compilation error, timeout, etc.).
        """
        if not self._slither_available:
            return None

        tmpdir = None
        try:
            from slither.slither import Slither

            tmpdir = tempfile.mkdtemp(prefix="vulnsage_")
            sol_path = os.path.join(tmpdir, "contract.sol")
            with open(sol_path, "w") as f:
                f.write(source_code)

            # Configure solc version from pragma
            solc_args = {}
            if self.solc_path:
                solc_args["solc"] = self.solc_path
            version = self._detect_version(source_code)
            if version:
                try:
                    self._install_solc(version)
                    solc_args["solc_solcs_select"] = version
                except Exception:
                    pass

            sl = Slither(sol_path, **solc_args)

            result = {
                "contracts": [],
                "functions": [],
                "state_variables": [],
                "cfg_data": [],
                "taint_sources": [],
                "taint_sinks": [],
                "external_calls": [],
                "internal_calls": [],
                "state_changes_after_call": [],
                "reentrancy_risks": [],
                "modifiers": {},
                "inheritance": [],
            }

            for contract in sl.contracts_derived:
                result["contracts"].append(contract.name)
                result["inheritance"].extend(
                    [b.name for b in contract.inheritance])

                for sv in contract.state_variables:
                    result["state_variables"].append({
                        "name": sv.name,
                        "type": str(sv.type),
                        "visibility": sv.visibility,
                        "is_mapping": "mapping" in str(sv.type).lower(),
                    })

                for fn in contract.functions:
                    fn_info = {
                        "name": fn.name,
                        "visibility": fn.visibility,
                        "modifiers": [m.name for m in fn.modifiers],
                        "is_payable": fn.payable,
                        "state_mutability": (
                            "payable" if fn.payable else
                            "view" if fn.view else
                            "pure" if fn.pure else "nonpayable"),
                    }
                    result["functions"].append(fn_info)

                    # CFG: extract basic blocks from nodes
                    for node in fn.nodes:
                        result["cfg_data"].append({
                            "function": fn.name,
                            "node_id": node.node_id,
                            "type": str(node.type),
                            "expression": str(node.expression) if node.expression else "",
                            "sons": [s.node_id for s in node.sons],
                        })

                    # External calls
                    for ext_call in fn.external_calls_as_expressions:
                        call_str = str(ext_call)
                        has_val = "value" in call_str.lower()
                        result["external_calls"].append({
                            "caller": fn.name,
                            "callee": call_str[:100],
                            "has_value_transfer": has_val,
                        })

                    # Internal calls
                    for int_call in fn.internal_calls:
                        result["internal_calls"].append({
                            "caller": fn.name,
                            "callee": int_call.name if hasattr(int_call, "name") else str(int_call),
                        })

                    # Reentrancy: state vars written after external calls
                    ext_call_seen = False
                    state_written_after = []
                    for node in fn.nodes:
                        if node.external_calls_as_expressions:
                            ext_call_seen = True
                        if ext_call_seen and node.state_variables_written:
                            state_written_after.extend(
                                [sv.name for sv in node.state_variables_written])
                    if state_written_after:
                        result["state_changes_after_call"].append(fn.name)
                        result["reentrancy_risks"].append({
                            "function": fn.name,
                            "vars_written": state_written_after,
                        })

                    # Modifiers
                    for mod in fn.modifiers:
                        result["modifiers"].setdefault(mod.name, []).append(fn.name)

                # Taint sources and sinks via variable read/write analysis
                for fn in contract.functions:
                    for node in fn.nodes:
                        expr = str(node.expression) if node.expression else ""
                        for src_name in ["msg.sender", "msg.value", "msg.data",
                                         "tx.origin", "block.timestamp", "block.number"]:
                            if src_name in expr:
                                result["taint_sources"].append({
                                    "name": src_name,
                                    "function": fn.name,
                                })
                        for sink_kw in [".call", ".send", ".transfer", "selfdestruct",
                                        "delegatecall"]:
                            if sink_kw in expr:
                                result["taint_sinks"].append({
                                    "name": sink_kw,
                                    "function": fn.name,
                                })

            return result

        except Exception as exc:
            logger.debug(f"Slither analysis failed: {exc}")
            return None
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _detect_version(src: str) -> Optional[str]:
        m = re.search(r'pragma\s+solidity\s+[\^~>=<]*\s*([\d.]+)', src)
        return m.group(1) if m else None

    @staticmethod
    def _install_solc(version: str):
        """Attempt to install the required solc version via solc-select."""
        try:
            from solc_select.solc_select import (
                installed_versions, install_artifacts, switch_global_version)
            if version not in installed_versions():
                install_artifacts([version])
            switch_global_version(version, always_install=False)
        except Exception:
            pass


# ───────────────────────── individual extractors ────────────────────────────


class CFGExtractor:
    """Extract control-flow graph from Solidity source code."""

    def extract(self, source_code: str, slither_data: Optional[Dict] = None) -> CFGEvidence:
        evidence = CFGEvidence()
        try:
            if slither_data and slither_data.get("cfg_data"):
                evidence = self._from_slither(slither_data)
            else:
                evidence = self._from_regex(source_code)
        except Exception as exc:
            logger.warning(f"CFG extraction failed: {exc}")
            evidence.quality_score = 0.0
        return evidence

    def _from_slither(self, data: Dict) -> CFGEvidence:
        """Build CFG evidence from Slither's node graph."""
        blocks = []
        edges = []
        node_map = {}
        cfg_nodes = data["cfg_data"]

        for i, node in enumerate(cfg_nodes):
            bid = node["node_id"]
            node_map[bid] = i
            blocks.append(BasicBlock(
                block_id=bid,
                instructions=[node.get("expression", "") or str(node.get("type", ""))],
                is_entry=(node.get("type", "") == "ENTRY_POINT"),
            ))
            for son_id in node.get("sons", []):
                edge_type = "conditional" if "IF" in str(node.get("type", "")) else "sequential"
                edges.append((bid, son_id, edge_type))

        evidence = CFGEvidence(
            basic_blocks=blocks,
            edges=edges,
            num_blocks=len(blocks),
            num_edges=len(edges),
            has_loops=self._detect_loops(blocks, edges),
            critical_paths=self._find_critical_paths_slither(cfg_nodes),
            quality_score=1.0,
            extraction_method="slither",
        )
        return evidence

    def _find_critical_paths_slither(self, cfg_nodes: List[Dict]) -> List[List[int]]:
        """Find nodes involving external calls or dangerous operations."""
        critical_kws = ["call", "send", "transfer", "delegatecall", "selfdestruct"]
        critical = []
        for node in cfg_nodes:
            expr = node.get("expression", "").lower()
            if any(kw in expr for kw in critical_kws):
                critical.append([node["node_id"]])
        return critical[:5]

    def _from_regex(self, source: str) -> CFGEvidence:
        """Regex-based lightweight CFG extraction (fallback)."""
        blocks, edges = self._parse_cfg_regex(source)
        evidence = CFGEvidence(
            basic_blocks=blocks,
            edges=edges,
            num_blocks=len(blocks),
            num_edges=len(edges),
            has_loops=self._detect_loops(blocks, edges),
            critical_paths=self._find_critical_paths_regex(blocks, edges, source),
            quality_score=min(1.0, len(blocks) / 3) * 0.7 if blocks else 0.0,
            extraction_method="regex",
        )
        return evidence

    def _parse_cfg_regex(self, source: str) -> Tuple[List[BasicBlock], List[Tuple[int, int, str]]]:
        blocks: List[BasicBlock] = []
        edges: List[Tuple[int, int, str]] = []
        fn_pattern = re.compile(
            r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.MULTILINE)
        matches = list(fn_pattern.finditer(source))
        for idx, m in enumerate(matches):
            block = BasicBlock(block_id=idx, instructions=[m.group(0).strip()],
                               is_entry=(idx == 0))
            blocks.append(block)
            if idx > 0:
                edges.append((idx - 1, idx, "sequential"))
            fn_body_start = m.end()
            fn_body = source[fn_body_start:fn_body_start + 2000]
            if re.search(r'\bif\b|\brequire\b|\bassert\b', fn_body):
                if idx + 1 < len(blocks):
                    edges.append((idx, idx + 1, "conditional"))
        return blocks, edges

    @staticmethod
    def _detect_loops(blocks, edges) -> bool:
        adj = {}
        for s, d, _ in edges:
            adj.setdefault(s, []).append(d)
        visited, stack = set(), set()
        def dfs(n):
            visited.add(n); stack.add(n)
            for nb in adj.get(n, []):
                if nb in stack:
                    return True
                if nb not in visited and dfs(nb):
                    return True
            stack.discard(n)
            return False
        return any(dfs(b.block_id) for b in blocks if b.block_id not in visited)

    def _find_critical_paths_regex(self, blocks, edges, source) -> List[List[int]]:
        critical_kws = [".call", ".send", ".transfer", "delegatecall",
                        "selfdestruct", "suicide"]
        critical_blocks = set()
        for b in blocks:
            for instr in b.instructions:
                if any(kw in instr.lower() for kw in critical_kws):
                    critical_blocks.add(b.block_id)
        return [[bid] for bid in list(critical_blocks)[:5]]


class ASTAnalyzer:
    """Extract abstract-syntax-tree features from Solidity source."""

    def extract(self, source_code: str, slither_data: Optional[Dict] = None) -> ASTEvidence:
        evidence = ASTEvidence()
        try:
            if slither_data and slither_data.get("functions"):
                evidence = self._from_slither(source_code, slither_data)
            else:
                evidence = self._from_regex(source_code)
        except Exception as exc:
            logger.warning(f"AST extraction failed: {exc}")
            evidence.quality_score = 0.0
        return evidence

    def _from_slither(self, source: str, data: Dict) -> ASTEvidence:
        functions = []
        for fn in data["functions"]:
            functions.append(FunctionSig(
                name=fn["name"],
                visibility=fn.get("visibility", "public"),
                modifiers=fn.get("modifiers", []),
                has_payable=fn.get("is_payable", False),
                has_external_call=any(
                    ec["caller"] == fn["name"] for ec in data.get("external_calls", [])),
                state_mutability=fn.get("state_mutability", "nonpayable"),
            ))

        state_vars = []
        for sv in data.get("state_variables", []):
            state_vars.append(StateVar(
                name=sv["name"],
                var_type=sv["type"],
                visibility=sv.get("visibility", "internal"),
                is_mapping=sv.get("is_mapping", False),
            ))

        return ASTEvidence(
            functions=functions,
            state_variables=state_vars,
            modifiers_used=data.get("modifiers", {}),
            inheritance_chain=data.get("inheritance", []),
            external_calls=[ec["callee"] for ec in data.get("external_calls", [])],
            uses_safemath="SafeMath" in source,
            compiler_version=self._extract_version(source),
            quality_score=1.0,
            extraction_method="slither",
        )

    def _from_regex(self, source: str) -> ASTEvidence:
        return ASTEvidence(
            compiler_version=self._extract_version(source),
            functions=self._extract_functions(source),
            state_variables=self._extract_state_vars(source),
            inheritance_chain=self._extract_inheritance(source),
            external_calls=self._extract_external_calls(source),
            uses_safemath="SafeMath" in source or "safemath" in source.lower(),
            modifiers_used=self._extract_modifiers(source),
            quality_score=0.7,
            extraction_method="regex",
        )

    @staticmethod
    def _extract_version(src: str) -> str:
        m = re.search(r'pragma\s+solidity\s+[\^~>=<]*\s*([\d.]+)', src)
        return m.group(1) if m else ""

    def _extract_functions(self, src: str) -> List[FunctionSig]:
        fns = []
        pat = re.compile(
            r'function\s+(\w+)\s*\(([^)]*)\)\s*((?:public|external|internal|private)?)'
            r'([^{]*)\{', re.MULTILINE)
        for m in pat.finditer(src):
            name = m.group(1)
            vis = m.group(3) or "public"
            qualifiers = m.group(4) if m.group(4) else ""
            mods = re.findall(r'\b(\w+)\b', qualifiers)
            mods = [md for md in mods if md not in
                    ("returns", "pure", "view", "payable", "virtual", "override", "memory")]
            fn_body = src[m.end():m.end() + 3000]
            has_ext = bool(re.search(r'\.call\b|\.send\b|\.transfer\b|delegatecall', fn_body))
            fns.append(FunctionSig(
                name=name, visibility=vis, modifiers=mods,
                has_payable="payable" in qualifiers,
                has_external_call=has_ext,
                state_mutability="payable" if "payable" in qualifiers
                    else "view" if "view" in qualifiers
                    else "pure" if "pure" in qualifiers else "nonpayable"))
        return fns

    @staticmethod
    def _extract_state_vars(src: str) -> List[StateVar]:
        vs = []
        pat = re.compile(r'^\s*(mapping|address|uint\d*|int\d*|bool|string|bytes\d*)\s+'
                         r'(public\s+|private\s+|internal\s+)?(\w+)', re.MULTILINE)
        for m in pat.finditer(src):
            vs.append(StateVar(name=m.group(3), var_type=m.group(1),
                               visibility=(m.group(2) or "internal").strip(),
                               is_mapping=m.group(1) == "mapping"))
        return vs

    @staticmethod
    def _extract_inheritance(src: str) -> List[str]:
        m = re.search(r'contract\s+\w+\s+is\s+([\w\s,]+)\s*\{', src)
        if m:
            return [c.strip() for c in m.group(1).split(",")]
        return []

    @staticmethod
    def _extract_external_calls(src: str) -> List[str]:
        return re.findall(r'(\w+\.(?:call|send|transfer|delegatecall)\b)', src)

    @staticmethod
    def _extract_modifiers(src: str) -> Dict[str, List[str]]:
        mods: Dict[str, List[str]] = {}
        for m in re.finditer(r'modifier\s+(\w+)', src):
            mods[m.group(1)] = []
        return mods


class TaintAnalyzer:
    """Taint-tracking analysis (Slither-backed + regex fallback)."""

    SOURCES = {
        "msg.sender": "user_input", "msg.value": "user_input",
        "msg.data": "user_input", "tx.origin": "user_input",
        "block.timestamp": "block_attr", "block.number": "block_attr",
        "block.coinbase": "block_attr",
    }
    SINKS = {
        ".call": "external_call", ".send": "external_call",
        ".transfer": "external_call", "delegatecall": "delegatecall",
        "selfdestruct": "selfdestruct", "suicide": "selfdestruct",
    }

    def extract(self, source_code: str, slither_data: Optional[Dict] = None) -> TaintEvidence:
        evidence = TaintEvidence()
        try:
            if slither_data and slither_data.get("taint_sources"):
                evidence = self._from_slither(source_code, slither_data)
            else:
                evidence = self._from_regex(source_code)
        except Exception as exc:
            logger.warning(f"Taint extraction failed: {exc}")
            evidence.quality_score = 0.0
        return evidence

    def _from_slither(self, source: str, data: Dict) -> TaintEvidence:
        sources = []
        seen_sources = set()
        for ts in data["taint_sources"]:
            key = ts["name"]
            if key not in seen_sources:
                stype = self.SOURCES.get(key, "unknown")
                sources.append(TaintSource(name=key, source_type=stype,
                                           location=ts.get("function", "")))
                seen_sources.add(key)

        sinks = []
        seen_sinks = set()
        for ts in data.get("taint_sinks", []):
            key = (ts["name"], ts.get("function", ""))
            if key not in seen_sinks:
                stype = self.SINKS.get(ts["name"], "unknown")
                sinks.append(TaintSink(name=ts["name"], sink_type=stype,
                                       location=ts.get("function", "")))
                seen_sinks.add(key)

        # Build paths: source in function -> sink in same function
        paths = []
        src_fns = {ts["function"]: ts["name"] for ts in data["taint_sources"]}
        sink_fns = {ts["function"]: ts["name"] for ts in data.get("taint_sinks", [])}
        for fn_name in set(src_fns.keys()) & set(sink_fns.keys()):
            src_obj = TaintSource(name=src_fns[fn_name],
                                  source_type=self.SOURCES.get(src_fns[fn_name], "unknown"))
            sink_obj = TaintSink(name=sink_fns[fn_name],
                                 sink_type=self.SINKS.get(sink_fns[fn_name], "unknown"))
            # Check for sanitization via modifiers
            has_san = any(fn_name in fns for fns in
                          data.get("modifiers", {}).values())
            paths.append(TaintPath(source=src_obj, sink=sink_obj,
                                   intermediaries=[fn_name],
                                   has_sanitization=has_san))

        unvalidated = [p for p in paths if not p.has_sanitization]

        return TaintEvidence(
            sources=sources, sinks=sinks,
            taint_paths=paths, unvalidated_paths=unvalidated,
            quality_score=1.0,
            extraction_method="slither",
        )

    def _from_regex(self, source_code: str) -> TaintEvidence:
        sources = self._find_sources(source_code)
        sinks = self._find_sinks(source_code)
        paths = self._trace_paths(source_code, sources, sinks)
        unvalidated = [p for p in paths if not p.has_sanitization]
        q = 0.7 if sources and sinks else 0.3
        return TaintEvidence(
            sources=sources, sinks=sinks,
            taint_paths=paths, unvalidated_paths=unvalidated,
            quality_score=q,
            extraction_method="regex",
        )

    def _find_sources(self, src: str) -> List[TaintSource]:
        found = []
        for name, stype in self.SOURCES.items():
            if name in src:
                found.append(TaintSource(name=name, source_type=stype))
        return found

    def _find_sinks(self, src: str) -> List[TaintSink]:
        found = []
        for pattern, stype in self.SINKS.items():
            for m in re.finditer(re.escape(pattern), src):
                ctx = src[max(0, m.start()-40):m.end()+20]
                found.append(TaintSink(name=pattern, sink_type=stype, location=ctx.strip()))
        return found

    def _trace_paths(self, src: str, sources: List[TaintSource],
                     sinks: List[TaintSink]) -> List[TaintPath]:
        paths = []
        sanitisers = {"require", "assert", "revert", "onlyOwner", "SafeMath"}
        fn_pat = re.compile(r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.MULTILINE)
        for source in sources:
            for sink in sinks:
                for fm in fn_pat.finditer(src):
                    body = src[fm.end():fm.end()+3000]
                    if source.name in body and sink.name in body:
                        src_pos = body.find(source.name)
                        sink_pos = body.find(sink.name)
                        has_san = any(s in body for s in sanitisers)
                        if src_pos < sink_pos:
                            between = body[src_pos:sink_pos]
                            has_san = has_san or any(s in between for s in sanitisers)
                        paths.append(TaintPath(source=source, sink=sink,
                                               intermediaries=[fm.group(1)],
                                               has_sanitization=has_san))
        return paths


class CallGraphAnalyzer:
    """Inter-function and inter-contract call-graph analysis."""

    def extract(self, source_code: str, slither_data: Optional[Dict] = None) -> CallGraphEvidence:
        evidence = CallGraphEvidence()
        try:
            if slither_data and (slither_data.get("internal_calls") or slither_data.get("external_calls")):
                evidence = self._from_slither(slither_data)
            else:
                evidence = self._from_regex(source_code)
        except Exception as exc:
            logger.warning(f"CallGraph extraction failed: {exc}")
            evidence.quality_score = 0.0
        return evidence

    def _from_slither(self, data: Dict) -> CallGraphEvidence:
        internal = [CallInfo(caller=c["caller"], callee=c["callee"], call_type="internal")
                    for c in data.get("internal_calls", [])]
        external = [CallInfo(
            caller=c["caller"], callee=c["callee"][:80], call_type="external",
            has_value_transfer=c.get("has_value_transfer", False))
            for c in data.get("external_calls", [])]
        callback_risks = [r["function"] for r in data.get("reentrancy_risks", [])]
        state_after = data.get("state_changes_after_call", [])

        n = len(internal) + len(external)
        return CallGraphEvidence(
            internal_calls=internal,
            external_calls=external,
            callback_risks=callback_risks,
            state_changes_after_external=state_after,
            quality_score=1.0,
            extraction_method="slither",
        )

    def _from_regex(self, source_code: str) -> CallGraphEvidence:
        internal = self._find_internal_calls(source_code)
        external = self._find_external_calls(source_code)
        callback_risks = self._find_callback_risks(source_code)
        state_after = self._find_state_after_call(source_code)

        n = len(internal) + len(external)
        return CallGraphEvidence(
            internal_calls=internal,
            external_calls=external,
            callback_risks=callback_risks,
            state_changes_after_external=state_after,
            quality_score=min(1.0, n / 2) * 0.7 if n else 0.2,
            extraction_method="regex",
        )

    @staticmethod
    def _find_internal_calls(src: str) -> List[CallInfo]:
        fns = set(re.findall(r'function\s+(\w+)', src))
        calls = []
        for fn in fns:
            body_match = re.search(rf'function\s+{fn}\s*\([^)]*\)[^{{]*\{{', src)
            if not body_match:
                continue
            body = src[body_match.end():body_match.end()+3000]
            for other in fns:
                if other != fn and re.search(rf'\b{other}\s*\(', body):
                    calls.append(CallInfo(caller=fn, callee=other, call_type="internal"))
        return calls

    @staticmethod
    def _find_external_calls(src: str) -> List[CallInfo]:
        calls = []
        fn_pat = re.compile(r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.MULTILINE)
        ext_pat = re.compile(r'(\w+)\.(call|send|transfer|delegatecall)\b')
        for fm in fn_pat.finditer(src):
            fn_name = fm.group(1)
            body = src[fm.end():fm.end()+3000]
            for em in ext_pat.finditer(body):
                has_val = ".value(" in body[em.start():em.start()+60] or \
                          "call{value" in body[em.start():em.start()+60]
                calls.append(CallInfo(
                    caller=fn_name, callee=em.group(1),
                    call_type=em.group(2),
                    has_value_transfer=has_val))
        return calls

    @staticmethod
    def _find_callback_risks(src: str) -> List[str]:
        risks = []
        fn_pat = re.compile(r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.MULTILINE)
        for fm in fn_pat.finditer(src):
            fn_name = fm.group(1)
            body = src[fm.end():fm.end()+3000]
            if re.search(r'\.call\b|\.send\b', body):
                risks.append(fn_name)
        return risks

    @staticmethod
    def _find_state_after_call(src: str) -> List[str]:
        flagged = []
        fn_pat = re.compile(r'function\s+(\w+)\s*\([^)]*\)[^{]*\{', re.MULTILINE)
        state_write = re.compile(r'\b\w+\s*(?:\[.*?\])?\s*=\s*(?!.*==)')
        for fm in fn_pat.finditer(src):
            fn_name = fm.group(1)
            body = src[fm.end():fm.end()+3000]
            ext_match = re.search(r'\.call\b|\.send\b|\.transfer\b', body)
            if ext_match:
                after = body[ext_match.end():]
                if state_write.search(after):
                    flagged.append(fn_name)
        return flagged


# ───────────────────────── evidence structurer ──────────────────────────────

class EvidenceStructurer:
    """Merge per-extractor evidence into a structured LLM prompt."""

    TEMPLATE = (
        "=== PROGRAM ANALYSIS EVIDENCE ===\n\n"
        "[CFG]\n{cfg}\n\n[AST]\n{ast}\n\n"
        "[TAINT]\n{taint}\n\n[CALL GRAPH]\n{callgraph}\n"
    )

    def structure(self, ev: AnalysisEvidence, max_tokens: int = 1024) -> str:
        parts = {
            "cfg": ev.cfg.to_natural_language() if ev.cfg else "No CFG evidence.",
            "ast": ev.ast.to_natural_language() if ev.ast else "No AST evidence.",
            "taint": ev.taint.to_natural_language() if ev.taint else "No taint evidence.",
            "callgraph": ev.callgraph.to_natural_language() if ev.callgraph else "No call-graph evidence.",
        }
        text = self.TEMPLATE.format(**parts)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n... [truncated]"
        return text

    @staticmethod
    def quality_vector(ev: AnalysisEvidence) -> List[float]:
        """Return [q_cfg, q_ast, q_taint, q_cg] in [0,1]^4."""
        return [
            ev.cfg.quality_score if ev.cfg else 0.0,
            ev.ast.quality_score if ev.ast else 0.0,
            ev.taint.quality_score if ev.taint else 0.0,
            ev.callgraph.quality_score if ev.callgraph else 0.0,
        ]


# ───────────────────────── orchestrating pipeline ───────────────────────────

class ProgramAnalysisPipeline:
    """
    Top-level pipeline: run all enabled extractors on a contract.

    Uses Slither as the primary analysis backend when available,
    falling back to regex-based extraction on failure.
    """

    def __init__(self, config):
        self.config = config
        self.slither_runner = SlitherRunner(
            slither_path=config.slither_path,
            solc_path=config.solc_path,
            timeout=config.max_analysis_timeout,
        ) if config.use_slither else None

        self.cfg_ext = CFGExtractor() if config.extract_cfg else None
        self.ast_ext = ASTAnalyzer() if config.extract_ast else None
        self.taint_ext = TaintAnalyzer() if config.extract_taint else None
        self.cg_ext = CallGraphAnalyzer() if config.extract_callgraph else None
        self.structurer = EvidenceStructurer()

        self.cache_dir = Path(config.evidence_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.slither_runner and self.slither_runner.available:
            logger.info("Slither available — using tool-backed analysis")
        else:
            logger.info("Slither unavailable — using regex fallback analysis")

    # ── single contract ────────────────────────────────────────────────────

    def analyze(self, source_code: str, contract_id: str = "") -> AnalysisEvidence:
        """Analyse one contract; return cached result if available."""
        cache_key = hashlib.sha256(source_code.encode()).hexdigest()[:16]
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        t0 = time.time()
        ev = AnalysisEvidence(source_code=source_code, contract_id=contract_id)

        # Try Slither first for all extractors
        slither_data = None
        if self.slither_runner and self.slither_runner.available:
            slither_data = self.slither_runner.analyze(source_code)
            if slither_data is None:
                ev.errors.append("slither_compilation_failed")

        if self.cfg_ext:
            ev.cfg = self.cfg_ext.extract(source_code, slither_data)
        if self.ast_ext:
            ev.ast = self.ast_ext.extract(source_code, slither_data)
        if self.taint_ext:
            ev.taint = self.taint_ext.extract(source_code, slither_data)
        if self.cg_ext:
            ev.callgraph = self.cg_ext.extract(source_code, slither_data)
        ev.analysis_time_s = time.time() - t0

        self._save_cache(cache_key, ev)
        return ev

    # ── batch (parallel) ───────────────────────────────────────────────────

    def analyze_batch(self, contracts: List[Tuple[str, str]],
                      max_workers: Optional[int] = None) -> List[AnalysisEvidence]:
        workers = max_workers or self.config.parallel_workers
        results: List[Optional[AnalysisEvidence]] = [None] * len(contracts)

        with ProcessPoolExecutor(max_workers=min(workers, len(contracts) or 1)) as pool:
            future_map = {
                pool.submit(self.analyze, src, cid): idx
                for idx, (src, cid) in enumerate(contracts)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result(timeout=self.config.max_analysis_timeout)
                except Exception as exc:
                    logger.warning(f"Analysis failed for contract {idx}: {exc}")
                    results[idx] = AnalysisEvidence(
                        source_code=contracts[idx][0],
                        contract_id=contracts[idx][1],
                        errors=[str(exc)])
        return results  # type: ignore[return-value]

    # ── prompt generation ──────────────────────────────────────────────────

    def format_evidence_prompt(self, ev: AnalysisEvidence) -> str:
        return self.structurer.structure(ev, max_tokens=self.config.max_evidence_tokens)

    def evidence_quality(self, ev: AnalysisEvidence) -> List[float]:
        return self.structurer.quality_vector(ev)

    # ── disk cache ─────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[AnalysisEvidence]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                d = json.load(f)
            ev = AnalysisEvidence()
            ev.source_code = d.get("source_code", "")
            ev.contract_id = d.get("contract_id", "")
            ev.analysis_time_s = d.get("analysis_time_s", 0)
            if d.get("cfg"):
                ev.cfg = CFGEvidence(**{k: v for k, v in d["cfg"].items()
                                        if k in CFGEvidence.__dataclass_fields__})
            if d.get("ast"):
                ev.ast = ASTEvidence(**{k: v for k, v in d["ast"].items()
                                        if k in ASTEvidence.__dataclass_fields__})
            if d.get("taint"):
                ev.taint = TaintEvidence(**{k: v for k, v in d["taint"].items()
                                            if k in TaintEvidence.__dataclass_fields__})
            if d.get("callgraph"):
                ev.callgraph = CallGraphEvidence(**{k: v for k, v in d["callgraph"].items()
                                                    if k in CallGraphEvidence.__dataclass_fields__})
            return ev
        except Exception:
            return None

    def _save_cache(self, key: str, ev: AnalysisEvidence) -> None:
        try:
            d = {
                "source_code": ev.source_code[:500],
                "contract_id": ev.contract_id,
                "analysis_time_s": ev.analysis_time_s,
                "cfg": asdict(ev.cfg) if ev.cfg else None,
                "ast": asdict(ev.ast) if ev.ast else None,
                "taint": asdict(ev.taint) if ev.taint else None,
                "callgraph": asdict(ev.callgraph) if ev.callgraph else None,
            }
            with open(self._cache_path(key), "w") as f:
                json.dump(d, f, default=str)
        except Exception as exc:
            logger.debug(f"Cache write failed: {exc}")
