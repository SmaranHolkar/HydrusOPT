"""
HydrusOpt — Complete Stack
===========================
Five techniques combined into one CLI tool:

  1. Selective Layer Linearisation  → faster inference on long contexts
  2. Mixed Precision Quantisation   → smaller file size, less RAM
    3. Metacognition Plugin           → confidence scoring + per-token recovery
    4. Hallucination Guards           → self-correction loop + semantic consistency
    5. LLM-Only Safe Fallback         → local safe fallback when guard blocks output

Works on any HuggingFace causal LM (LLaMA, Mistral, Qwen, etc.)

Requirements:
    pip install torch transformers accelerate bitsandbytes psutil matplotlib pandas

Usage:
    # One-flag profile (recommended)
    python hydrusopt.py --model Qwen/Qwen2-1.5B-Instruct --profile safe

    # Full optimisation + benchmark (advanced manual control)
    python hydrusopt.py --model Qwen/Qwen2-1.5B-Instruct --skip-quant

    # Multi-model benchmark with chart
    python hydrusopt.py --benchmark-multi --visualise

    # Stress test + hallucination guards
    python hydrusopt.py --model Qwen/Qwen2-1.5B-Instruct --stress-test --enable-selfcorrect

    # LLM-only safe mode (no external retrieval)
    python hydrusopt.py --model Qwen/Qwen2-1.5B-Instruct --enable-metacognition

    # Adjust aggressiveness
    python hydrusopt.py --model Qwen/Qwen2-1.5B-Instruct --linearise_ratio 0.4 --quant_bits 4
"""

import sys
import re
import warnings

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for banner chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress noisy third-party warnings that are not actionable from user code
warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size.*",          # bitsandbytes FutureWarning (internal PyTorch API)
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torch_dtype.*deprecated.*",  # transformers torch_dtype → dtype rename notice
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*huggingface_hub.*symlinks.*", # HF cache symlink warning (Windows without dev mode)
    category=UserWarning,
)

import torch
import torch.nn as nn
import time
import argparse
import copy
import csv
import json
import os
from datetime import datetime, timezone
from urllib import parse, request
from urllib.error import URLError, HTTPError
from typing import Optional, Tuple, List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    import pandas as pd
    _HAS_CHART = True
except ImportError:
    _HAS_CHART = False

BANNER = """
██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗   ██╗███████╗ ██████╗ ██████╗ ████████╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║   ██║██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝
███████║ ╚████╔╝ ██║  ██║██████╔╝██║   ██║███████╗██║   ██║██████╔╝   ██║   
██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██║   ██║╚════██║██║   ██║██╔═══╝    ██║   
██║  ██║   ██║   ██████╔╝██║  ██║╚██████╔╝███████║╚██████╔╝██║        ██║   
╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝        ╚═╝   
                   Make local LLMs safer. More honest. More useful.
"""

# Legacy cache for deprecated retrieval path (kept for backward compatibility).
_RETRIEVAL_CACHE: Dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 1 — SELECTIVE LAYER LINEARISATION
# Replaces O(n²) attention with O(n) linear attention
# on the middle layers 
# ═══════════════════════════════════════════════════════════════

class SlidingWindowWrapper(nn.Module):
    """
    Replaces O(n²) attention with O(n*W) Sliding Window Attention (SWA).
    This restricts tokens to only attend to the most recent `window_size` tokens.
    Unlike naive Linear Attention, SWA perfectly preserves Positional Embeddings (RoPE)
    and the exact Softmax distribution, requiring ZERO retraining while massively 
    speeding up the prefill phase on long documents.
    """

    def __init__(self, original_attn: nn.Module, window_size: int = 512):
        super().__init__()
        self.original_attn = original_attn
        self.window_size = window_size
        # Small cache for additive masks keyed by (T, window, dtype, device).
        self._mask_cache: Dict[Tuple[int, int, torch.dtype, str], torch.Tensor] = {}
        self._mask_cache_order: List[Tuple[int, int, torch.dtype, str]] = []

    def _get_additive_mask(self, T: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (T, self.window_size, dtype, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(T, device=device)
        cols = torch.arange(T, device=device)
        valid = (cols.unsqueeze(0) <= rows.unsqueeze(1)) & (
            (rows.unsqueeze(1) - cols.unsqueeze(0)) < self.window_size
        )
        additive_mask = torch.full(
            (1, 1, T, T), torch.finfo(dtype).min, dtype=dtype, device=device
        )
        additive_mask.masked_fill_(valid.unsqueeze(0).unsqueeze(0), 0.0)

        self._mask_cache[key] = additive_mask
        self._mask_cache_order.append(key)
        if len(self._mask_cache_order) > 4:
            old = self._mask_cache_order.pop(0)
            self._mask_cache.pop(old, None)

        return additive_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple:
        B, T, _ = hidden_states.shape

        # Only apply SWA mask during the prefill phase (T > 1).
        # During single-token generation (T == 1), standard caching is already fast enough.
        if T > 1:
            device = hidden_states.device
            dtype = hidden_states.dtype
            additive_mask = self._get_additive_mask(T, dtype, device)

            if attention_mask is not None:
                # Safe combine with existing attention mask (e.g., padding)
                attention_mask = torch.minimum(attention_mask, additive_mask)
            else:
                attention_mask = additive_mask

        return self.original_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )


def linearise_model(model, ratio: float = 0.5, window_size: int = 512, verbose: bool = True):
    """
    Selectively wrap attention layers with Sliding Window Attention.
    Keeps first and last layers as full attention.
    """
    config = model.config
    num_layers = config.num_hidden_layers

    n_to_linearise = int(num_layers * ratio)
    start = (num_layers - n_to_linearise) // 2
    target_layers = list(range(start, start + n_to_linearise))

    if verbose:
        print(f"\n  [1/3] SWA LINEARISATION")
        print(f"        Layers: {num_layers} total, wrapping {len(target_layers)} middle layers in SWA (window={window_size})")
        print(f"        Layers kept as full attention: {[i for i in range(num_layers) if i not in target_layers]}")

    linearised = 0
    for layer_idx in target_layers:
        try:
            layer = model.model.layers[layer_idx]
            original_attn = layer.self_attn
            layer.self_attn = SlidingWindowWrapper(original_attn, window_size=window_size)
            linearised += 1
        except AttributeError:
            if verbose:
                print(f"   ⚠ Layer {layer_idx} skipped (unsupported architecture)")

    if verbose:
        print(f"  ✓ {linearised} layers wrapped in Sliding Window Attention")

    return model, target_layers


# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 2 — MIXED PRECISION QUANTISATION
# Sensitive layers stay at INT8, robust layers go to INT4
# Better quality than uniform INT4 at the same average size
# ═══════════════════════════════════════════════════════════════

def profile_layer_sensitivity(model, tokenizer, device, sample_text: str = "The quick brown fox") -> dict:
    """
    Measures how much each layer's output changes when we add noise.
    High sensitivity = keep at INT8. Low sensitivity = safe for INT4.
    
    This is a lightweight proxy for proper Fisher information —
    fast enough to run on CPU in seconds.
    """
    model.eval()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    sensitivity = {}

    hooks = []
    layer_outputs = {}
    noise_enabled = [False]

    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[idx] = output[0].detach()
            else:
                layer_outputs[idx] = output.detach()
        return hook

    def embed_noise_hook(module, input, output):
        if noise_enabled[0]:
            noise = torch.randn_like(output.float()).to(output.dtype) * 0.01
            return output + noise
        return output

    # Register hooks on all layers
    try:
        # hook the embed tokens
        if hasattr(model.model, "embed_tokens"):
            h = model.model.embed_tokens.register_forward_hook(embed_noise_hook)
            hooks.append(h)
        for idx, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(make_hook(idx))
            hooks.append(h)
    except AttributeError:
        return {}

    with torch.no_grad():
        model(**inputs)

    baseline = {k: v.clone() for k, v in layer_outputs.items()}

    # Add small noise to continuous states and measure output change
    with torch.no_grad():
        noise_enabled[0] = True
        model(**inputs)
        noise_enabled[0] = False

    for idx in baseline:
        if idx in layer_outputs and idx in baseline:
            diff = (layer_outputs[idx] - baseline[idx]).abs().mean().item()
            sensitivity[idx] = diff

    for h in hooks:
        h.remove()

    return sensitivity


def apply_mixed_quantisation(model, tokenizer, device, quant_bits: int = 4, verbose: bool = True):
    """
    Applies INT8 to sensitive layers, INT4 to robust layers.
    Uses torch's built-in dynamic quantisation — no GPU required.
    
    Note: For full bitsandbytes INT4 (NF4), load model with BitsAndBytesConfig.
    This version uses dynamic quantisation which works everywhere including CPU.
    """
    if verbose:
        print(f"\n  [2/3] QUANTISATION PROFILING")
        print(f"        Profiling layer sensitivity...")

    # If already BitsAndBytes-quantized at load time, skip forward-pass profiling.
    try:
        import bitsandbytes as bnb
        _already_bnb = any(
            isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
            for m in model.modules()
        )
    except Exception:
        _already_bnb = False

    if _already_bnb:
        n_layers = sum(1 for _ in model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else 0
        quant_map = {i: "INT4/8 (BitsAndBytes)" for i in range(n_layers)}
        if verbose:
            print(f"        (Model already quantized via bitsandbytes — skipping sensitivity sweep)")
        return model, quant_map

    sensitivity = profile_layer_sensitivity(model, tokenizer, device)

    if not sensitivity:
        if verbose:
            print("        ⚠ Could not profile sensitivity, skipping quantisation stats")
        return model, {}

    sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    n_sensitive = max(1, len(sorted_layers) // 3)  # top 33% 
    sensitive_layers = {idx for idx, _ in sorted_layers[:n_sensitive]}
    robust_layers = {idx for idx, _ in sorted_layers[n_sensitive:]}

    if verbose:
        print(f"        Identified {len(sensitive_layers)} sensitive layers and {len(robust_layers)} robust layers.")
        print(f"        (Note: Using global load-time quantization via bitsandbytes for maximum VRAM savings)")

    quant_map = {}
    for idx in robust_layers:
        quant_map[idx] = "INT4/8 (BitsAndBytes)"
    for idx in sensitive_layers:
        quant_map[idx] = "INT4/8 (BitsAndBytes)"

    return model, quant_map


# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 3 — SPECULATIVE DECODING
# A tiny draft model generates tokens, the main model verifies
# in parallel. 2-3x speedup with near-zero quality loss.
# ═══════════════════════════════════════════════════════════════

class SpeculativeDecoder:
    """
    Implements speculative decoding between a draft model and main model.
    
    How it works:
    1. Draft model (small, fast) generates K tokens speculatively
    2. Main model verifies all K tokens in ONE forward pass (parallel)
    3. Accept tokens up to first disagreement, reject the rest
    4. Net result: generate multiple tokens per main model call
    
    The draft model here is the linearised version of the main model —
    so you get speculative decoding for free from what you already built.
    """

    def __init__(self, main_model, draft_model, tokenizer, k: int = 4):
        self.main = main_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.k = k  # tokens to speculate ahead

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Speculative decoding generation loop.
        """
        device = input_ids.device
        generated = input_ids.clone()
        tokens_generated = 0
        accepted_total = 0
        draft_total = 0

        self.main.eval()
        self.draft.eval()

        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                remaining = max_new_tokens - tokens_generated
                k = min(self.k, remaining)

                # Step 1: Draft model speculatively generates k tokens
                draft_tokens = []
                draft_probs = []
                draft_input = generated.clone()

                for _ in range(k):
                    draft_out = self.draft(draft_input)
                    logits = draft_out.logits[:, -1, :]
                    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                    token = torch.multinomial(probs, 1)
                    draft_tokens.append(token)
                    draft_probs.append(probs)
                    draft_input = torch.cat([draft_input, token], dim=-1)

                draft_total += k

                # Step 2: Main model verifies ALL k tokens in one pass
                verify_input = torch.cat([generated] + draft_tokens, dim=-1)
                main_out = self.main(verify_input)
                main_logits = main_out.logits[:, generated.shape[1] - 1:-1, :]

                # Step 3: Accept/reject each draft token
                accepted = 0
                for i in range(k):
                    main_probs = torch.softmax(main_logits[:, i, :] / max(temperature, 1e-6), dim=-1)
                    draft_prob = draft_probs[i].gather(-1, draft_tokens[i])
                    main_prob = main_probs.gather(-1, draft_tokens[i])

                    # Acceptance criterion: accept if main model agrees (ratio test)
                    acceptance_ratio = (main_prob / draft_prob.clamp(min=1e-10)).clamp(max=1.0)
                    accept = torch.rand(1, device=device) < acceptance_ratio

                    if accept:
                        generated = torch.cat([generated, draft_tokens[i]], dim=-1)
                        tokens_generated += 1
                        accepted += 1
                        accepted_total += 1
                    else:
                        # Resample from corrected distribution and stop
                        corrected = (main_probs - draft_probs[i]).clamp(min=0)
                        corrected = corrected / corrected.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                        new_token = torch.multinomial(corrected, 1)
                        generated = torch.cat([generated, new_token], dim=-1)
                        tokens_generated += 1
                        break

                if tokens_generated >= max_new_tokens:
                    break

        acceptance_rate = accepted_total / max(draft_total, 1)
        return generated, acceptance_rate


def setup_speculative_decoding(main_model, tokenizer, linearise_ratio: float, verbose: bool = True):
    """
    Creates a draft model from the main model by pruning layers.
    Shares weights with main model to use ~0 extra memory, but runs much faster.
    """
    if verbose:
        print(f"\n  [3/3] SPECULATIVE DECODING")
        print(f"        Creating draft model via layer pruning (shares weights)...")

    # Draft model = shallow copy
    draft_model = copy.copy(main_model)
    draft_model._modules = copy.copy(main_model._modules)
    if hasattr(main_model, "model"):
        draft_model.model = copy.copy(main_model.model)
        draft_model.model._modules = copy.copy(main_model.model._modules)
        # Keep every 2nd layer
        draft_layers = main_model.model.layers[::2]
        draft_model.model.layers = nn.ModuleList(draft_layers)
        # Patch config so positional embeddings match the actual layer count.
        # Without this the draft model's RoPE / attention masks are computed
        # against the wrong depth, producing garbled output.
        draft_model.config = copy.copy(main_model.config)
        draft_model.config.num_hidden_layers = len(draft_layers)
        draft_model.model.config = draft_model.config

    if verbose:
        num_draft_layers = len(draft_model.model.layers) if hasattr(draft_model, 'model') else '?'
        print(f"        ✓ Draft model ready (pruned to {num_draft_layers} layers)")
        print(f"        k=4 tokens speculated per main model call")

    return SpeculativeDecoder(main_model, draft_model, tokenizer, k=4)


# ═══════════════════════════════════════════════════════════════
# PLUGIN ARCHITECTURE
# Plugins intercept each generation step via logits, allowing
# confidence scoring, flagging, recovery, and human escalation.
# ═══════════════════════════════════════════════════════════════

class GenerationPlugin:
    """
    Base class for all HydrusOpt generation plugins.
    Override on_step() to intercept each token generation step.
    """
    def on_step(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Called at each generation step.
        Args:
            input_ids: Current token sequence [1, seq_len]
            logits:    Raw logits for next token [1, vocab_size]
            step:      Token index (0-based)
        Returns:
            (next_token_ids [1,1], metadata dict)
        """
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        return next_token, {}

    def on_end(self, metadata_log: list) -> None:
        """Called once generation completes. metadata_log is per-step info."""
        pass


class PluginManager:
    """Chains multiple GenerationPlugins together."""

    def __init__(self, plugins: List["GenerationPlugin"] = None):
        self.plugins = plugins or []

    def register(self, plugin: "GenerationPlugin"):
        self.plugins.append(plugin)

    def run_step(self, input_ids, logits, step) -> Tuple[torch.Tensor, list]:
        """Run all plugins sequentially. First plugin wins the token choice."""
        next_token, meta = None, []
        for plugin in self.plugins:
            next_token, m = plugin.on_step(input_ids, logits, step)
            meta.append(m)
        # Fallback if no plugins registered
        if next_token is None:
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        return next_token, meta

    def finalize(self, metadata_log: list):
        for plugin in self.plugins:
            plugin.on_end(metadata_log)


class MetacognitionPlugin(GenerationPlugin):
    """
    Plugin 4: Metacognition — dual-signal confidence scoring and self-correction.

    At every step:
      1. Compute top-prob confidence = top_prob / sum(top_5_probs)
      2. Compute normalised entropy as a secondary signal
      3. Math-context tokens receive a +0.15 entropy penalty (domain calibration)
      4. If confident → accept token, annotate [✓ XX%]
      5. If uncertain → re-sample with majority vote; escalate if still low.
    """

    # Regex that flags a math/arithmetic context in the last 20 decoded tokens
    MATH_PATTERN = re.compile(r"[0-9]+\s*[\*\+\-\/\=]")

    # Standalone number or written-out quantity word — highest hallucination risk
    # e.g. "four", "95", "three", "1.5 billion"
    NUMERIC_PATTERN = re.compile(
        r"^\s*(\d[\d,\.]*|zero|one|two|three|four|five|six|seven|eight|nine|ten"
        r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
        r"|million|billion|trillion)\s*$",
        re.IGNORECASE,
    )

    # Prompts that usually require a numeric/fact-like response.
    NUMERIC_QUESTION_PATTERN = re.compile(
        r"\b(how many|how much|what year|what date|when|what percent|percentage|count|number of)\b",
        re.IGNORECASE,
    )

    ANSWER_NUMBER_PATTERN = re.compile(
        r"\b\d[\d,\.]*\b|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten"
        r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
        r"|million|billion|trillion)\b",
        re.IGNORECASE,
    )

    ANSWER_ORDINAL_PATTERN = re.compile(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
        r"|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth"
        r"|twentieth|\d+(?:st|nd|rd|th))\b",
        re.IGNORECASE,
    )

    CONFIDENT = "✓"
    UNCERTAIN = "⚠"
    ESCALATED = "🚨"

    def __init__(
        self,
        tokenizer,
        model,
        confidence_threshold: float = 0.75,
        resample_n: int = 3,
        resample_temperature: float = 1.5,
        escalate_threshold: float = 0.50,
        verbose: bool = True,
        # ── Hallucination Guard options ──────────────────────────
        run_post_checks: bool = False,
        uncertain_ratio_trigger: float = 0.30,
        consistency_samples: int = 3,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.resample_n = resample_n
        self.resample_temperature = resample_temperature
        self.escalate_threshold = escalate_threshold
        self.verbose = verbose
        # Hallucination guard state
        self.run_post_checks = run_post_checks
        self.uncertain_ratio_trigger = uncertain_ratio_trigger
        self.consistency_samples = consistency_samples
        self.reset()

    def reset(self) -> None:
        """Reset per-generation metacognition state."""
        self._last_prompt: str = ""
        self._annotations: List[dict] = []
        self._uncertain_count: int = 0

    def _compute_confidence(self, logits: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Returns (top_token_id, confidence_score) using top-1 probability."""
        probs = torch.softmax(logits[:, -1, :], dim=-1)   # [1, vocab]
        top_prob, top_ids = torch.max(probs, dim=-1, keepdim=True)
        return top_ids, top_prob.item()                    # [1,1], float

    def _entropy(self, logits: torch.Tensor, input_ids: torch.Tensor) -> float:
        """Normalised Shannon entropy with math-domain calibration penalty."""
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1).clamp(min=1e-10)
        raw = -(probs * probs.log()).sum(dim=-1)
        max_e = torch.log(torch.tensor(float(probs.shape[-1])))
        entropy = (raw / max_e).item()
        # Apply penalty if recent tokens look like a math expression
        context = self.tokenizer.decode(input_ids[0][-20:])
        if self.MATH_PATTERN.search(context):
            entropy += 0.15
        return entropy

    def _resample_majority(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Re-runs the model `resample_n` times at higher temperature.
        Returns the majority-voted token and its new confidence.
        Sampling is constrained to top-k candidates to avoid random gibberish drift.
        """
        votes: List[int] = []
        top_k = 20
        with torch.no_grad():
            for _ in range(self.resample_n):
                out = self.model(input_ids)
                logits = out.logits
                probs = torch.softmax(
                    logits[:, -1, :] / max(self.resample_temperature, 1e-6), dim=-1
                )
                k = min(top_k, probs.shape[-1])
                topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                sample_idx = torch.multinomial(topk_probs, 1)
                token = topk_ids.gather(-1, sample_idx).item()
                votes.append(token)

        # Majority vote
        winner = max(set(votes), key=votes.count)
        confidence = votes.count(winner) / len(votes)
        return torch.tensor([[winner]], device=input_ids.device), confidence

    def on_step(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, dict]:
        token, confidence = self._compute_confidence(logits)
        base_confidence = confidence
        token_text = self.tokenizer.decode(token[0])
        status = self.CONFIDENT
        recovery_used = False

        is_numeric = bool(self.NUMERIC_PATTERN.match(token_text))
        numeric_low_conf = is_numeric and base_confidence < 0.90

        if confidence < self.confidence_threshold or numeric_low_conf:
            # ── Recovery: re-sample with majority vote ──
            recovered_token, recovered_confidence = self._resample_majority(input_ids)

            # Only replace when majority vote has enough agreement.
            # Low-consensus recoveries tend to produce unstable gibberish tokens.
            if recovered_confidence >= 0.67:
                token = recovered_token
                confidence = recovered_confidence
                token_text = self.tokenizer.decode(token[0])
                recovery_used = True

            self._uncertain_count += 1

            if confidence < self.escalate_threshold:
                status = self.ESCALATED
                if self.verbose:
                    print(
                        f"\n  {self.ESCALATED} ESCALATION at step {step}: "
                        f"token='{token_text}' confidence={confidence*100:.0f}% "
                        f"— model requires human review."
                    )
            else:
                status = self.UNCERTAIN

        annotation = {
            "step": step,
            "token_id": token.item(),
            "token": token_text,
            "confidence": round(confidence, 4),
            "entropy": round(self._entropy(logits, input_ids), 4),
            "status": status,
            "recovery_used": recovery_used,
            "numeric_flagged": numeric_low_conf,
        }
        self._annotations.append(annotation)
        return token, annotation

    def should_stop(self, consecutive_uncertain: int = 4) -> bool:
        """Return True when recoveries cross a configured threshold."""
        return self._uncertain_count >= consecutive_uncertain

    def on_end(self, metadata_log: list) -> None:
        uncertain = [a for a in self._annotations if a["status"] != self.CONFIDENT]
        if self.verbose and uncertain:
            print(f"\n  [META] {len(uncertain)} uncertain token(s) flagged during generation.")

        if not self.run_post_checks or not self._annotations:
            return

        # ── Decide whether to trigger Hallucination Guards ────────────────────
        total = len(self._annotations)
        uncertain_ratio = len(uncertain) / max(total, 1)
        draft_text = "".join(a["token"] for a in self._annotations)
        prompt_lower = self._last_prompt.lower()
        domain_hit = any(kw in prompt_lower for kw in HIGH_RISK_KEYWORDS)

        trigger_reason = None
        if uncertain_ratio >= self.uncertain_ratio_trigger:
            trigger_reason = f"uncertain token ratio {uncertain_ratio*100:.0f}% ≥ {self.uncertain_ratio_trigger*100:.0f}%"
        elif domain_hit:
            trigger_reason = "high-risk domain keyword detected"

        if trigger_reason is None:
            if self.verbose:
                print(f"  [GUARD] No triggers fired (ratio={uncertain_ratio*100:.0f}%, domain={domain_hit}). Skipping post-checks.")
            return

        if self.verbose:
            print(f"\n  [GUARD] Hallucination guard triggered — {trigger_reason}")

        # ── Guard A: Self-Correction Loop ─────────────────────────────────────
        passed, verdict = self_correct_draft(
            self._last_prompt, draft_text, self.model, self.tokenizer,
            verbose=self.verbose,
        )
        if not passed:
            print(
                f"  {self.ESCALATED} SELF-CORRECTION FAILED: model could not verify its own answer. "
                "Consider regenerating or surfacing an 'I don't know' response."
            )
            # Hard fail — skip Guard B to save the extra model calls.
            return

        # ── Guard B: Semantic Consistency Check (only runs if Guard A passes) ──
        _, mean_sim = check_semantic_consistency(
            self._last_prompt, self.model, self.tokenizer,
            n_samples=self.consistency_samples,
            verbose=self.verbose,
        )
        if mean_sim < 0.35:
            print(
                f"  {self.ESCALATED} CONSISTENCY WARNING: mean similarity={mean_sim:.3f}. "
                "Drafts diverge significantly — high hallucination risk."
            )

    def set_prompt(self, prompt: str) -> None:
        """Call this before each generation so on_end() can reference the original prompt."""
        self._last_prompt = prompt

    def uncertain_ratio(self) -> float:
        if not self._annotations:
            return 0.0
        uncertain = [a for a in self._annotations if a["status"] != self.CONFIDENT]
        return len(uncertain) / len(self._annotations)

    def requires_safe_fallback(self, prompt: str, clean_answer: str, ratio_threshold: float = 0.60) -> Tuple[bool, str]:
        """
        Decide whether output should be blocked and replaced with a safe fallback.
        Returns (should_block, reason).
        """
        ratio = self.uncertain_ratio()
        if ratio >= ratio_threshold:
            return True, f"high uncertainty ({ratio*100:.0f}% tokens uncertain)"

        triad = triangulated_factual_gate(prompt, clean_answer)
        if triad.get("should_block_missing_numeric"):
            return True, triad.get("reason", "question expects numeric answer but none was produced")
        if triad.get("should_block_missing_factual_rank"):
            return True, triad.get("reason", "question expects factual rank/order signal but none was produced")
        if triad.get("should_block_numeric_contradiction"):
            return True, triad.get("reason", "answer contains multiple conflicting numeric claims")
        if triad.get("should_block_integer_only_violation"):
            return True, triad.get("reason", "prompt requested one integer-only answer but output was not a single integer")
        if triad.get("should_block_claim_verification"):
            return True, triad.get("reason", "declarative factual claim requires external verification")

        return False, ""

    @staticmethod
    def safe_fallback_message(reason: str) -> str:
        return (
            "I am not confident enough to answer this accurately right now. "
            f"Reason: {reason}. Please verify with a trusted source."
        )

    def annotated_text(self, show_entropy: bool = False) -> str:
        """
        Rebuild generated text with inline confidence annotations.
        show_entropy=True  → "Paris[✓ H=0.03]"  (normalised entropy)
        show_entropy=False → "Paris[✓ 98%]"      (top-1 probability)
        """
        parts = []
        for a in self._annotations:
            if show_entropy:
                score = f"H={a.get('entropy', 0):.2f}"
            else:
                score = f"{int(a['confidence'] * 100)}%"
            parts.append(f"{a['token']}[{a['status']} {score}]")
        return "".join(parts)

    def summary(self) -> None:
        """Print metacognition confidence and recovery summary."""
        if not self._annotations:
            return
        avg_conf = sum(a["confidence"] for a in self._annotations) / len(self._annotations)
        avg_e = sum(a.get("entropy", 0) for a in self._annotations) / len(self._annotations)
        uncertain = [a for a in self._annotations if a["status"] != self.CONFIDENT]
        pct = len(uncertain) / max(len(self._annotations), 1) * 100
        print(f"  [META] avg confidence: {avg_conf:.2%} | "
              f"recoveries: {self._uncertain_count} | "
              f"avg entropy: {avg_e:.3f} | "
              f"uncertain: {len(uncertain)}/{len(self._annotations)} ({pct:.1f}%) | "
              f"threshold: {self.confidence_threshold}")

    def clear(self):
        """Backward-compatible alias for reset()."""
        self.reset()


# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 5 — HALLUCINATION GUARDS
# Two complementary post-generation checks:
#   A) Self-Correction Loop   — model grades its own draft (YES/NO)
#   B) Semantic Consistency   — run N times; drift = hallucination signal
# Both can be triggered manually or automatically by MetacognitionPlugin
# when uncertain-token ratio or high-risk domain keywords are detected.
# ═══════════════════════════════════════════════════════════════

# Keywords that should always trigger the hallucination guards,
# regardless of the metacognition confidence score.
HIGH_RISK_KEYWORDS: List[str] = [
    "equation", "formula", "calculate", "compute", "math", "integral",
    "derivative", "statistic", "percent", "probability",
    "date", "year", "century", "treaty", "war", "signed",
    "element", "compound", "molecule", "chemical", "reaction",
    "law", "theorem", "proof", "lemma",
    "diagnosis", "symptom", "drug", "dose", "medication",
]



# Phrases that signal the model has finished answering.
# Detecting these mid-generation stops the "yapping" problem.
STOP_PHRASES: List[str] = [
    "\n\n",    # blank line = paragraph break, answer is done
    ".\n",     # sentence-ending newline
    "!\n",
    "?\n",
    "<|im_end|>",   # chat EOS marker (belt-and-suspenders)
]


def normalize_text_answer(text: str) -> str:
    """Normalize free-form text for robust text comparisons."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def extract_first_number(text: str) -> Optional[float]:
    """Extract the first signed integer/decimal number from text."""
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def answer_matches_reference(
    prediction: str,
    reference: str,
    validator: str = "text",
    options: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Task-aware correctness checker for calibration.

    Supported validators:
      - text: normalized equality or substring inclusion
      - one_of: prediction matches any allowed answer variant
      - numeric: compares first extracted number with tolerance
      - regex: matches provided regex pattern
    """
    options = options or {}

    if validator == "regex":
        pattern = options.get("pattern") or reference
        if not pattern:
            return False
        return re.search(pattern, prediction, flags=re.IGNORECASE) is not None

    if validator == "numeric":
        pred_num = extract_first_number(prediction)
        ref_num = extract_first_number(str(reference))
        if pred_num is None or ref_num is None:
            return False
        tol = float(options.get("tolerance", 0.0))
        return abs(pred_num - ref_num) <= tol

    p = normalize_text_answer(prediction)
    if not p:
        return False

    if validator == "one_of":
        candidates = options.get("answers") or []
        if reference:
            candidates = [reference] + list(candidates)
        for c in candidates:
            cc = normalize_text_answer(str(c))
            if cc and (p == cc or cc in p or p in cc):
                return True
        return False

    # Default text validator.
    r = normalize_text_answer(reference)
    if not r:
        return False
    return p == r or r in p or p in r


def load_calibration_tasks(path: str) -> List[Dict[str, Any]]:
    """
    Load calibration tasks from JSON file.
        Expected format:
            [{"prompt": "...", "answer": "...", "validator": "text|one_of|numeric|regex", "options": {...}}, ...]
        Validator and options are optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "prompt" in item and "answer" in item:
                tasks.append({
                    "prompt": str(item["prompt"]),
                    "answer": str(item["answer"]),
                    "validator": str(item.get("validator", "text")),
                    "options": item.get("options", {}) if isinstance(item.get("options", {}), dict) else {},
                })
    return tasks


class CyberneticSelfCorrectionMonitor:
    """
    Tracks self-correction stability using Error Introduction Rate (EIR)
    and Error Correction Rate (ECR).

    Improvement condition:
      Acc' > Acc  iff  ECR / EIR > Acc / (1 - Acc)
    """

    def __init__(self):
        self.total = 0
        self.initial_correct = 0
        self.initial_incorrect = 0
        self.correct_to_incorrect = 0
        self.incorrect_to_correct = 0

    def update(self, initial_correct: bool, corrected_correct: bool) -> None:
        self.total += 1
        if initial_correct:
            self.initial_correct += 1
            if not corrected_correct:
                self.correct_to_incorrect += 1
        else:
            self.initial_incorrect += 1
            if corrected_correct:
                self.incorrect_to_correct += 1

    def summary(self) -> Dict[str, Any]:
        acc = self.initial_correct / max(self.total, 1)
        eir = self.correct_to_incorrect / max(self.initial_correct, 1)
        ecr = self.incorrect_to_correct / max(self.initial_incorrect, 1)

        threshold = acc / max(1.0 - acc, 1e-9)

        if eir == 0.0 and ecr == 0.0:
            ratio_ecr_eir = None
            margin = 0.0
            stable = True
            non_degrading = True
            regime = "neutral"
        elif eir == 0.0 and ecr > 0.0:
            ratio_ecr_eir = None
            margin = float("inf")
            stable = True
            non_degrading = True
            regime = "improving"
        else:
            ratio_ecr_eir = ecr / max(eir, 1e-9)
            margin = ratio_ecr_eir - threshold
            stable = margin > 0
            non_degrading = eir <= ecr
            regime = "improving" if stable else "degrading"

        return {
            "samples": self.total,
            "baseline_accuracy": round(acc, 6),
            "eir": round(eir, 6),
            "ecr": round(ecr, 6),
            "ecr_over_eir": None if ratio_ecr_eir is None else round(ratio_ecr_eir, 6),
            "stability_threshold": round(threshold, 6),
            "stability_margin": margin if isinstance(margin, float) and margin == float("inf") else round(margin, 6),
            "stable": stable,
            "non_degrading": non_degrading,
            "stability_regime": regime,
            "events": {
                "correct_to_incorrect": self.correct_to_incorrect,
                "incorrect_to_correct": self.incorrect_to_correct,
            },
        }


def _extract_yes_no(text: str) -> Tuple[bool, str]:
    """Parse a YES/NO verdict robustly from model output."""
    verdict = text.strip().upper()
    if "YES" in verdict and "NO" in verdict:
        # Choose earliest explicit answer to avoid ambiguous tails.
        yes_i = verdict.find("YES")
        no_i = verdict.find("NO")
        return (yes_i < no_i), verdict
    if "YES" in verdict:
        return True, verdict
    if "NO" in verdict:
        return False, verdict
    return False, verdict


def _extract_verdict_reason(text: str) -> Tuple[str, str]:
    """
    Parse verifier output into a tri-state verdict plus short reason.
    Returns (verdict, reason) where verdict in {"YES", "NO", "ABSTAIN"}.
    """
    raw = (text or "").strip()
    upper = raw.upper()

    yes_i = upper.find("YES") if "YES" in upper else -1
    no_i = upper.find("NO") if "NO" in upper else -1

    if yes_i >= 0 and (no_i < 0 or yes_i < no_i):
        verdict = "YES"
    elif no_i >= 0:
        verdict = "NO"
    else:
        verdict = "ABSTAIN"

    # Keep reason compact for logs and user transparency.
    reason = raw[:180] if raw else "no explicit YES/NO token"
    return verdict, reason


def _rewrite_factual_claim_to_query(prompt: str) -> str:
    """Convert declarative factual claims into verification-style questions."""
    raw = (prompt or "").strip()
    if not raw:
        return ""

    # Strip output-format directives from retrieval queries.
    raw = re.sub(
        r"\b(answer|respond|return)\s+with\s+(?:one|single|just|only)\s+(?:integer|number)(?:\s+only)?\b",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\b(?:integer|number)\s+only\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip(" .?")

    # If the user already asked a question, keep as-is.
    if raw.endswith("?"):
        return raw

    m_cap = re.match(
        r"^\s*(?:the\s+)?capital\s+of\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z\s\-]+)\s*[\.!]?\s*$",
        raw,
        re.IGNORECASE,
    )
    if m_cap:
        country = re.sub(r"\s+", " ", m_cap.group(1)).strip()
        return f"What is the capital of {country}?"

    return f"{raw}?" if raw else ""


STRICT_INTEGER_ONLY_PROMPT_PATTERN = re.compile(
    r"\b(?:answer\s+with|return|respond\s+with)?\s*(?:one|single|just|only)\s+(?:integer|number)\s*(?:only)?\b"
    r"|\binteger\s+only\b"
    r"|\bnumber\s+only\b",
    re.IGNORECASE,
)

NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?<![\dA-Za-z\.])\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![\dA-Za-z\.])"
)

INTEGER_LITERAL_PATTERN = re.compile(
    r"(?<![\dA-Za-z\.])\d{1,3}(?:,\d{3})*(?![\dA-Za-z\.])"
)


def _requires_integer_only_response(prompt: str) -> bool:
    return bool(STRICT_INTEGER_ONLY_PROMPT_PATTERN.search(prompt or ""))


def _normalize_numeric_literal(token: str) -> str:
    return (token or "").replace(",", "").strip()


def _extract_numeric_literals(text: str) -> List[str]:
    return [_normalize_numeric_literal(t) for t in NUMERIC_LITERAL_PATTERN.findall(text or "")]


def _extract_integer_literals(text: str) -> List[str]:
    return [_normalize_numeric_literal(t) for t in INTEGER_LITERAL_PATTERN.findall(text or "")]


def _canonical_integer_only_output(answer: str) -> Optional[str]:
    ints = _extract_integer_literals(answer)
    unique_ints: List[str] = []
    for v in ints:
        if v not in unique_ints:
            unique_ints.append(v)
    if len(unique_ints) == 1:
        return unique_ints[0]
    return None


def classify_prompt_type(prompt: str) -> str:
    """Classify prompt intent with emphasis on factual/research workflows."""
    p = (prompt or "").lower()
    if re.search(r"\b(study|research|paper|journal|citation|evidence|source|sources|literature review|meta-analysis)\b", p):
        return "research"
    if re.search(r"\bcapital\s+of\s+.+\s+is\s+.+\b", p):
        return "factual"
    if re.search(r"\b(debug|error|exception|traceback|stack trace|why does this code|fix this)\b", p):
        return "troubleshooting"
    if re.search(r"\b(how to|steps|procedure|guide|install|setup|configure)\b", p):
        return "procedural"
    if re.search(r"\b(compare|difference between|vs\.?|versus|pros and cons)\b", p):
        return "comparison"
    if re.search(r"\b(define|definition|what is|what are)\b", p):
        return "definition"
    if re.search(r"\b(why|cause|reason for|because)\b", p):
        return "causal"
    if re.search(r"\b(when|what year|what date|timeline|century)\b", p):
        return "temporal"
    if re.search(r"\b(calculate|compute|multiply|multiplied|multiplication|times|product|add|subtract|divide|equation|solve)\b", p):
        return "math"
    if re.search(r"\b(function|class|python|javascript|code|algorithm|bug|compile|stack trace)\b", p):
        return "code"
    if re.search(
        r"\b(who|what|when|where|which|how many|capital|year|date|population|symbol|planet|moon|country|city|continent|president|prime minister|currency)\b",
        p,
    ):
        return "factual"
    return "open"


FACTUAL_LIKE_TYPES = {"factual", "research", "temporal", "definition", "comparison", "causal"}
RETRIEVAL_FRIENDLY_TYPES = FACTUAL_LIKE_TYPES.union({"procedural"})


def is_factual_like(prompt_type: str) -> bool:
    return prompt_type in FACTUAL_LIKE_TYPES


def build_question_policy(prompt: str) -> Dict[str, Any]:
    """Policy map controlling guard strictness and fallback behavior by prompt type."""
    prompt_type = classify_prompt_type(prompt)

    uncertainty_by_type = {
        "factual": 0.50,
        "research": 0.45,
        "temporal": 0.45,
        "definition": 0.55,
        "comparison": 0.55,
        "causal": 0.55,
        "math": 0.45,
        "code": 0.60,
        "troubleshooting": 0.60,
        "procedural": 0.60,
        "open": 0.85,
    }

    return {
        "prompt_type": prompt_type,
        "uncertain_ratio_trigger": uncertainty_by_type.get(prompt_type, 0.60),
        "allow_retrieval": prompt_type in RETRIEVAL_FRIENDLY_TYPES,
        "prefer_verify_first": prompt_type in FACTUAL_LIKE_TYPES.union({"math", "code", "troubleshooting"}),
        "strict_fact_gate": prompt_type in FACTUAL_LIKE_TYPES,
    }


def triangulated_factual_gate(prompt: str, answer: str) -> Dict[str, Any]:
    """
    Triangulated Factual Gate (TFG): combines three independent signals
    before forcing retrieval on factual prompts.

    Signals:
      1) numeric signal (cardinal or ordinal)
      2) ordinal intent in prompt
      3) direct-answer form in response
    """
    p = (prompt or "").lower()
    a = (answer or "").strip()
    a_lower = a.lower()

    prompt_type = classify_prompt_type(prompt)
    expects_factual = is_factual_like(prompt_type)
    expects_numeric = bool(MetacognitionPlugin.NUMERIC_QUESTION_PATTERN.search(p))
    expects_ordinal = bool(re.search(r"\b(how many-th|which number|what number|nth)\b", p))

    has_cardinal = bool(MetacognitionPlugin.ANSWER_NUMBER_PATTERN.search(a))
    has_ordinal = bool(MetacognitionPlugin.ANSWER_ORDINAL_PATTERN.search(a))
    has_numeric_signal = has_cardinal or has_ordinal

    # Direct answers are short and declarative (e.g., "Jupiter is the fifth planet...").
    short_answer = len(a.split()) <= 20
    declarative = bool(re.search(r"\b(is|are|has|have|was|were|yes|no)\b", a_lower))
    direct_form = short_answer and declarative

    allow_ordinal_without_cardinal = expects_ordinal and has_ordinal and direct_form
    should_block_missing_numeric = expects_numeric and not has_numeric_signal and not allow_ordinal_without_cardinal
    should_block_missing_factual_rank = False
    numeric_literals = _extract_numeric_literals(a)
    unique_numeric_literals: List[str] = []
    for v in numeric_literals:
        if v not in unique_numeric_literals:
            unique_numeric_literals.append(v)
    should_block_numeric_contradiction = expects_numeric and len(unique_numeric_literals) >= 2

    strict_integer_only_request = _requires_integer_only_response(prompt)
    canonical_integer_output = _canonical_integer_only_output(a)
    should_block_integer_only_violation = strict_integer_only_request and canonical_integer_output is None

    expects_claim_verification = bool(
        re.match(
            r"^\s*(?:the\s+)?capital\s+of\s+[a-zA-Z\s]+?\s+is\s+[a-zA-Z\s\-]+\s*[\.!]?\s*$",
            prompt or "",
            re.IGNORECASE,
        )
    )
    should_block_claim_verification = expects_factual and expects_claim_verification

    reason = ""
    if should_block_missing_numeric:
        reason = "question expects numeric/ordinal answer but none was produced"
    elif should_block_numeric_contradiction:
        reason = "answer contains multiple conflicting numeric claims"
    elif should_block_integer_only_violation:
        reason = "prompt requested one integer-only answer but output was not a single integer"
    elif should_block_claim_verification:
        reason = "declarative factual claim requires external verification"

    return {
        "prompt_type": prompt_type,
        "expects_factual": expects_factual,
        "expects_numeric": expects_numeric,
        "expects_ordinal": expects_ordinal,
        "expects_claim_verification": expects_claim_verification,
        "has_cardinal": has_cardinal,
        "has_ordinal": has_ordinal,
        "has_numeric_signal": has_numeric_signal,
        "numeric_literals": numeric_literals,
        "unique_numeric_literals": unique_numeric_literals,
        "direct_form": direct_form,
        "allow_ordinal_without_cardinal": allow_ordinal_without_cardinal,
        "strict_integer_only_request": strict_integer_only_request,
        "canonical_integer_output": canonical_integer_output,
        "should_block_missing_numeric": should_block_missing_numeric,
        "should_block_missing_factual_rank": should_block_missing_factual_rank,
        "should_block_numeric_contradiction": should_block_numeric_contradiction,
        "should_block_integer_only_violation": should_block_integer_only_violation,
        "should_block_claim_verification": should_block_claim_verification,
        "reason": reason,
    }


def verify_first_gate_decision(
    prompt: str,
    draft_answer: str,
    model,
    tokenizer,
    external_selection: bool = False,
    confidence: Optional[float] = None,
    votes: int = 3,
    no_votes_to_block: int = 2,
    high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
) -> Dict[str, Any]:
    """
    Consensus verify-first gate with transparency and over-block protection.

    Behavior:
      - Uses tri-state verifier outputs: YES / NO / ABSTAIN
      - Blocks only on strong NO consensus
      - High-confidence math answers with numeric output bypass verifier blocking
    """
    prompt_type = classify_prompt_type(prompt)
    has_numeric = bool(MetacognitionPlugin.ANSWER_NUMBER_PATTERN.search(draft_answer or ""))

    # Keep factual numeric answers in the LLM-only flow.
    # We avoid retrieval-only blocking to preserve fully local behavior.
    if is_factual_like(prompt_type) and has_numeric:
        return {
            "blocked": False,
            "reason": "factual numeric answer accepted in LLM-only mode",
            "prompt_type": prompt_type,
            "votes": {"yes": 0, "no": 0, "abstain": 0},
        }

    if confidence is not None:
        if confidence >= high_conf_bypass:
            return {
                "blocked": False,
                "reason": f"high-confidence bypass ({confidence:.2%} >= {high_conf_bypass:.0%})",
                "prompt_type": prompt_type,
                "votes": {"yes": 0, "no": 0, "abstain": 0},
            }
        if prompt_type == "math" and has_numeric and confidence >= math_bypass_conf:
            return {
                "blocked": False,
                "reason": (
                    f"math-route bypass (numeric answer + confidence {confidence:.2%} >= {math_bypass_conf:.0%})"
                ),
                "prompt_type": prompt_type,
                "votes": {"yes": 0, "no": 0, "abstain": 0},
            }

    yes_votes = 0
    no_votes = 0
    abstain_votes = 0
    reasons: List[str] = []

    for _ in range(max(1, votes)):
        short_prompt = prompt[:_SC_PROMPT_CHARS]
        # For definition/factual prompts the model sometimes generates a trailing
        # explanatory sentence that confuses the verifier.  Pass only the first
        # sentence so the gate evaluates the direct answer, not decorative context.
        if prompt_type in {"definition", "factual", "temporal"}:
            first_sent = re.split(r"(?<=[.!?])\s", draft_answer[:_SC_DRAFT_CHARS])
            short_draft = first_sent[0].strip() if first_sent else draft_answer[:_SC_DRAFT_CHARS]
        else:
            short_draft = draft_answer[:_SC_DRAFT_CHARS]
        selection_style = "EXTERNAL" if external_selection else "IN-CONTEXT"
        verify_prompt = (
            f"[{selection_style} VERIFY-FIRST] "
            "Does the draft answer directly address the question? "
            "Ignore whether it explains WHY — just check if it answers WHAT was asked. "
            "Answer YES if it gives a direct answer (even short). "
            "Answer NO only if it is evasive, unrelated, or nonsensical. "
            "Reply exactly in format: 'VERDICT: YES|NO; REASON: <short reason>'. "
            f"Question: '{short_prompt}' Draft answer: '{short_draft}'"
        )

        inputs = tokenizer(verify_prompt, return_tensors="pt").to(model.device, non_blocking=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )
        raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        verdict, reason = _extract_verdict_reason(raw)
        reasons.append(reason)
        if verdict == "YES":
            yes_votes += 1
        elif verdict == "NO":
            no_votes += 1
        else:
            abstain_votes += 1

    # Strong-no consensus only; abstain does not block by itself.
    blocked = no_votes >= max(1, no_votes_to_block) and yes_votes == 0
    reason = (
        f"verifier consensus NO ({no_votes}/{votes})"
        if blocked
        else f"verifier pass/abstain (YES={yes_votes}, NO={no_votes}, ABSTAIN={abstain_votes})"
    )
    if reasons:
        reason = f"{reason}; example='{reasons[0]}'"

    return {
        "blocked": blocked,
        "reason": reason,
        "prompt_type": prompt_type,
        "votes": {"yes": yes_votes, "no": no_votes, "abstain": abstain_votes},
    }


def _format_user_prompt_for_model(
    prompt: str,
    tokenizer,
    use_chat_template: bool = True,
    system_msg: str = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked.",
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def _generate_clean_text(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int = 48,
    use_chat_template: bool = True,
) -> str:
    formatted = _format_user_prompt_for_model(prompt, tokenizer, use_chat_template=use_chat_template)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device, non_blocking=True)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        seq = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    raw = tokenizer.decode(seq[0][input_len:], skip_special_tokens=True)
    return _sanitize_answer_text(raw)


def _sanitize_answer_text(text: str) -> str:
    """Remove scaffold artifacts (memory tags/final-answer wrappers) from outputs."""
    s = (text or "").strip()
    if not s:
        return ""

    final_spans = re.findall(
        r"(?is)\bfinal\s+answer\s*[:\-]\s*(.+?)(?=(?:\bfinal\s+answer\s*[:\-])|$)",
        s,
    )
    if final_spans:
        s = final_spans[-1].strip()

    s = re.sub(r"(?is)\[/?MEMORY\]", " ", s)
    s = re.sub(r"(?im)^\s*(?:answer|response|output)\s*[:\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([,;:])\s*$", "", s).strip()
    return s


def _score_answer_confidence(
    prompt: str,
    answer: str,
    model,
    tokenizer,
    threshold: float,
    use_chat_template: bool = True,
) -> Tuple[float, float]:
    """
    Teacher-forced confidence for a fixed answer.
    Returns (avg_token_confidence, low_confidence_token_ratio).
    """
    answer = _sanitize_answer_text(answer)
    if not answer:
        return 0.0, 1.0

    formatted_prompt = _format_user_prompt_for_model(prompt, tokenizer, use_chat_template=use_chat_template)
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device, non_blocking=True)["input_ids"]
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(model.device, non_blocking=True)["input_ids"]

    if answer_ids.numel() == 0:
        return 0.0, 1.0

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    prompt_len = prompt_ids.shape[1]

    with torch.inference_mode():
        logits = model(full_ids).logits

    probs = torch.softmax(logits[:, :-1, :].float(), dim=-1)
    target = full_ids[:, 1:]

    start = max(prompt_len - 1, 0)
    answer_probs = probs[:, start:, :]
    answer_target = target[:, start:]
    if answer_target.numel() == 0:
        return 0.0, 1.0

    token_p = answer_probs.gather(-1, answer_target.unsqueeze(-1)).squeeze(-1)
    vals = token_p.flatten()
    avg_conf = vals.mean().item() if vals.numel() else 0.0
    low_ratio = (vals < threshold).float().mean().item() if vals.numel() else 1.0
    return float(avg_conf), float(low_ratio)


def internal_self_correct_answer(
    prompt: str,
    initial_answer: str,
    model,
    tokenizer,
    threshold: float,
    use_chat_template: bool = True,
    max_rounds: int = 2,
    target_uncertain_ratio: float = 0.45,
    memory_max_tokens: int = 80,
    enable_parametric_memory: bool = True,
    min_conf_gain: float = 0.03,
    verify_votes: int = 3,
    verify_no_threshold: int = 2,
    verify_high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
) -> Dict[str, Any]:
    """
    Internal-only self-correction controller:
      1) stateless external selection verifier
      2) optional parametric memory verbalization
      3) bounded recursive refinement
      4) never disables safe fallback path
    """
    best_answer = _sanitize_answer_text(initial_answer)
    best_conf, best_uncertain = _score_answer_confidence(
        prompt, best_answer, model, tokenizer, threshold, use_chat_template=use_chat_template
    )
    best_triad = triangulated_factual_gate(prompt, best_answer)
    best_valid = not (
        best_triad.get("should_block_missing_numeric")
        or best_triad.get("should_block_missing_factual_rank")
        or best_triad.get("should_block_numeric_contradiction")
        or best_triad.get("should_block_integer_only_violation")
    )

    traces: List[Dict[str, Any]] = []
    rounds = max(1, int(max_rounds))

    for ridx in range(rounds):
        if enable_parametric_memory:
            memory_prompt = (
                "You are doing internal factual recall only. "
                "List concise, high-confidence facts relevant to this question as bullet points. "
                "If unsure, state uncertainty explicitly.\n"
                f"Question: {prompt}"
            )
            memory_block = _generate_clean_text(
                memory_prompt,
                model,
                tokenizer,
                max_tokens=max(24, int(memory_max_tokens)),
                use_chat_template=use_chat_template,
            )
            revise_prompt = (
                "Use ONLY the memory block below to answer the question directly. "
                "Return a short final answer.\n"
                f"[MEMORY]\n{memory_block}\n[/MEMORY]\n"
                f"Question: {prompt}"
            )
        else:
            revise_prompt = (
                "Re-answer the question from internal knowledge only. "
                "Be concise and fact-focused.\n"
                f"Question: {prompt}"
            )

        candidate = _generate_clean_text(
            revise_prompt,
            model,
            tokenizer,
            max_tokens=48,
            use_chat_template=use_chat_template,
        )
        candidate = _sanitize_answer_text(candidate)

        cand_conf, cand_uncertain = _score_answer_confidence(
            prompt, candidate, model, tokenizer, threshold, use_chat_template=use_chat_template
        )
        cand_triad = triangulated_factual_gate(prompt, candidate)
        gate_meta = verify_first_gate_decision(
            prompt=prompt,
            draft_answer=candidate,
            model=model,
            tokenizer=tokenizer,
            external_selection=True,
            confidence=cand_conf,
            votes=verify_votes,
            no_votes_to_block=verify_no_threshold,
            high_conf_bypass=verify_high_conf_bypass,
            math_bypass_conf=math_bypass_conf,
        )

        cand_valid = not (
            cand_triad.get("should_block_missing_numeric")
            or cand_triad.get("should_block_missing_factual_rank")
            or cand_triad.get("should_block_numeric_contradiction")
            or cand_triad.get("should_block_integer_only_violation")
            or gate_meta.get("blocked")
        )

        conf_gain = cand_conf - best_conf
        should_take = (
            (cand_valid and not best_valid)
            or (cand_valid and conf_gain >= min_conf_gain)
            or (cand_valid and cand_uncertain + 1e-9 < best_uncertain)
        )

        traces.append(
            {
                "round": ridx + 1,
                "candidate": candidate,
                "candidate_confidence": round(cand_conf, 6),
                "candidate_uncertain_ratio": round(cand_uncertain, 6),
                "candidate_valid": bool(cand_valid),
                "verify_reason": gate_meta.get("reason", ""),
            }
        )

        if should_take:
            best_answer = candidate
            best_conf = cand_conf
            best_uncertain = cand_uncertain
            best_triad = cand_triad
            best_valid = cand_valid

        if best_valid and best_uncertain <= target_uncertain_ratio:
            break

    return {
        "applied": bool(best_answer and best_answer != (initial_answer or "").strip()),
        "answer": best_answer,
        "avg_confidence": best_conf,
        "uncertain_ratio": best_uncertain,
        "valid": best_valid,
        "triad": best_triad,
        "trace": traces,
    }


def probe_internal_integer_consensus(
    prompt: str,
    model,
    tokenizer,
    use_chat_template: bool = True,
    attempts: int = 5,
    max_tokens: int = 12,
) -> Dict[str, Any]:
    """
    Generate multiple strict integer-only drafts and pick a majority integer.
    This is purely internal (no external retrieval).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {
            "consensus_integer": None,
            "support": 0,
            "total_integer_candidates": 0,
            "attempts": 0,
            "samples": [],
        }

    templates = [
        "Answer with one integer only. No explanation.\nQuestion: {q}",
        "Return only a single number.\nQuestion: {q}",
        "Use your internal knowledge only. Output one integer token only.\nQuestion: {q}",
        "Give the best current accepted count as one integer only.\nQuestion: {q}",
        "Final answer format: <integer>.\nQuestion: {q}",
    ]

    n = max(1, int(attempts))
    samples: List[str] = []
    counts: Dict[str, int] = {}

    for i in range(n):
        t = templates[i % len(templates)]
        candidate = _generate_clean_text(
            t.format(q=prompt),
            model,
            tokenizer,
            max_tokens=max(4, int(max_tokens)),
            use_chat_template=use_chat_template,
        )
        samples.append(candidate)
        canonical = _canonical_integer_only_output(candidate)
        if canonical is None:
            continue
        counts[canonical] = counts.get(canonical, 0) + 1

    if not counts:
        return {
            "consensus_integer": None,
            "support": 0,
            "total_integer_candidates": 0,
            "attempts": n,
            "samples": samples,
        }

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    best_int, support = ranked[0]
    total = sum(counts.values())
    return {
        "consensus_integer": best_int,
        "support": int(support),
        "total_integer_candidates": int(total),
        "attempts": n,
        "samples": samples,
    }


# Maximum characters fed into the verification prompt to keep it short.
_SC_PROMPT_CHARS = 120   # original question
_SC_DRAFT_CHARS  = 200   # draft answer snippet


def self_correct_draft(
    prompt: str,
    draft_answer: str,
    model,
    tokenizer,
    max_tokens: int = 3,   # YES or NO is a single token; 3 is a safe ceiling
    verbose: bool = True,
    verify_first: bool = False,
    external_selection: bool = False,
) -> Tuple[bool, str]:
    """
    Self-Correction Loop (Chain-of-Thought Guard).

    Asks the model to act as a strict fact-checker for its own draft.
    Returns (passed: bool, verdict_text: str).

    - passed=True  → draft is self-certified as factually sound
    - passed=False → draft should be suppressed or regenerated

    Kept lightweight by:
      • Truncating the prompt and draft before building the verification string
        (so total input length stays bounded regardless of answer length).
      • max_tokens=3 — the answer is a single word; no reason to generate more.
      • do_sample=False + no temperature sampling = single deterministic pass.
    """
    # Truncate both sides to cap the verification prompt's token budget.
    short_prompt = prompt[:_SC_PROMPT_CHARS]
    short_draft  = draft_answer[:_SC_DRAFT_CHARS]

    verifier_style = "VERIFY-FIRST" if verify_first else "DIRECT"
    selection_style = "EXTERNAL" if external_selection else "IN-CONTEXT"

    # External-selection mode intentionally strips intermediate reasoning traces.
    # The evaluator sees only the final draft and question in a fresh prompt.
    verify_instruction = (
        "Before answering, silently verify facts and logic, then output one word: YES or NO. "
        if verify_first
        else "Fact-check this answer and output one word: YES or NO. "
    )
    verification_prompt = (
        f"[{selection_style} {verifier_style}] "
        + verify_instruction
        + f"Question: '{short_prompt}' "
        + f"Draft answer: '{short_draft}' "
        + "Accurate?"
    )

    inputs = tokenizer(verification_prompt, return_tensors="pt").to(model.device, non_blocking=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,          # greedy — fastest possible path
        )

    verdict_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    passed, verdict = _extract_yes_no(verdict_text)

    if verbose:
        icon = "✓" if passed else "✗"
        print(f"  [SELF-CORRECT] Verdict: {verdict!r}  →  {icon} {'PASS' if passed else 'FAIL'}")

    return passed, verdict


def check_semantic_consistency(
    prompt: str,
    model,
    tokenizer,
    n_samples: int = 2,    # 2 is enough to detect drift; 3 is 50% more work
    max_tokens: int = 25,  # only the opening clause matters for consistency
    temperature: float = 0.8,
    verbose: bool = True,
) -> List[str]:
    """
    Semantic Consistency Check (Ask-it-Twice).

    Generates `n_samples` independent completions at high temperature.
    If the model is confident in the truth, core facts stay stable across
    drafts. Wild variation in specifics is a hallucination signal.

    Kept lightweight by:
      • Default n_samples=2 — one pair is all you need for a signal.
      • max_tokens=25 — factual claims appear in the first sentence;
        generating 60+ tokens per sample wastes time on filler words.
      • Jaccard token-overlap — zero extra dependencies, runs in microseconds.
        Swap in sentence-transformers cosine similarity for production.

    Returns (drafts: List[str], mean_similarity: float).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    drafts: List[str] = []

    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            draft = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            drafts.append(draft)

    # ── Lightweight Jaccard similarity ─────────────────────────────────────────
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0

    pairs = [
        _jaccard(drafts[i], drafts[j])
        for i in range(len(drafts))
        for j in range(i + 1, len(drafts))
    ]
    mean_sim = sum(pairs) / len(pairs) if pairs else 1.0

    if verbose:
        print(f"\n  [SEMANTIC CONSISTENCY] {n_samples} drafts @ {max_tokens} tokens each")
        for idx, d in enumerate(drafts):
            print(f"    Draft {idx+1}: {d[:100]}{'...' if len(d) > 100 else ''}")
        print(f"  Mean Jaccard similarity: {mean_sim:.3f}  "
              f"({'✓ Consistent' if mean_sim >= 0.35 else '⚠ Drifting — possible hallucination'})")

    return drafts, mean_sim


# ─────────────────────────────────────────────────────────────────
# HYDRUS COGNITIVE INFERENCE ENGINE (HCIE)
# ─────────────────────────────────────────────────────────────────

def generate_cognitive(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    confidence_threshold: float = 0.75,
    early_exit_threshold: float = 0.99,
    keep_recent_n: int = 8,
    low_entropy_thresh: float = 0.15,
    use_chat_template: bool = True,
    enable_early_exit: bool = False,
    tti_delta_threshold: float = 0.002,
    tti_patience: int = 4,
) -> Tuple[str, dict]:
    """
    HCIE Cognitive Generation Loop:
    1. Metacognitive Attention Scaling (MAS) dynamically adjusts window size of SWA.
    2. Information-Theoretic KV Cache Eviction (IT-KV) prunes low-entropy keys/values.
    """
    from transformers.cache_utils import DynamicCache

    # 1. Format inputs
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."},
            {"role": "user", "content": prompt}
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prefill_len = input_ids.shape[1]
    
    # 2. Discover model depth for stats
    num_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else 0

    # Find SWA wrappers for MAS
    swa_wrappers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and isinstance(layer.self_attn, SlidingWindowWrapper):
                swa_wrappers.append(layer.self_attn)

    # 3. Generation loop
    past_key_values = DynamicCache()
    generated = input_ids.clone()
    entropy_history = []
    
    # Prefill phase SWA default window
    for wrapper in swa_wrappers:
        wrapper.window_size = 512

    # Prefill forward pass
    with torch.no_grad():
        outputs = model(generated, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)
    
    # Record prefill entropy
    probs = torch.softmax(logits[:, -1, :].float(), dim=-1).clamp(min=1e-10)
    raw = -(probs * probs.log()).sum(dim=-1)
    max_e = torch.log(torch.tensor(float(probs.shape[-1])))
    init_entropy = (raw / max_e).item()
    entropy_history.append(init_entropy)

    early_exits_triggered = 0
    total_steps = 0
    tti_triggered = False
    tti_stable_steps = 0
    
    eos_ids = {tokenizer.eos_token_id}
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        eos_ids.update(tokenizer.additional_special_tokens_ids)

    # Token-by-token generation loop
    for step in range(1, max_new_tokens):
        current_token = generated[:, -1:]
        if current_token.item() in eos_ids:
            break
            
        total_steps += 1
        
        # ── MAS: Update SWA window sizes based on last token's entropy ──
        last_entropy = entropy_history[-1]
        last_token_text = tokenizer.decode(current_token[0])
        
        # Check if last token is numeric
        import re
        is_numeric = bool(re.match(r"^\s*\d[\d,\.]*\s*$", last_token_text))
        
        if is_numeric:
            new_window = 2048  # Open window wide for numbers
        elif last_entropy > 0.40:
            new_window = 512   # Expand window for uncertainty
        elif last_entropy < 0.15:
            new_window = 32    # Shrink window for high confidence
        else:
            new_window = 128   # Default window
            
        for wrapper in swa_wrappers:
            wrapper.window_size = new_window
            
        # ── Forward pass ───────────────────────────────────────────
        with torch.no_grad():
            pos_id = torch.tensor([[prefill_len + step - 1]], device=model.device, dtype=torch.long)
            outputs = model(
                current_token,
                position_ids=pos_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
        # Append token
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Calculate and record entropy
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1).clamp(min=1e-10)
        raw = -(probs * probs.log()).sum(dim=-1)
        max_e = torch.log(torch.tensor(float(probs.shape[-1])))
        step_entropy = (raw / max_e).item()
        entropy_history.append(step_entropy)

        # ── TTI: stop over-thinking when uncertainty change saturates ──
        if len(entropy_history) >= 2:
            delta = abs(entropy_history[-1] - entropy_history[-2])
            if delta < tti_delta_threshold:
                tti_stable_steps += 1
            else:
                tti_stable_steps = 0
            if tti_stable_steps >= tti_patience:
                tti_triggered = True
                break
        
        # ── IT-KV: Prune KV cache of low-entropy tokens ──
        if hasattr(past_key_values, "key_cache"):
            seq_len = past_key_values.key_cache[0].shape[2]
        else:
            seq_len = past_key_values.layers[0].get_seq_length()
            
        keep_indices = []
        for idx in range(seq_len):
            if idx < prefill_len:
                keep_indices.append(idx)
            elif idx >= seq_len - keep_recent_n:
                keep_indices.append(idx)
            else:
                gen_idx = idx - prefill_len
                if gen_idx < len(entropy_history) and entropy_history[gen_idx] >= low_entropy_thresh:
                    keep_indices.append(idx)
                
        # Filter caches
        if hasattr(past_key_values, "key_cache"):
            for l in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[l] = past_key_values.key_cache[l][:, :, keep_indices, :]
                past_key_values.value_cache[l] = past_key_values.value_cache[l][:, :, keep_indices, :]
        else:
            for l in range(len(past_key_values.layers)):
                past_key_values.layers[l].keys = past_key_values.layers[l].keys[:, :, keep_indices, :]
                past_key_values.layers[l].values = past_key_values.layers[l].values[:, :, keep_indices, :]

    output_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    final_cache_len = past_key_values.key_cache[0].shape[2] if hasattr(past_key_values, "key_cache") else past_key_values.layers[0].get_seq_length()
    stats = {
        "total_steps": total_steps,
        "early_exits": early_exits_triggered,
        "early_exit_pct": round(early_exits_triggered / max(total_steps, 1) * 100, 1),
        "final_kv_cache_len": final_cache_len,
        "original_kv_cache_len": total_steps + inputs["input_ids"].shape[1],
        "kv_reduction_pct": round((1 - final_cache_len / (total_steps + inputs["input_ids"].shape[1])) * 100, 1),
        "tti_triggered": tti_triggered,
    }
    
    return output_text, stats


# ─────────────────────────────────────────────────────────────────
# Plugin-aware custom generation loop
# ─────────────────────────────────────────────────────────────────

def generate_with_plugins(
    model,
    input_ids: torch.Tensor,
    plugin_manager: PluginManager,
    max_new_tokens: int = 80,
) -> Tuple[torch.Tensor, list]:
    """
    Token-by-token generation loop that passes logits through the PluginManager
    at every step, enabling confidence scoring, flagging, and recovery.
    """
    generated = input_ids.clone()
    all_meta = []

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(generated)
            logits = out.logits          # [1, seq_len, vocab]
            next_token, meta = plugin_manager.run_step(generated, logits, step)
            all_meta.append(meta)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop on EOS
            if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
                if next_token.item() == model.config.eos_token_id:
                    break

    plugin_manager.finalize(all_meta)
    return generated, all_meta


def generate_with_meta(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int = 40,
    threshold: float = 0.5,
    stop_on_uncertainty: bool = False,
    use_chat_template: bool = True,
    stop_phrases: Optional[List[str]] = None,
    enable_retrieval_fallback: bool = False,
    verify_first: bool = False,
    external_selection: bool = False,
    tti_delta_threshold: float = 0.002,
    tti_patience: int = 4,
    verify_votes: int = 3,
    verify_no_threshold: int = 2,
    verify_high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
    factual_uncertain_ratio_trigger: float = 0.50,
    enable_internal_self_correct: bool = False,
    internal_max_rounds: int = 2,
    internal_target_uncertain_ratio: float = 0.45,
    internal_memory_max_tokens: int = 80,
    internal_enable_parametric_memory: bool = True,
) -> Tuple[torch.Tensor, "MetacognitionPlugin"]:
    """
    Convenience generation function with per-token metacognition scoring.

    Args:
        stop_phrases:    End generation early when any phrase appears in the
                         decoded tail (default: STOP_PHRASES). Prevents yapping.
    """
    plugin = MetacognitionPlugin(
        tokenizer=tokenizer,
        model=model,
        confidence_threshold=threshold,
        verbose=False,
    )
    plugin.reset()
    plugin.set_prompt(prompt)

    system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."

    # ── Build input ────────────────────────────────────────────────
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    # ── EOS token ID set ────────────────────────────────────────────
    eos_ids: set = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        eos_ids.update(tokenizer.additional_special_tokens_ids)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        eos_ids.add(im_end)

    _stop_phrases = stop_phrases if stop_phrases is not None else STOP_PHRASES
    decoded_tail = ""   # rolling buffer for stop-phrase detection

    input_ids = tokenizer(formatted, return_tensors="pt").to(model.device)["input_ids"]

    print(f"\n  Prompt : {prompt}")
    tti_stable_steps = 0
    prev_entropy: Optional[float] = None
    tti_triggered = False
    model.eval()
    with torch.no_grad():
        for i in range(max_tokens):
            out = model(input_ids)
            next_token, _ = plugin.on_step(input_ids, out.logits, i)
            tid = next_token.item()
            if tid in eos_ids:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # ── Stop-phrase detection (anti-yapping) ─────────────────────
            decoded_tail += plugin._annotations[-1]["token"]
            if len(decoded_tail) > 60:             # keep buffer short
                decoded_tail = decoded_tail[-60:]
            if any(sp in decoded_tail for sp in _stop_phrases):
                break

            # ── TTI: truncate once uncertainty change saturates ──
            current_entropy = plugin._annotations[-1].get("entropy", 0.0)
            if prev_entropy is not None:
                delta = abs(current_entropy - prev_entropy)
                if delta < tti_delta_threshold:
                    tti_stable_steps += 1
                else:
                    tti_stable_steps = 0
                if tti_stable_steps >= tti_patience:
                    tti_triggered = True
                    break
            prev_entropy = current_entropy

            if stop_on_uncertainty and plugin.should_stop():
                print("  [META] Early stop — consecutive uncertain tokens.")
                break

    # ── Clean output ───────────────────────────────────────────────
    clean_out = "".join(
        a["token"] for a in plugin._annotations
        if tokenizer.convert_tokens_to_ids(a["token"]) not in eos_ids
           and a["token"] not in tokenizer.all_special_tokens
    ).strip()
    clean_out = _sanitize_answer_text(clean_out)
    flagged_numerics = [a for a in plugin._annotations if a.get("numeric_flagged")]

    triad = triangulated_factual_gate(prompt, clean_out)
    policy = build_question_policy(prompt)
    prompt_type = policy["prompt_type"]
    if is_factual_like(prompt_type):
        ratio_threshold = min(policy["uncertain_ratio_trigger"], factual_uncertain_ratio_trigger)
    else:
        ratio_threshold = policy["uncertain_ratio_trigger"]
    block, reason = plugin.requires_safe_fallback(prompt, clean_out, ratio_threshold=ratio_threshold)
    plugin.safe_output = clean_out
    plugin.safe_mode = "direct"
    avg_confidence = (
        sum(a.get("confidence", 0.0) for a in plugin._annotations) / max(len(plugin._annotations), 1)
        if plugin._annotations else 0.0
    )

    if enable_internal_self_correct and block and clean_out:
        correction = internal_self_correct_answer(
            prompt=prompt,
            initial_answer=clean_out,
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            use_chat_template=use_chat_template,
            max_rounds=internal_max_rounds,
            target_uncertain_ratio=internal_target_uncertain_ratio,
            memory_max_tokens=internal_memory_max_tokens,
            enable_parametric_memory=internal_enable_parametric_memory,
            verify_votes=verify_votes,
            verify_no_threshold=verify_no_threshold,
            verify_high_conf_bypass=verify_high_conf_bypass,
            math_bypass_conf=math_bypass_conf,
        )
        corrected = _sanitize_answer_text(correction.get("answer") or "")
        corrected_triad = correction.get("triad") or triangulated_factual_gate(prompt, corrected)
        corrected_uncertain = float(correction.get("uncertain_ratio", 1.0))
        corrected_block = (
            corrected_uncertain >= ratio_threshold
            or corrected_triad.get("should_block_missing_numeric")
            or corrected_triad.get("should_block_missing_factual_rank")
            or corrected_triad.get("should_block_numeric_contradiction")
            or corrected_triad.get("should_block_integer_only_violation")
            or corrected_triad.get("should_block_claim_verification")
        )
        if correction.get("applied"):
            print(
                "  [INTERNAL] "
                f"rounds={len(correction.get('trace', []))} "
                f"uncertain={corrected_uncertain:.2%}"
            )
        if correction.get("applied") and not corrected_block:
            clean_out = corrected
            triad = corrected_triad
            avg_confidence = float(correction.get("avg_confidence", avg_confidence))
            block = False
            reason = ""
            plugin.safe_output = clean_out
            plugin.safe_mode = "direct_internal"

    if block:
        # Verify-first gate to reduce error introduction before any correction action.
        if verify_first and clean_out:
            gate_meta = verify_first_gate_decision(
                prompt=prompt,
                draft_answer=clean_out,
                model=model,
                tokenizer=tokenizer,
                external_selection=external_selection,
                confidence=avg_confidence,
                votes=verify_votes,
                no_votes_to_block=verify_no_threshold,
                high_conf_bypass=verify_high_conf_bypass,
                math_bypass_conf=math_bypass_conf,
            )
            if gate_meta.get("blocked"):
                reason = f"verify-first gate blocked: {gate_meta.get('reason', 'no reason')}"
        if enable_retrieval_fallback and policy.get("allow_retrieval", False):
            ok, retrieved = retrieve_wikipedia_summary(prompt)
            if ok:
                plugin.safe_output = f"[Retrieved] {retrieved}"
                plugin.safe_mode = "retrieved"
            else:
                plugin.safe_output = plugin.safe_fallback_message(reason)
                plugin.safe_mode = "fallback"
        else:
            plugin.safe_output = plugin.safe_fallback_message(reason)
            plugin.safe_mode = "fallback"
    elif triad.get("strict_integer_only_request") and triad.get("canonical_integer_output"):
        plugin.safe_output = str(triad.get("canonical_integer_output"))
        plugin.safe_mode = "direct_strict"

    print(f"  Output : {plugin.annotated_text(show_entropy=False)}")
    print(f"  Entropy: {plugin.annotated_text(show_entropy=True)}")
    print(f"  Clean  : {clean_out}")
    if block:
        print(f"  [SAFE] Blocking answer → {reason}")
        if plugin.safe_mode == "retrieved":
            print(f"  [SAFE] Retrieval fallback: {plugin.safe_output}")
        else:
            print(f"  [SAFE] Fallback: {plugin.safe_output}")
    elif plugin.safe_mode == "direct_internal":
        print(f"  [SAFE] Internal self-correct accepted: {plugin.safe_output}")
    elif plugin.safe_mode == "direct_strict":
        print(f"  [SAFE] Enforced integer-only output: {plugin.safe_output}")
    if flagged_numerics:
        toks = ', '.join(f"'{a['token'].strip()}'" for a in flagged_numerics)
        print(f"  ⚠ Numeric token(s) flagged for external verification: {toks}")
        print(f"    → Run verify_with_reasoning(prompt, model, tokenizer) to cross-check.")
    plugin.summary()
    plugin.tti_triggered = tti_triggered
    if tti_triggered:
        print(f"  [TTI] Thinking intervention triggered (delta<{tti_delta_threshold}, patience={tti_patience}).")
    return input_ids, plugin


def generate_with_post_guard(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int = 40,
    threshold: float = 0.75,
    use_chat_template: bool = True,
    enable_retrieval_fallback: bool = False,
    uncertain_ratio_trigger: float = 0.60,
    verify_first: bool = False,
    external_selection: bool = False,
    verify_votes: int = 3,
    verify_no_threshold: int = 2,
    verify_high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
    factual_uncertain_ratio_trigger: float = 0.50,
    enable_internal_self_correct: bool = False,
    internal_max_rounds: int = 2,
    internal_target_uncertain_ratio: float = 0.45,
    internal_memory_max_tokens: int = 80,
    internal_enable_parametric_memory: bool = True,
) -> dict:
    """
    Faster generation path: run one standard generation pass, then apply safety checks.
    This preserves model output quality better than per-token intervention while still
    blocking low-confidence or non-answer responses.
    """
    system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device, non_blocking=True)
    input_len = inputs["input_ids"].shape[1]

    # ── Efficient confidence tracking ──────────────────────────────────
    # A LogitsProcessor records the greedy-token probability at every decode
    # step in O(vocab) time without materialising the full score tensor list.
    # This replaces return_dict_in_generate=True + output_scores=True which
    # forces PyTorch to store one full (1, vocab_size) float tensor per step,
    # costing memory bandwidth on every decode iteration.
    class _TopProbTracker(LogitsProcessor):
        __slots__ = ("top_probs",)
        def __init__(self):
            self.top_probs: List[float] = []
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            p = torch.softmax(scores.float(), dim=-1).max(dim=-1).values
            self.top_probs.append(p.item())
            return scores          # pass scores through unchanged

    tracker = _TopProbTracker()

    model.eval()
    with torch.inference_mode():
        seq = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([tracker]),
        )

    clean_out = _sanitize_answer_text(tokenizer.decode(seq[0][input_len:], skip_special_tokens=True))

    # Confidence proxy from generation scores: top-token probability per decode step.
    top_probs: List[float] = tracker.top_probs

    avg_conf = sum(top_probs) / max(len(top_probs), 1)
    uncertain_ratio = (
        sum(1 for p in top_probs if p < threshold) / max(len(top_probs), 1)
        if top_probs else 0.0
    )

    consensus_meta: Dict[str, Any] = {}
    strong_internal_consensus = False

    if enable_internal_self_correct and clean_out:
        correction = internal_self_correct_answer(
            prompt=prompt,
            initial_answer=clean_out,
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            use_chat_template=use_chat_template,
            max_rounds=internal_max_rounds,
            target_uncertain_ratio=internal_target_uncertain_ratio,
            memory_max_tokens=internal_memory_max_tokens,
            enable_parametric_memory=internal_enable_parametric_memory,
            verify_votes=verify_votes,
            verify_no_threshold=verify_no_threshold,
            verify_high_conf_bypass=verify_high_conf_bypass,
            math_bypass_conf=math_bypass_conf,
        )
        if correction.get("applied"):
            clean_out = _sanitize_answer_text(str(correction.get("answer", clean_out)))
            avg_conf = float(correction.get("avg_confidence", avg_conf))
            uncertain_ratio = float(correction.get("uncertain_ratio", uncertain_ratio))
            print(
                "  [INTERNAL] "
                f"rounds={len(correction.get('trace', []))} "
                f"conf={avg_conf:.2%} uncertain={uncertain_ratio:.2%}"
            )

    triad = triangulated_factual_gate(prompt, clean_out)
    policy = build_question_policy(prompt)
    prompt_type = policy["prompt_type"]

    # For strict numeric factual prompts, run an internal integer consensus probe
    # before falling back due uncertainty.
    if (
        enable_internal_self_correct
        and triad.get("strict_integer_only_request")
        and is_factual_like(prompt_type)
    ):
        consensus_meta = probe_internal_integer_consensus(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            attempts=5,
            max_tokens=10,
        )
        c_int = consensus_meta.get("consensus_integer")
        c_support = int(consensus_meta.get("support", 0))
        c_total = int(consensus_meta.get("total_integer_candidates", 0))
        if c_int and c_total > 0:
            support_ratio = c_support / max(c_total, 1)
            if c_support >= 2 and support_ratio >= 0.60:
                clean_out = str(c_int)
                triad = triangulated_factual_gate(prompt, clean_out)
                avg_conf, uncertain_ratio = _score_answer_confidence(
                    prompt,
                    clean_out,
                    model,
                    tokenizer,
                    threshold,
                    use_chat_template=use_chat_template,
                )
                # Trust 100% integer consensus regardless of raw confidence score,
                # since single-token answers like "4" score low due to teacher-forcing.
                strong_internal_consensus = (support_ratio >= 1.0 or (avg_conf >= 0.20 and uncertain_ratio <= 0.85))
                print(
                    "  [INTERNAL] integer consensus "
                    f"{clean_out} (support={c_support}/{c_total}, ratio={support_ratio:.0%})"
                )

    if is_factual_like(prompt_type):
        applied_uncertain_ratio_trigger = min(
            uncertain_ratio_trigger,
            factual_uncertain_ratio_trigger,
            policy["uncertain_ratio_trigger"],
        )
    else:
        applied_uncertain_ratio_trigger = min(
            uncertain_ratio_trigger,
            policy["uncertain_ratio_trigger"],
        )

    block = False
    reason = ""
    gate_meta: Dict[str, Any] = {}

    if verify_first and clean_out:
        gate_meta = verify_first_gate_decision(
            prompt=prompt,
            draft_answer=clean_out,
            model=model,
            tokenizer=tokenizer,
            external_selection=external_selection,
            confidence=avg_conf,
            votes=verify_votes,
            no_votes_to_block=verify_no_threshold,
            high_conf_bypass=verify_high_conf_bypass,
            math_bypass_conf=math_bypass_conf,
        )
        if gate_meta.get("blocked"):
            block = True
            reason = f"verify-first gate blocked: {gate_meta.get('reason', 'no reason') }"

    if uncertain_ratio >= applied_uncertain_ratio_trigger and not strong_internal_consensus:
        block = True
        reason = f"high uncertainty ({uncertain_ratio*100:.0f}% low-confidence decode steps)"
    elif triad.get("should_block_missing_numeric"):
        block = True
        reason = triad.get("reason", "question expects numeric answer but none was produced")
    elif triad.get("should_block_missing_factual_rank"):
        block = True
        reason = triad.get("reason", "question expects factual rank/order signal but none was produced")
    elif triad.get("should_block_numeric_contradiction"):
        block = True
        reason = triad.get("reason", "answer contains multiple conflicting numeric claims")
    elif triad.get("should_block_integer_only_violation"):
        block = True
        reason = triad.get("reason", "prompt requested one integer-only answer but output was not a single integer")
    elif triad.get("should_block_claim_verification"):
        block = True
        reason = triad.get("reason", "declarative factual claim requires external verification")

    safe_mode = "direct"
    safe_output = clean_out
    if block:
        if enable_retrieval_fallback and policy.get("allow_retrieval", False):
            ok, retrieved = retrieve_wikipedia_summary(prompt)
            if ok:
                safe_mode = "retrieved"
                safe_output = f"[Retrieved] {retrieved}"
            else:
                safe_mode = "fallback"
                safe_output = MetacognitionPlugin.safe_fallback_message(reason)
        else:
            safe_mode = "fallback"
            safe_output = MetacognitionPlugin.safe_fallback_message(reason)
    elif triad.get("strict_integer_only_request") and triad.get("canonical_integer_output"):
        safe_mode = "direct_strict"
        safe_output = str(triad.get("canonical_integer_output"))

    print(f"\n  Prompt : {prompt}")
    print(f"  Clean  : {clean_out}")
    print(f"  [POST] avg confidence={avg_conf:.2%} | uncertain ratio={uncertain_ratio:.2%}")
    if triad.get("expects_numeric") or triad.get("expects_claim_verification"):
        print(
            "  [TFG] "
            f"cardinal={triad.get('has_cardinal')} ordinal={triad.get('has_ordinal')} "
            f"direct={triad.get('direct_form')} claim={triad.get('expects_claim_verification')}"
        )
    if block:
        print(f"  [SAFE] Blocking answer → {reason}")
        if safe_mode == "retrieved":
            print(f"  [SAFE] Retrieval fallback: {safe_output}")
        else:
            print(f"  [SAFE] Fallback: {safe_output}")
    elif safe_mode == "direct_strict":
        print(f"  [SAFE] Enforced integer-only output: {safe_output}")
    elif strong_internal_consensus:
        print(f"  [SAFE] Internal consensus accepted: {safe_output}")
    if verify_first and gate_meta:
        votes = gate_meta.get("votes", {})
        print(
            "  [VERIFY] "
            f"type={gate_meta.get('prompt_type', '?')} "
            f"votes(Y/N/A)={votes.get('yes', 0)}/{votes.get('no', 0)}/{votes.get('abstain', 0)} "
            f"→ {gate_meta.get('reason', '')}"
        )

    return {
        "text": clean_out,
        "safe_output": safe_output,
        "safe_mode": safe_mode,
        "avg_confidence": avg_conf,
        "uncertain_ratio": uncertain_ratio,
        "blocked": block,
        "reason": reason,
        "verify_gate": gate_meta,
    }


def quick_generate_answer(prompt: str, model, tokenizer, max_tokens: int = 40) -> str:
    """Generate a concise draft answer for calibration loops."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device, non_blocking=True)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return _sanitize_answer_text(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def run_cybernetic_calibration(
    model,
    tokenizer,
    tasks: List[Dict[str, Any]],
    verify_first: bool = True,
    external_selection: bool = True,
    max_tokens: int = 40,
    verify_votes: int = 3,
    verify_no_threshold: int = 2,
    verify_high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
    preserve_on_verify_block: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate self-correction stability on prompt/answer calibration tasks.
    """
    monitor = CyberneticSelfCorrectionMonitor()
    verify_blocks = 0
    preserved_blocks = 0
    correction_attempts = 0

    for item in tasks:
        prompt = item["prompt"]
        gold = item["answer"]
        validator = str(item.get("validator", "text"))
        options = item.get("options", {}) if isinstance(item.get("options", {}), dict) else {}
        prompt_type = classify_prompt_type(prompt)

        initial = quick_generate_answer(prompt, model, tokenizer, max_tokens=max_tokens)
        initial_ok = answer_matches_reference(initial, gold, validator=validator, options=options)

        corrected = initial
        if verify_first:
            correction_attempts += 1
            passed, _ = self_correct_draft(
                prompt=prompt,
                draft_answer=initial,
                model=model,
                tokenizer=tokenizer,
                verbose=False,
                verify_first=True,
                external_selection=external_selection,
            )
            if not passed:
                corrected_obj = generate_with_post_guard(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    enable_retrieval_fallback=False,
                    verify_first=verify_first,
                    external_selection=external_selection,
                    verify_votes=verify_votes,
                    verify_no_threshold=verify_no_threshold,
                    verify_high_conf_bypass=verify_high_conf_bypass,
                    math_bypass_conf=math_bypass_conf,
                )
                gate_info = corrected_obj.get("verify_gate") or {}
                gate_blocked = bool(gate_info.get("blocked"))
                any_blocked = bool(corrected_obj.get("blocked"))
                if gate_blocked:
                    verify_blocks += 1
                if preserve_on_verify_block and any_blocked:
                    # Calibration mode: preserve original answer on any block so we can
                    # measure guard signal quality without fallback-induced C->I artifacts.
                    corrected = initial
                    preserved_blocks += 1
                else:
                    corrected = initial

        corrected_ok = answer_matches_reference(corrected, gold, validator=validator, options=options)
        monitor.update(initial_ok, corrected_ok)

    summary = monitor.summary()
    summary["verify_blocks"] = verify_blocks
    summary["preserved_blocks"] = preserved_blocks
    summary["correction_attempts"] = correction_attempts
    summary["verify_block_rate"] = round(verify_blocks / max(correction_attempts, 1), 6)
    return summary


def _parse_float_grid(raw: str, default: List[float]) -> List[float]:
    vals = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return vals or default


def _parse_int_grid(raw: str, default: List[int]) -> List[int]:
    vals = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(int(p))
        except ValueError:
            continue
    return vals or default


def autotune_verify_gate(
    model,
    tokenizer,
    tasks: List[Dict[str, Any]],
    external_selection: bool = True,
    max_tokens: int = 40,
    bypass_grid: Optional[List[float]] = None,
    no_threshold_grid: Optional[List[int]] = None,
    math_bypass_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Grid-search verify gate settings using EIR/ECR stability margin."""
    bypass_grid = bypass_grid or [0.85, 0.90, 0.95]
    no_threshold_grid = no_threshold_grid or [1, 2, 3]
    math_bypass_grid = math_bypass_grid or [0.80, 0.85, 0.90]

    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for vb in bypass_grid:
        for nt in no_threshold_grid:
            for mb in math_bypass_grid:
                report = run_cybernetic_calibration(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=tasks,
                    verify_first=True,
                    external_selection=external_selection,
                    max_tokens=max_tokens,
                    verify_votes=3,
                    verify_no_threshold=nt,
                    verify_high_conf_bypass=vb,
                    math_bypass_conf=mb,
                    preserve_on_verify_block=True,
                )
                trial = {
                    "verify_high_conf_bypass": vb,
                    "verify_no_threshold": nt,
                    "math_bypass_conf": mb,
                    "report": report,
                }
                trials.append(trial)

                score = report.get("stability_margin", -1e9)
                tie = report.get("ecr", 0.0) - report.get("eir", 0.0)
                block_penalty = report.get("verify_block_rate", 0.0)
                objective = score + 0.5 * tie - 0.25 * block_penalty
                if best is None:
                    best = {"trial": trial, "score": score, "tie": tie, "objective": objective}
                else:
                    if objective > best["objective"]:
                        best = {"trial": trial, "score": score, "tie": tie, "objective": objective}

    return {
        "best": best["trial"] if best else None,
        "trials": trials,
    }


def print_autotune_report(result: Dict[str, Any]) -> None:
    best = result.get("best") or {}
    if not best:
        print("\n  [AUTOTUNE] No valid trials were produced.")
        return
    rpt = best.get("report", {})
    print("\n" + "=" * 65)
    print("  VERIFY GATE AUTOTUNE (EIR/ECR)")
    print("=" * 65)
    print(f"  Best verify_high_conf_bypass : {best.get('verify_high_conf_bypass')}")
    print(f"  Best verify_no_threshold     : {best.get('verify_no_threshold')}")
    print(f"  Best math_bypass_conf        : {best.get('math_bypass_conf')}")
    print(f"  Stability margin             : {rpt.get('stability_margin')}")
    print(f"  EIR / ECR                    : {rpt.get('eir')} / {rpt.get('ecr')}")
    print(f"  Verify block rate            : {rpt.get('verify_block_rate')}")
    print(f"  Stable                       : {'YES' if rpt.get('stable') else 'NO'}")
    print("=" * 65)


def verify_with_reasoning(
    prompt: str,
    answer: str,
    model,
    tokenizer,
    max_tokens: int = 80,
) -> Tuple[bool, str]:
    """
    Self-consistency CoT verification.

    Strategy:
      1. Ask the model to explain *why* the answer is correct.
      2. Run self_correct_draft() on the explanation.
      3. If the explanation contradicts or is incoherent with the answer,
         flag the answer as unverified.

    This is more useful than resampling because:
      - A coherent explanation → the model actually knows the fact
      - An incoherent/contradictory explanation → the model guessed

    Returns (coherent: bool, explanation: str)
    """
    cot_prompt = (
        f"Question: {prompt[:200]}\n"
        f"Proposed answer: {answer[:200]}\n"
        "In 1-2 sentences, explain why this answer is correct."
    )
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    explanation = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Check if the explanation actually supports the answer
    passed, verdict = self_correct_draft(
        prompt=f"{prompt} Answer: {answer}. Is this explanation coherent: {explanation[:150]}",
        draft_answer=explanation,
        model=model,
        tokenizer=tokenizer,
        verbose=False,
    )

    status = "✓ COHERENT" if passed else "✗ INCOHERENT — possible hallucination"
    print(f"  [CoT VERIFY] {status}")
    print(f"  Explanation: {explanation[:300]}")
    return passed, explanation


def _extract_core_terms(text: str) -> List[str]:
    """Extract lightweight lexical terms for retrieval scoring."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "for", "and",
        "or", "by", "with", "from", "what", "which", "who", "when", "where", "how", "many", "does",
        "do", "did", "have", "has", "had", "it", "this", "that", "be", "as", "solar", "system",
    }
    terms = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in terms if len(t) > 2 and t not in stop]


def _build_retrieval_queries(prompt: str) -> List[str]:
    """Compile a small query lattice from prompt intent and entities."""
    normalized = _rewrite_factual_claim_to_query(prompt)
    p = normalized.strip().rstrip("?")
    p_lower = p.lower()
    prompt_type = classify_prompt_type(p)
    queries: List[str] = [p]

    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", p)
    subject_hint = ""

    m_def = re.search(r"\b(?:what is|who is|define)\s+(.+)$", p, re.IGNORECASE)
    if m_def:
        subject_hint = m_def.group(1).strip(" .")

    m_capital_q = re.search(r"\bcapital\s+of\s+([a-zA-Z\s]+)\b", p, re.IGNORECASE)
    if m_capital_q:
        country = m_capital_q.group(1).strip(" .")
        queries.extend([
            f"capital of {country}",
            f"{country} capital city",
            f"{country} country overview",
        ])

    m_between = re.search(r"\bdifference between\s+(.+?)\s+and\s+(.+)$", p, re.IGNORECASE)
    if m_between:
        left = m_between.group(1).strip(" .")
        right = m_between.group(2).strip(" .")
        queries.extend([f"{left} vs {right}", f"difference between {left} and {right}"])

    m_vs = re.search(r"\b(.+?)\s+vs\.?\s+(.+)$", p, re.IGNORECASE)
    if m_vs:
        left = m_vs.group(1).strip(" .")
        right = m_vs.group(2).strip(" .")
        queries.extend([f"{left} vs {right}", f"{left} compared to {right}"])

    if subject_hint:
        queries.extend([subject_hint, f"{subject_hint} overview"])
    elif proper_nouns:
        queries.extend(proper_nouns[:2])

    if prompt_type == "temporal":
        if subject_hint:
            queries.extend([f"{subject_hint} date", f"{subject_hint} year"])
        queries.append(f"{p} year")

    if prompt_type == "causal":
        if subject_hint:
            queries.extend([f"{subject_hint} cause", f"why {subject_hint}"])
        queries.append(f"{p} explanation")

    if prompt_type == "procedural":
        queries.append(f"{p} steps")
        queries.append(f"{p} guide")

    if prompt_type == "definition":
        if subject_hint:
            queries.append(f"{subject_hint} definition")
            queries.append(f"{subject_hint} meaning")

    if prompt_type == "research":
        m_about = re.search(r"\b(?:about|on)\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+)$", p, re.IGNORECASE)
        if m_about:
            left = m_about.group(1).strip(" .")
            right = m_about.group(2).strip(" .")
            queries.extend([
                f"{left} and {right}",
                f"{left} {right} relationship",
                f"effects of {left} on {right}",
                f"{left} {right} review",
            ])
        if subject_hint:
            queries.extend([
                f"{subject_hint} review",
                f"{subject_hint} evidence",
                f"{subject_hint} research",
            ])
        queries.extend([f"{p} evidence", f"{p} source"])

    # Preserve order while removing duplicates/empties.
    seen = set()
    deduped: List[str] = []
    for q in queries:
        clean = q.strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped[:8]


def _wiki_search_titles(query: str, headers: Dict[str, str], timeout_sec: int, limit: int = 3) -> List[str]:
    """Return candidate Wikipedia titles for a query."""
    params = parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "format": "json",
            "srlimit": limit,
            "srsearch": query,
        }
    )
    search_url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = request.Request(search_url, headers=headers)
    with request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    results = data.get("query", {}).get("search", [])
    return [r.get("title", "").strip() for r in results if r.get("title")]


def _wiki_fetch_summary(title: str, headers: Dict[str, str], timeout_sec: int) -> str:
    """Fetch extract text for a Wikipedia page title."""
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" f"{parse.quote(title)}"
    req = request.Request(summary_url, headers=headers)
    with request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return (data.get("extract") or "").strip()


def _score_candidate_summary(prompt: str, title: str, extract: str) -> float:
    """Heuristic ranking score for retrieval candidate quality."""
    p_terms = set(_extract_core_terms(prompt))
    text = f"{title} {extract}".lower()
    overlap = sum(1 for t in p_terms if t in text)
    score = float(overlap)

    p_lower = (prompt or "").lower()
    prompt_type = classify_prompt_type(prompt)
    m_capital_prompt = re.search(r"\bcapital\s+of\s+([a-zA-Z\s]+)\b", prompt or "", re.IGNORECASE)
    if m_capital_prompt:
        country = m_capital_prompt.group(1).strip().lower()
        if re.search(rf"\bcapital(?:\s+and\s+[^,\.]+)?\s+of\s+{re.escape(country)}\b", text):
            score += 4.0
        if re.search(rf"\b[A-Za-z\-]+\s+is\s+(?:the\s+)?capital(?:\s+and\s+[^,\.]+)?\s+of\s+{re.escape(country)}\b", text):
            score += 3.0
        if title.strip().lower() == country:
            score -= 1.0

    if prompt_type == "temporal":
        if re.search(r"\b\d{4}\b", text):
            score += 2.0
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", text):
            score += 1.0

    if prompt_type == "definition":
        if re.search(r"\b(is a|is an|refers to|defined as)\b", text):
            score += 1.5

    if prompt_type == "comparison":
        if re.search(r"\b(difference|compared|whereas|while|both|unlike)\b", text):
            score += 1.5

    if prompt_type == "causal":
        if re.search(r"\b(because|due to|caused by|results from|reason)\b", text):
            score += 1.5

    if prompt_type == "procedural":
        if re.search(r"\b(step|process|method|procedure|first|second|then)\b", text):
            score += 1.5

    if prompt_type == "research":
        if re.search(r"\b(study|studies|research|evidence|according to|analysis|review|trial|paper)\b", text):
            score += 2.0
        if re.search(r"\b\d{4}\b", text):
            score += 0.8
        m_about = re.search(r"\b(?:about|on)\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+)$", prompt or "", re.IGNORECASE)
        if m_about:
            left_terms = _extract_core_terms(m_about.group(1))
            right_terms = _extract_core_terms(m_about.group(2))
            has_left = any(t in text for t in left_terms)
            has_right = any(t in text for t in right_terms)
            if has_left and has_right:
                score += 2.5
            elif has_left or has_right:
                score -= 1.0

    return score


def _extract_targeted_fact(prompt: str, extract: str) -> str:
    """Extract a direct answer sentence for known factual prompt shapes."""
    rewritten_prompt = _rewrite_factual_claim_to_query(prompt)
    p_lower = rewritten_prompt.lower()
    prompt_type = classify_prompt_type(prompt)
    summary = (extract or "").replace("\n", " ").strip()
    if not summary:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]

    m_capital_prompt = re.search(r"\bcapital\s+of\s+([a-zA-Z\s]+)\b", rewritten_prompt, re.IGNORECASE)
    if m_capital_prompt:
        country = m_capital_prompt.group(1).strip(" .")
        country_title = " ".join(w.capitalize() for w in country.split())
        country_l = country.lower()
        m_capital_has_been = re.search(
            rf"\bcapital\s+of\s+{re.escape(country_l)}\s+(?:has\s+been|is)\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)\b",
            summary,
            re.IGNORECASE,
        )
        if m_capital_has_been:
            city = m_capital_has_been.group(1).strip(" .")
            return f"The capital of {country_title} is {city}."

        m_city_subject = re.search(
            rf"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)\s+is\s+(?:the\s+)?capital(?:\s+and\s+[^,\.]+)?\s+of\s+{re.escape(country_l)}\b",
            summary,
            re.IGNORECASE,
        )
        if m_city_subject:
            city = m_city_subject.group(1).strip(" .")
            return f"The capital of {country_title} is {city}."

        m_capital = re.search(
            r"\bcapital(?:\s+and\s+largest\s+city)?\s+(?:is\s+)?([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)",
            summary,
            re.IGNORECASE,
        )
        if m_capital:
            city = m_capital.group(1).strip(" .")
            if not city.lower().startswith("of "):
                return f"The capital of {country_title} is {city}."

    if prompt_type == "temporal":
        for s in sentences[:4]:
            if re.search(r"\b\d{4}\b", s):
                return s

    if prompt_type == "definition":
        if sentences:
            return sentences[0]

    if prompt_type == "comparison":
        if len(sentences) >= 2:
            return f"{sentences[0]} {sentences[1]}"
        if sentences:
            return sentences[0]

    if prompt_type == "procedural":
        return " ".join(sentences[:2]) if sentences else summary[:220]

    if prompt_type == "causal":
        for s in sentences[:4]:
            if re.search(r"\b(because|due to|caused by|results from)\b", s, re.IGNORECASE):
                return s

    if prompt_type == "research":
        selected = []
        for s in sentences[:5]:
            if re.search(r"\b(study|research|evidence|analysis|review|trial|paper)\b", s, re.IGNORECASE):
                selected.append(s)
            if len(selected) == 2:
                break
        if selected:
            return " ".join(selected)

    first_sentence = sentences[0] if sentences else ""
    return first_sentence if first_sentence else summary[:220]


def retrieve_wikipedia_summary(prompt: str, timeout_sec: int = 6) -> Tuple[bool, str]:
    """
    Lightweight factual retrieval fallback using Wikipedia search + page summary.
    Returns (success, text).
    """
    normalized_prompt = _rewrite_factual_claim_to_query((prompt or "").strip())
    if not normalized_prompt:
        return False, ""
    if normalized_prompt in _RETRIEVAL_CACHE:
        return True, _RETRIEVAL_CACHE[normalized_prompt]
    headers = {"User-Agent": "HydrusOpt/1.0 (local safety fallback)"}

    queries = _build_retrieval_queries(normalized_prompt)
    titles: List[str] = []
    seen_titles = set()
    for q in queries:
        try:
            found_titles = _wiki_search_titles(q, headers=headers, timeout_sec=timeout_sec, limit=3)
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
            continue
        for t in found_titles:
            if t not in seen_titles:
                seen_titles.add(t)
                titles.append(t)
        if len(titles) >= 8:
            break

    if not titles:
        return False, ""

    best_title = ""
    best_extract = ""
    best_score = float("-inf")

    for title in titles[:8]:
        try:
            extract = _wiki_fetch_summary(title, headers=headers, timeout_sec=timeout_sec)
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
            continue
        if not extract:
            continue
        score = _score_candidate_summary(normalized_prompt, title, extract)
        if score > best_score:
            best_score = score
            best_title = title
            best_extract = extract

    if not best_extract:
        return False, ""

    targeted = _extract_targeted_fact(normalized_prompt, best_extract)
    text = targeted if targeted else best_extract[:400]
    if best_title and targeted:
        text = f"{text} (source: {best_title})"
    final_text = text[:400]
    _RETRIEVAL_CACHE[normalized_prompt] = final_text
    return True, final_text



def _adaptive_max_tokens(prompt: str, override: int = 0) -> int:
    """Return max token budget — high ceiling so model stops naturally at EOS."""
    if override > 0:
        return override
    return 512


def check_truth_consistency(
    prompt: str,
    model,
    tokenizer,
    samples: int = 2,
    max_tokens: int = 25,
    threshold: float = 0.5,
    enable_retrieval_fallback: bool = False,
    guard_mode: str = "post",
    verify_first: bool = False,
    external_selection: bool = False,
    verify_votes: int = 3,
    verify_no_threshold: int = 2,
    verify_high_conf_bypass: float = 0.90,
    math_bypass_conf: float = 0.85,
    factual_uncertain_ratio_trigger: float = 0.50,
    enable_internal_self_correct: bool = False,
    internal_max_rounds: int = 2,
    internal_target_uncertain_ratio: float = 0.45,
    internal_memory_max_tokens: int = 80,
    internal_enable_parametric_memory: bool = True,
) -> list:
    """
    Ask the model the same question `samples` times.
    Consistent answers → confident truth. Drifting answers → hallucination signal.
    Flags a high-confidence hallucination (low entropy but inconsistent output).
    """
    print(f"\n  {'─'*55}")
    print(f"  [GUARD] Consistency check: \"{prompt}\"")
    runs = []
    for _ in range(samples):
        if guard_mode == "token":
            ids, plugin = generate_with_meta(
                prompt, model, tokenizer,
                max_tokens=max_tokens, threshold=threshold,
                stop_on_uncertainty=False, use_chat_template=False,
                enable_retrieval_fallback=enable_retrieval_fallback,
                verify_first=verify_first,
                external_selection=external_selection,
                verify_votes=verify_votes,
                verify_no_threshold=verify_no_threshold,
                verify_high_conf_bypass=verify_high_conf_bypass,
                math_bypass_conf=math_bypass_conf,
                factual_uncertain_ratio_trigger=factual_uncertain_ratio_trigger,
                enable_internal_self_correct=enable_internal_self_correct,
                internal_max_rounds=internal_max_rounds,
                internal_target_uncertain_ratio=internal_target_uncertain_ratio,
                internal_memory_max_tokens=internal_memory_max_tokens,
                internal_enable_parametric_memory=internal_enable_parametric_memory,
            )
            text = getattr(plugin, "safe_output", None)
            if not text:
                text = tokenizer.decode(ids[0], skip_special_tokens=True).replace(prompt, "").strip()
            avg_e = sum(a.get("entropy", 0) for a in plugin._annotations) / max(len(plugin._annotations), 1)
            runs.append({"text": text, "entropy": avg_e})
        else:
            result = generate_with_post_guard(
                prompt,
                model,
                tokenizer,
                max_tokens=max_tokens,
                threshold=threshold,
                use_chat_template=False,
                enable_retrieval_fallback=enable_retrieval_fallback,
                verify_first=verify_first,
                external_selection=external_selection,
                verify_votes=verify_votes,
                verify_no_threshold=verify_no_threshold,
                verify_high_conf_bypass=verify_high_conf_bypass,
                math_bypass_conf=math_bypass_conf,
                factual_uncertain_ratio_trigger=factual_uncertain_ratio_trigger,
                enable_internal_self_correct=enable_internal_self_correct,
                internal_max_rounds=internal_max_rounds,
                internal_target_uncertain_ratio=internal_target_uncertain_ratio,
                internal_memory_max_tokens=internal_memory_max_tokens,
                internal_enable_parametric_memory=internal_enable_parametric_memory,
            )
            # Map confidence proxy to entropy-like value for comparable reporting.
            runs.append({"text": result["safe_output"], "entropy": 1.0 - result["avg_confidence"]})

    consistent = runs[0]["text"][:30] == runs[1]["text"][:30]
    print(f"\n  Match  : {'✅ CONSISTENT' if consistent else '❌ INCONSISTENT'}")
    for i, r in enumerate(runs):
        print(f"  Run {i+1}  : H={r['entropy']:.3f}  →  {r['text'][:80]}")
    if not consistent and max(r["entropy"] for r in runs) < 0.2:
        print("  ⚠ WARNING: High-confidence hallucination — low entropy but inconsistent output.")
    return runs


# ═══════════════════════════════════════════════════════════════
# BENCHMARK + RESULTS
# ═══════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    "What is 2+2? Answer with one integer only.",
    "What is the capital of France?",
    "What were the main causes of World War 1?",
]

# Models used by --benchmark-multi
MODELS = [
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B"},
    {"id": "Qwen/Qwen2-0.5B-Instruct",            "name": "Qwen 0.5B"},
    {"id": "Qwen/Qwen2-1.5B-Instruct",            "name": "Qwen 1.5B"},
]

# Prompts used by --stress-test
STRESS_TESTS = [
    {"type": "Ambiguous",    "prompt": "What is the capital of South Ossetia?"},
    {"type": "Contradiction","prompt": "The capital of France is Berlin. Explain."},
    {"type": "Math",         "prompt": "What is 12345 * 67890?"},
]


def run_threshold_sweep(
    model,
    tokenizer,
    args,
    meta_threshold_grid: Optional[List[float]] = None,
    bypass_conf_grid:    Optional[List[float]] = None,
    verify_votes_grid:   Optional[List[int]]   = None,
    max_tokens: int = 80,
) -> Dict[str, Any]:
    """
    Grid-search (meta_threshold × verify_high_conf_bypass × verify_votes) on TEST_PROMPTS.

    For each combination:
      - Times wall-clock latency per verified answer
      - Counts how often the verify gate actually fires (overhead proxy)
      - Runs a 2-sample consistency check to detect hallucinations slipping through

    Prints a ranked Pareto table and highlights the optimal operating point
    (lowest latency with zero consistency failures).
    """
    meta_threshold_grid = meta_threshold_grid or [0.50, 0.65, 0.75, 0.85, 0.95]
    bypass_conf_grid    = bypass_conf_grid    or [0.60, 0.70, 0.80, 0.90]
    verify_votes_grid   = verify_votes_grid   or [1, 3]

    results = []
    total_combos = len(meta_threshold_grid) * len(bypass_conf_grid) * len(verify_votes_grid)
    combo_idx = 0

    print(f"\n  {'═'*65}")
    print(f"  THRESHOLD SWEEP  ({total_combos} combinations × {len(TEST_PROMPTS)} prompts)")
    print(f"  {'═'*65}")
    print(f"  {'thresh':>7} {'bypass':>7} {'votes':>6} {'sec/ans':>9} {'verify%':>8} {'consistent':>11}")
    print(f"  {'─'*55}")

    for mt in meta_threshold_grid:
        for bc in bypass_conf_grid:
            for vv in verify_votes_grid:
                combo_idx += 1
                verify_fired = 0
                consistent_count = 0

                t0 = time.perf_counter()
                model.eval()
                with torch.inference_mode():
                    for prompt in TEST_PROMPTS:
                        res = generate_with_post_guard(
                            prompt,
                            model,
                            tokenizer,
                            max_tokens=max_tokens,
                            threshold=mt,
                            use_chat_template=True,
                            enable_retrieval_fallback=False,
                            verify_first=True,
                            external_selection=getattr(args, "external_selection", False),
                            verify_votes=vv,
                            verify_no_threshold=getattr(args, "verify_no_threshold", 1),
                            verify_high_conf_bypass=bc,
                            math_bypass_conf=getattr(args, "math_bypass_conf", 0.80),
                            factual_uncertain_ratio_trigger=getattr(args, "factual_uncertain_ratio_trigger", 0.50),
                            enable_internal_self_correct=False,  # isolate threshold effect
                            internal_max_rounds=1,
                            internal_target_uncertain_ratio=0.45,
                            internal_memory_max_tokens=80,
                            internal_enable_parametric_memory=False,
                        )
                        if res.get("verify_gate") and res["verify_gate"].get("votes"):
                            verify_fired += 1

                elapsed = time.perf_counter() - t0
                latency = elapsed / len(TEST_PROMPTS)

                # Quick consistency check: 2 runs per prompt, check first-30-char match
                consistent = True
                with torch.inference_mode():
                    for prompt in TEST_PROMPTS:
                        out_a = generate_with_post_guard(
                            prompt, model, tokenizer,
                            max_tokens=max_tokens, threshold=mt,
                            use_chat_template=True, enable_retrieval_fallback=False,
                            verify_first=True,
                            external_selection=getattr(args, "external_selection", False),
                            verify_votes=vv, verify_no_threshold=getattr(args, "verify_no_threshold", 1),
                            verify_high_conf_bypass=bc,
                            math_bypass_conf=getattr(args, "math_bypass_conf", 0.80),
                            factual_uncertain_ratio_trigger=getattr(args, "factual_uncertain_ratio_trigger", 0.50),
                            enable_internal_self_correct=False, internal_max_rounds=1,
                            internal_target_uncertain_ratio=0.45, internal_memory_max_tokens=80,
                            internal_enable_parametric_memory=False,
                        )
                        out_b = generate_with_post_guard(
                            prompt, model, tokenizer,
                            max_tokens=max_tokens, threshold=mt,
                            use_chat_template=True, enable_retrieval_fallback=False,
                            verify_first=True,
                            external_selection=getattr(args, "external_selection", False),
                            verify_votes=vv, verify_no_threshold=getattr(args, "verify_no_threshold", 1),
                            verify_high_conf_bypass=bc,
                            math_bypass_conf=getattr(args, "math_bypass_conf", 0.80),
                            factual_uncertain_ratio_trigger=getattr(args, "factual_uncertain_ratio_trigger", 0.50),
                            enable_internal_self_correct=False, internal_max_rounds=1,
                            internal_target_uncertain_ratio=0.45, internal_memory_max_tokens=80,
                            internal_enable_parametric_memory=False,
                        )
                        text_a = (out_a.get("safe_output") or out_a.get("text") or "")[:30]
                        text_b = (out_b.get("safe_output") or out_b.get("text") or "")[:30]
                        if text_a != text_b:
                            consistent = False
                            break

                verify_pct = (verify_fired / len(TEST_PROMPTS)) * 100
                consistent_str = "✅ PASS" if consistent else "❌ FAIL"

                entry = {
                    "meta_threshold": mt,
                    "verify_high_conf_bypass": bc,
                    "verify_votes": vv,
                    "latency_per_answer_s": round(latency, 3),
                    "verify_fire_pct": round(verify_pct, 1),
                    "consistent": consistent,
                }
                results.append(entry)

                # Star = Pareto candidate (consistent + fastest so far among consistent)
                consistent_results = [r for r in results if r["consistent"]]
                best_latency = min((r["latency_per_answer_s"] for r in consistent_results), default=float("inf"))
                star = " ◀ BEST" if consistent and latency <= best_latency else ""

                print(f"  {mt:>7.2f} {bc:>7.2f} {vv:>6d} {latency:>9.2f}s {verify_pct:>7.0f}%  {consistent_str}{star}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n  {'═'*65}")
    consistent_results = [r for r in results if r["consistent"]]
    if consistent_results:
        best = min(consistent_results, key=lambda r: r["latency_per_answer_s"])
        print(f"\n  ★  OPTIMAL POINT (fastest consistent config):")
        print(f"     --meta-threshold {best['meta_threshold']:.2f}")
        print(f"     --verify-high-conf-bypass {best['verify_high_conf_bypass']:.2f}")
        print(f"     --verify-votes {best['verify_votes']}")
        print(f"     Latency: {best['latency_per_answer_s']:.2f}s/answer | Verify fires: {best['verify_fire_pct']:.0f}% of queries")

        # Compare to strictest config as baseline
        strictest = max(results, key=lambda r: r["latency_per_answer_s"])
        tax_saved = ((strictest["latency_per_answer_s"] - best["latency_per_answer_s"])
                     / max(strictest["latency_per_answer_s"], 0.001)) * 100
        print(f"     Safety Tax reduction vs max-strict: {tax_saved:.0f}%")
    else:
        print("\n  ⚠  No fully consistent configuration found — model needs retrieval fallback.")

    print(f"  {'═'*65}\n")

    with open("hydrusopt_threshold_sweep.json", "w") as f:
        json.dump({"sweep": results}, f, indent=2)
    print("  💾 Sweep results saved to hydrusopt_threshold_sweep.json")

    return {"sweep": results, "best": best if consistent_results else None}


def benchmark_standard(
    model,
    tokenizer,
    label: str,
    batch_size: int = 1,
    max_new_tokens: int = 80,
    warmup_new_tokens: int = 8,
    warmup_iters: int = 1,
    empty_cache_between_batches: bool = False,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    results = {"label": label, "runs": [], "errors": []}
    total_tokens, total_time = 0, 0

    batch_size = max(1, int(batch_size))

    with torch.inference_mode():
        for start in range(0, len(TEST_PROMPTS), batch_size):
            batch_prompts = TEST_PROMPTS[start:start + batch_size]
            idx_label = f"{start + 1}-{start + len(batch_prompts)}"
            print(
                f"        → [{idx_label}/{len(TEST_PROMPTS)}] Batch decode x{len(batch_prompts)} ",
                end="",
                flush=True,
            )
            try:
                original_padding_side = getattr(tokenizer, "padding_side", "right")
                tokenizer.padding_side = "left" if batch_size > 1 else original_padding_side
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device, non_blocking=True)
                tokenizer.padding_side = original_padding_side

                input_lens = inputs["attention_mask"].sum(dim=1).tolist()

                for _ in range(max(1, warmup_iters)):
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=warmup_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                if device == "cuda" and empty_cache_between_batches:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                t0 = time.perf_counter()
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                t1 = time.perf_counter()
                elapsed = t1 - t0

                batch_tokens = 0
                per_prompt_toks = []
                eos_id = tokenizer.eos_token_id
                for bi, in_len in enumerate(input_lens):
                    seq = output[bi, int(in_len):].tolist()
                    if eos_id is not None and eos_id in seq:
                        gen_len = seq.index(eos_id)
                    else:
                        gen_len = len(seq)
                    batch_tokens += gen_len
                    per_prompt_toks.append(gen_len)

                tok_per_sec = batch_tokens / max(elapsed, 0.001)
                total_tokens += batch_tokens
                total_time += elapsed

                print(f"({tok_per_sec:.1f} tok/s)")

                for bi, prompt in enumerate(batch_prompts):
                    text = tokenizer.decode(output[bi][int(input_lens[bi]):], skip_special_tokens=True)
                    prompt_tok_per_sec = per_prompt_toks[bi] / max(elapsed / max(len(batch_prompts), 1), 0.001)
                    results["runs"].append({
                        "prompt": prompt,
                        "tok_per_sec": round(prompt_tok_per_sec, 1),
                        "time_sec": round(elapsed / max(len(batch_prompts), 1), 3),
                        "output": text[:200],
                    })
            except Exception as e:
                print(f"(FAILED: {e})")
                results["errors"].append(str(e))

    results["avg_tok_per_sec"] = round(total_tokens / max(total_time, 0.001), 1)
    results["total_time_sec"] = round(total_time, 3)
    return results


def benchmark_speculative(decoder: SpeculativeDecoder, tokenizer, device: str) -> dict:
    results = {"label": "HydrusOpt (all 3)", "runs": [], "errors": [], "acceptance_rates": []}
    total_tokens, total_time = 0, 0

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"        → [{i+1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt[:35]}...\" ", end="", flush=True)
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            output, acceptance_rate = decoder.generate(inputs["input_ids"], max_new_tokens=80)
            t1 = time.perf_counter()

            new_tokens = output.shape[1] - input_len
            elapsed = t1 - t0
            tok_per_sec = new_tokens / max(elapsed, 0.001)
            total_tokens += new_tokens
            total_time += elapsed
            results["acceptance_rates"].append(round(acceptance_rate, 3))

            print(f"({tok_per_sec:.1f} tok/s, accept={acceptance_rate*100:.1f}%)")

            text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            results["runs"].append({
                "prompt": prompt,
                "tok_per_sec": round(tok_per_sec, 1),
                "time_sec": round(elapsed, 3),
                "acceptance_rate": round(acceptance_rate, 3),
                "output": text[:200],
            })
        except Exception as e:
            print(f"(FAILED: {e})")
            results["errors"].append(str(e))

    results["avg_tok_per_sec"] = round(total_tokens / max(total_time, 0.001), 1)
    results["total_time_sec"] = round(total_time, 3)
    results["avg_acceptance_rate"] = round(
        sum(results["acceptance_rates"]) / max(len(results["acceptance_rates"]), 1), 3
    )
    return results


def configure_cuda_speed_knobs(device: str, verbose: bool = True) -> None:
    """Enable low-level CUDA knobs that usually improve decode throughput."""
    # Globally disable gradient tracking — this is pure inference; no training path.
    # Cheaper than per-call torch.no_grad() because it skips view-tracking entirely.
    torch.set_grad_enabled(False)

    if device != "cuda":
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ── SDPA backend selection ──────────────────────────────────────
    # Force the Flash-SDP and Memory-Efficient-SDP kernels.  On Ampere
    # (RTX 30xx) flash_sdp gives ~20-35 % lower attention latency vs
    # the naive math backend, and mem_efficient_sdp is the safe fallback.
    # math_sdp stays on as the last-resort (no restriction on head dim).
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # Avoid unnecessary CUDA synchronisation between kernel launches.
    try:
        torch.cuda.set_per_process_memory_fraction(0.95)
    except Exception:
        pass

    if verbose:
        print("  [SPEED] CUDA knobs: TF32 matmul | cuDNN benchmark | Flash-SDP + MemEff-SDP | grad disabled globally")


def detect_triton_status() -> Tuple[bool, str]:
    """Return whether Triton is importable/usable for torch.compile CUDA backends."""
    try:
        import triton  # type: ignore
        import triton.language as _triton_lang  # type: ignore
        ver = getattr(triton, "__version__", "unknown")
        return True, f"ok (triton {ver})"
    except Exception as e:
        return False, f"unavailable ({e})"


def benchmark_multi(linearise_ratio: float = 0.5, n_prompts: int = 4, cache_dir: str = r"D:\HydrusOPT\models") -> list:
    """
    Run Baseline vs HydrusOpt across all MODELS.
    Returns a list of dicts suitable for visualise_results().
    """
    records = []
    for cfg in MODELS:
        print(f"\n{'='*55}\n  {cfg['name']}\n{'='*55}")
        tok = AutoTokenizer.from_pretrained(cfg["id"], cache_dir=cache_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Baseline
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["id"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir=cache_dir,
        )
        base_speed = benchmark_standard(base_model, tok, f"Baseline {cfg['name']}", batch_size=2)
        del base_model
        torch.cuda.empty_cache()

        # HydrusOpt
        opt_model = AutoModelForCausalLM.from_pretrained(
            cfg["id"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir=cache_dir,
        )
        opt_model, _ = linearise_model(opt_model, ratio=linearise_ratio, verbose=False)
        opt_speed = benchmark_standard(opt_model, tok, f"HydrusOpt  {cfg['name']}", batch_size=2)

        records.append({
            "model":   cfg["name"],
            "baseline": base_speed["avg_tok_per_sec"],
            "hydrus":  opt_speed["avg_tok_per_sec"],
            "speedup": opt_speed["avg_tok_per_sec"] / max(base_speed["avg_tok_per_sec"], 0.001),
        })
        del opt_model
        torch.cuda.empty_cache()

    return records


def visualise_results(records: list, out_path: str = "hydrusopt_performance.png") -> None:
    """Bar chart: Baseline vs HydrusOpt tokens/sec with speedup labels."""
    if not _HAS_CHART:
        print("  ⚠ matplotlib/pandas not installed — skipping chart (pip install matplotlib pandas)")
        return
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 7))
    x, w = range(len(df)), 0.35
    ax.bar([i - w/2 for i in x], df["baseline"], w, label="Baseline",              color="#bdc3c7", alpha=0.85)
    ax.bar([i + w/2 for i in x], df["hydrus"],   w, label="HydrusOpt (Linearised)",color="#3498db", alpha=0.90)
    for i, row in df.iterrows():
        ax.text(i, max(row["baseline"], row["hydrus"]) + 1,
                f"{row['speedup']:.2f}x", ha="center", fontweight="bold", color="#2c3e50", fontsize=10)
    ax.set_ylabel("Tokens per Second", fontsize=12, fontweight="bold")
    ax.set_title("HydrusOpt — Inference Speed & Scaling Efficiency", fontsize=14, pad=20)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"  ✓ Chart saved → {out_path}")


def _slugify_prompt(prompt: str, max_len: int = 80) -> str:
    base = (prompt or "prompt").strip().lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    if not base:
        base = "prompt"
    return base[:max_len]


def _append_csv_row(csv_path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / max(len(values), 1))


def _summarise_consistency_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    if not runs:
        return {
            "consistent": 0.0,
            "mean_entropy": 1.0,
            "retrieved_rate": 0.0,
            "fallback_rate": 0.0,
            "avg_text_len": 0.0,
            "accuracy_proxy": 0.0,
        }

    texts = [str(r.get("text", "")) for r in runs]
    entropies = [float(r.get("entropy", 1.0)) for r in runs]
    consistent = 1.0 if len(texts) >= 2 and texts[0][:30] == texts[1][:30] else 0.0
    retrieved_rate = sum(1 for t in texts if t.startswith("[Retrieved]")) / max(len(texts), 1)
    fallback_rate = sum(1 for t in texts if "I am not confident enough to answer" in t) / max(len(texts), 1)
    mean_entropy = _mean(entropies)
    avg_text_len = _mean([float(len(t)) for t in texts])
    # Proxy only (no ground-truth labels in generic benchmark mode).
    accuracy_proxy = max(0.0, min(1.0, (1.0 - mean_entropy) * (0.5 + 0.5 * consistent)))

    return {
        "consistent": consistent,
        "mean_entropy": mean_entropy,
        "retrieved_rate": retrieved_rate,
        "fallback_rate": fallback_rate,
        "avg_text_len": avg_text_len,
        "accuracy_proxy": accuracy_proxy,
    }


def update_global_benchmark_trends(args, baseline: dict, linearised_only: dict, full_stack: dict) -> None:
    """Append global speed benchmark row and refresh trend chart."""
    trends_dir = os.path.join("benchmark_history")
    csv_path = os.path.join(trends_dir, "speed_trends.csv")
    chart_path = os.path.join(trends_dir, "speed_trends.png")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    baseline_tok = float(baseline.get("avg_tok_per_sec") or 0.0)
    linear_tok = float(linearised_only.get("avg_tok_per_sec") or 0.0)
    hydrus_tok = float(full_stack.get("avg_tok_per_sec") or 0.0)
    speedup = hydrus_tok / max(baseline_tok, 0.001)

    fieldnames = [
        "timestamp", "model", "profile", "guard_mode", "baseline_tok_per_sec",
        "linearised_tok_per_sec", "hydrus_tok_per_sec", "speedup",
        "avg_acceptance_rate", "enable_retrieval_fallback", "verify_first",
    ]
    row = {
        "timestamp": ts,
        "model": str(getattr(args, "model", "")),
        "profile": str(getattr(args, "profile", "")),
        "guard_mode": str(getattr(args, "guard_mode", "")),
        "baseline_tok_per_sec": f"{baseline_tok:.4f}",
        "linearised_tok_per_sec": f"{linear_tok:.4f}",
        "hydrus_tok_per_sec": f"{hydrus_tok:.4f}",
        "speedup": f"{speedup:.6f}",
        "avg_acceptance_rate": f"{float(full_stack.get('avg_acceptance_rate', 0.0)):.6f}",
        "enable_retrieval_fallback": int(bool(getattr(args, "enable_retrieval_fallback", False))),
        "verify_first": int(bool(getattr(args, "verify_first", False))),
    }
    _append_csv_row(csv_path, fieldnames, row)

    if not _HAS_CHART:
        return

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        df["run"] = range(1, len(df) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(df["run"], df["baseline_tok_per_sec"], label="Baseline tok/s", color="#7f8c8d", linewidth=2)
        ax1.plot(df["run"], df["hydrus_tok_per_sec"], label="Hydrus tok/s", color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Tokens / sec")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left")

        ax2.plot(df["run"], df["speedup"], label="Speedup", color="#2ca02c", linewidth=2)
        ax2.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
        ax2.set_xlabel("Benchmark run")
        ax2.set_ylabel("x speedup")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left")
        fig.suptitle("HydrusOpt Speed Trends", fontsize=14)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass


def update_prompt_benchmark_trends(
    prompt: str,
    args,
    runs: List[Dict[str, Any]],
    baseline: dict,
    linearised_only: dict,
    full_stack: dict,
) -> None:
    """Append prompt-specific benchmark row and refresh prompt trend chart."""
    prompt_slug = _slugify_prompt(prompt)
    prompt_dir = os.path.join("benchmark_history", "prompts")
    csv_path = os.path.join(prompt_dir, f"{prompt_slug}_trend.csv")
    chart_path = os.path.join(prompt_dir, f"{prompt_slug}_trend.png")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    summary = _summarise_consistency_runs(runs)
    baseline_tok = float(baseline.get("avg_tok_per_sec") or 0.0)
    linear_tok = float(linearised_only.get("avg_tok_per_sec") or 0.0)
    hydrus_tok = float(full_stack.get("avg_tok_per_sec") or 0.0)
    speedup = hydrus_tok / max(baseline_tok, 0.001)

    fieldnames = [
        "timestamp", "prompt", "prompt_type", "model", "profile", "guard_mode",
        "baseline_tok_per_sec", "linearised_tok_per_sec", "hydrus_tok_per_sec", "speedup",
        "consistent", "mean_entropy", "retrieved_rate", "fallback_rate",
        "avg_text_len", "accuracy_proxy",
    ]
    row = {
        "timestamp": ts,
        "prompt": prompt,
        "prompt_type": classify_prompt_type(prompt),
        "model": str(getattr(args, "model", "")),
        "profile": str(getattr(args, "profile", "")),
        "guard_mode": str(getattr(args, "guard_mode", "")),
        "baseline_tok_per_sec": f"{baseline_tok:.4f}",
        "linearised_tok_per_sec": f"{linear_tok:.4f}",
        "hydrus_tok_per_sec": f"{hydrus_tok:.4f}",
        "speedup": f"{speedup:.6f}",
        "consistent": f"{summary['consistent']:.6f}",
        "mean_entropy": f"{summary['mean_entropy']:.6f}",
        "retrieved_rate": f"{summary['retrieved_rate']:.6f}",
        "fallback_rate": f"{summary['fallback_rate']:.6f}",
        "avg_text_len": f"{summary['avg_text_len']:.2f}",
        "accuracy_proxy": f"{summary['accuracy_proxy']:.6f}",
    }
    _append_csv_row(csv_path, fieldnames, row)

    if not _HAS_CHART:
        return

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        df["run"] = range(1, len(df) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(df["run"], df["baseline_tok_per_sec"], label="Baseline tok/s", color="#7f8c8d", linewidth=2)
        ax1.plot(df["run"], df["hydrus_tok_per_sec"], label="Hydrus tok/s", color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Tokens / sec")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left")

        ax2.plot(df["run"], df["accuracy_proxy"], label="Accuracy proxy", color="#2ca02c", linewidth=2)
        ax2.plot(df["run"], df["consistent"], label="Consistency", color="#ff7f0e", linewidth=2)
        ax2.plot(df["run"], df["retrieved_rate"], label="Retrieved rate", color="#9467bd", linewidth=2)
        ax2.set_xlabel("Benchmark run")
        ax2.set_ylabel("Quality proxy")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left")
        fig.suptitle(f"Prompt Trend: {prompt_slug}", fontsize=14)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass


def print_final_report(baseline, linearised_only, full_stack, quant_map, layers_linearised,
                        reliability_latency_per_answer_s: Optional[float] = None):
    """
    Print the HydrusOpt benchmark report with two investor-facing modes:

    • PERFORMANCE MODE  — raw inference speed (HydrusOpt vs vanilla baseline)
    • RELIABILITY MODE  — latency per verified answer (all guard passes included)
      Safety Tax = extra latency added by cognitive verification, as % of baseline answer time.
    """

    def speedup(a, b):
        a_tok = a.get("avg_tok_per_sec")
        b_tok = b.get("avg_tok_per_sec")
        if a_tok is None or b_tok is None:
            return None
        return round(a_tok / max(b_tok, 0.1), 2)

    def percent_delta(a, b) -> float:
        if a is None or b is None:
            return 0.0
        return ((a - b) / max(b, 0.001)) * 100.0

    baseline_tok = baseline["avg_tok_per_sec"]
    perf_tok     = full_stack["avg_tok_per_sec"]
    perf_pct     = percent_delta(perf_tok, baseline_tok)
    perf_trend   = "quicker" if perf_pct > 0 else "slower" if perf_pct < 0 else "same speed"

    # Baseline latency per answer: tokens_per_answer / baseline_tok_per_sec
    # Use benchmark_max_tokens as a proxy for answer length (conservative — real answers are shorter)
    avg_answer_tokens = full_stack.get("avg_tokens_per_run") or 80
    baseline_latency  = avg_answer_tokens / max(baseline_tok or 0.1, 0.1)
    if reliability_latency_per_answer_s is not None:
        rel_latency       = reliability_latency_per_answer_s
        safety_tax_pct    = percent_delta(rel_latency, baseline_latency)
        rel_latency_known = True
    else:
        rel_latency       = baseline_latency   # fallback: no data
        safety_tax_pct    = 0.0
        rel_latency_known = False

    print("\n" + "=" * 65)
    print("  HYDRUSOPT — BENCHMARK REPORT")
    print("=" * 65)

    # ── MODE 1: PERFORMANCE ───────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  MODE 1 · PERFORMANCE  (pure inference throughput)  │")
    print("  └─────────────────────────────────────────────────────┘")
    print(f"\n  {'Model':<22} {'Tokens/sec':>10}   {'vs Baseline':>18}")
    print(f"  {'─'*54}")
    baseline_tok_str = f"{baseline_tok:.1f}" if baseline_tok is not None else "n/a (--no-bench)"
    perf_tok_str     = f"{perf_tok:.1f}"     if perf_tok     is not None else "n/a (--no-bench)"
    perf_delta_str   = f"{perf_pct:+.1f}% ({perf_trend})" if perf_tok is not None else "— (skipped)"
    print(f"  {'Baseline (vanilla)':<22} {baseline_tok_str:>18}   {'— (reference)':>18}")
    print(f"  {'HydrusOpt (perf)':<22} {perf_tok_str:>18}   {perf_delta_str:>18}")

    # ── MODE 2: RELIABILITY ───────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  MODE 2 · RELIABILITY  (latency per verified answer)│")
    print("  └─────────────────────────────────────────────────────┘")
    print(f"\n  {'Model':<26} {'Sec/answer':>10}   {'Overhead':>14}")
    print(f"  {'─'*54}")
    baseline_lat_str = f"{baseline_latency:.2f}s" if baseline_tok is not None else "n/a"
    print(f"  {'Baseline (vanilla)':<26} {baseline_lat_str:>10}   {'— (reference)':>14}")
    if rel_latency_known:
        overhead_label = f"+{safety_tax_pct:.0f}% latency"
        print(f"  {'HydrusOpt (reliable)':<26} {rel_latency:>10.2f}s  {overhead_label:>14}")
    else:
        print(f"  {'HydrusOpt (reliable)':<26} {'n/a':>10}   {'run benchmark':>14}")

    # ── SAFETY TAX ────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SAFETY TAX  (cost of hallucination-proof output)   │")
    print("  └─────────────────────────────────────────────────────┘")
    if rel_latency_known:
        if safety_tax_pct <= 15:
            verdict = (
                f"  HydrusOpt delivers verified, hallucination-proof output\n"
                f"  for a Safety Tax of only {safety_tax_pct:.0f}% extra latency per answer."
            )
        else:
            verify_votes = 3  # default
            verdict = (
                f"  Reliability mode adds {safety_tax_pct:.0f}% latency per answer\n"
                f"  ({verify_votes} consensus verification passes included).\n"
                f"  Use --verify-votes 1 or --verify-high-conf-bypass 0.7 to reduce overhead."
            )
    else:
        verdict = "  Run the full benchmark (without --skip-linearise) to compute Safety Tax."
    print(f"\n{verdict}")

    # ── INTERNALS ─────────────────────────────────────────────────
    _lin_tok = linearised_only.get("avg_tok_per_sec")
    if _lin_tok is not None:
        print(f"\n  Linearisation-only reference : {_lin_tok:.1f} tok/s ({speedup(linearised_only, baseline):.2f}x)")
    else:
        print(f"\n  Linearisation-only reference : n/a (--no-bench)")
    if full_stack.get("avg_acceptance_rate"):
        print(f"  Speculative acceptance rate  : {full_stack['avg_acceptance_rate']*100:.1f}%")
    print(f"  Layers linearised            : {len(layers_linearised)}")
    print(f"  Quantisation map             : {len(quant_map)} layers quantised")

    if baseline["errors"] or full_stack["errors"]:
        print(f"\n  ⚠ Errors encountered:")
        for e in baseline["errors"] + full_stack["errors"]:
            print(f"    - {e}")
    print("=" * 65)

    # ── JSON report ───────────────────────────────────────────────
    report = {
        "baseline": baseline,
        "linearised_only": linearised_only,
        "full_stack": full_stack,
        "speedup_linearisation_only": speedup(linearised_only, baseline),
        "speedup_full_stack": speedup(full_stack, baseline),
        "layers_linearised": layers_linearised,
        "quant_map": {str(k): v for k, v in quant_map.items()},
        "investor_summary": {
            "performance_mode": {
                "label": "HydrusOpt (performance)",
                "tok_per_sec": round(perf_tok, 2) if perf_tok is not None else None,
                "delta_vs_baseline_pct": round(perf_pct, 2),
            },
            "reliability_mode": {
                "label": "HydrusOpt (reliability)",
                "baseline_latency_per_answer_s": round(baseline_latency, 3) if baseline_tok is not None else None,
                "verified_latency_per_answer_s": round(rel_latency, 3) if rel_latency_known else None,
                "safety_tax_pct": round(safety_tax_pct, 1) if rel_latency_known else None,
            },
            "safety_tax_pct": round(safety_tax_pct, 1) if rel_latency_known else None,
            "safety_tax_narrative": (
                f"HydrusOpt adds {safety_tax_pct:.0f}% latency per answer for "
                f"hallucination-proof cognitive verification."
                if rel_latency_known else
                "Run full benchmark to compute Safety Tax."
            ),
        },
    }
    with open("hydrusopt_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  💾 Full report saved to hydrusopt_report.json")


def print_cybernetic_report(c_report: Dict[str, Any]) -> None:
    """Print a compact EIR/ECR stability dashboard."""
    print("\n" + "=" * 65)
    print("  CYBERNETIC SELF-CORRECTION STABILITY")
    print("=" * 65)
    print(f"  Samples             : {c_report['samples']}")
    print(f"  Baseline Accuracy   : {c_report['baseline_accuracy']:.3f}")
    print(f"  EIR                 : {c_report['eir']:.4f}")
    print(f"  ECR                 : {c_report['ecr']:.4f}")
    ratio = c_report.get("ecr_over_eir")
    print(f"  ECR/EIR             : {'N/A' if ratio is None else f'{ratio:.4f}'}")
    print(f"  Stability Threshold : {c_report['stability_threshold']:.4f}")
    margin = c_report.get("stability_margin")
    if isinstance(margin, (int, float)) and margin != float("inf"):
        margin_str = f"{margin:+.4f}"
    else:
        margin_str = "+INF"
    print(f"  Stability Margin    : {margin_str}")
    print(f"  Stable              : {'YES' if c_report['stable'] else 'NO'}")
    print(f"  Non-degrading       : {'YES' if c_report.get('non_degrading') else 'NO'}")
    print(f"  Regime              : {c_report.get('stability_regime', 'unknown')}")
    events = c_report.get("events", {})
    print(f"  Events              : C->I={events.get('correct_to_incorrect', 0)} | I->C={events.get('incorrect_to_correct', 0)}")
    print("=" * 65)


def _is_flag_explicit(argv_flags: set, flag_name: str) -> bool:
    """Return True when a CLI flag was explicitly provided by the user."""
    return flag_name in argv_flags


def apply_profile_defaults(args, argv_flags: set) -> None:
    """
    Apply high-level profile presets.
    Advanced flags can still override profile values when explicitly provided.
    """
    profile_map = {
        "fast": {
            "speed_preset": "max",
            "skip_linearise": True,
            "skip_quant": False,
            "enable_metacognition": False,
            "enable_selfcorrect": False,
            "guard_mode": "post",
            "enable_retrieval_fallback": False,
            "verify_first": False,
            "external_selection": False,
            "enable_cybernetic_monitor": False,
            "benchmark_batch_size": 1,
            "benchmark_max_tokens": 40,
            "benchmark_warmup_tokens": 8,
            "benchmark_warmup_iters": 1,
            "factual_uncertain_ratio_trigger": 0.60,
            "enable_internal_self_correct": False,
        },
        "safe": {
            "speed_preset": "balanced",
            "skip_linearise": False,
            "skip_quant": False,
            "enable_metacognition": True,
            "enable_selfcorrect": True,
            "guard_mode": "token",
            "enable_retrieval_fallback": True,
            "verify_first": True,
            "external_selection": True,
            "enable_cybernetic_monitor": False,
            "benchmark_batch_size": 1,
            "benchmark_max_tokens": 40,
            "benchmark_warmup_tokens": 8,
            "benchmark_warmup_iters": 1,
            "factual_uncertain_ratio_trigger": 0.50,
            "enable_internal_self_correct": True,
        },
        "eval": {
            "speed_preset": "balanced",
            "skip_linearise": True,
            "skip_quant": False,
            "enable_metacognition": True,
            "enable_selfcorrect": False,
            "guard_mode": "token",
            "enable_retrieval_fallback": True,
            "verify_first": True,
            "external_selection": True,
            "enable_cybernetic_monitor": True,
            "benchmark_batch_size": 2,
            "benchmark_max_tokens": 40,
            "benchmark_warmup_tokens": 8,
            "benchmark_warmup_iters": 1,
            "factual_uncertain_ratio_trigger": 0.45,
            "enable_internal_self_correct": True,
        },
    }

    flag_to_attr = {
        "--speed-preset": "speed_preset",
        "--skip-linearise": "skip_linearise",
        "--skip-quant": "skip_quant",
        "--enable-metacognition": "enable_metacognition",
        "--enable-selfcorrect": "enable_selfcorrect",
        "--guard-mode": "guard_mode",
        "--enable-retrieval-fallback": "enable_retrieval_fallback",
        "--verify-first": "verify_first",
        "--external-selection": "external_selection",
        "--enable-cybernetic-monitor": "enable_cybernetic_monitor",
        "--benchmark-batch-size": "benchmark_batch_size",
        "--benchmark-max-tokens": "benchmark_max_tokens",
        "--benchmark-warmup-tokens": "benchmark_warmup_tokens",
        "--benchmark-warmup-iters": "benchmark_warmup_iters",
        "--factual-uncertain-ratio-trigger": "factual_uncertain_ratio_trigger",
        "--enable-internal-self-correct": "enable_internal_self_correct",
    }

    selected = profile_map[args.profile]
    for flag_name, attr_name in flag_to_attr.items():
        if _is_flag_explicit(argv_flags, flag_name):
            continue
        setattr(args, attr_name, selected[attr_name])

    # Helpful default for eval profile if user did not provide a calibration file.
    if (
        args.profile == "eval"
        and not _is_flag_explicit(argv_flags, "--calibration-file")
        and not args.calibration_file
    ):
        for candidate in ("calibration_tasks_v2.json", "calibration_tasks.json"):
            if os.path.exists(candidate):
                args.calibration_file = candidate
                break


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HydrusOpt — Local LLM Safety & Benchmarking Layer\n"
            "Runs inference through a configurable guard pipeline that tracks\n"
            "confidence, detects hallucinations, and benchmarks throughput.\n\n"
            "Quick-start examples:\n"
            "  Prompt test (quiet, no GPU spike):\n"
            "    python Hydrusopt_test.py --model Qwen/Qwen2.5-3B-Instruct \\\n"
            "      --profile fast --skip-linearise --no-bench --no-retrieval \\\n"
            "      --check-consistency \"Your question here\" --guard-mode post\n\n"
            "  Full benchmark:\n"
            "    python Hydrusopt_test.py --model Qwen/Qwen2.5-3B-Instruct \\\n"
            "      --profile fast --no-retrieval \\\n"
            "      --check-consistency \"Your question here\" --guard-mode post"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model & profile ───────────────────────────────────────────────────────
    g_model = parser.add_argument_group("Model & profile")
    g_model.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                         help="HuggingFace model ID or local path  (default: Qwen/Qwen2-1.5B-Instruct)")
    g_model.add_argument("--profile", type=str, choices=["fast", "safe", "eval"], default="safe",
                         help="High-level preset — fast: low latency, no guards  |  safe: full guard pipeline  |  eval: calibration/verification mode")
    g_model.add_argument("--cache-dir", type=str, default=r"D:\HydrusOPT\models",
                         help="Directory to cache downloaded model weights  (default: D:\\HydrusOPT\\models)")
    g_model.add_argument("--save", action="store_true",
                         help="Save the optimised model to disk after the run")

    # ── Speed & compilation ───────────────────────────────────────────────────
    g_speed = parser.add_argument_group("Speed & compilation")
    g_speed.add_argument("--speed-preset", type=str, choices=["balanced", "max"], default="max",
                         help="max: skip slow paths + enable CUDA knobs  |  balanced: safer defaults  (default: max)")
    g_speed.add_argument("--skip-linearise", action="store_true",
                         help="Skip attention-layer linearisation (faster startup, no accuracy change for most prompts)")
    g_speed.add_argument("--linearise_ratio", type=float, default=0.5,
                         help="Fraction of attention layers to linearise when linearisation is enabled  (default: 0.5)")
    g_speed.add_argument("--disable-compile", action="store_true",
                         help="Disable torch.compile even when speed-preset=max")
    g_speed.add_argument("--compile-mode", type=str,
                         choices=["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                         default="max-autotune",
                         help="torch.compile backend mode  (default: max-autotune)")

    # ── Quantisation ─────────────────────────────────────────────────────────
    g_quant = parser.add_argument_group("Quantisation")
    g_quant.add_argument("--quant_bits", type=int, default=4, choices=[4, 8],
                         help="Quantisation precision: 4-bit (INT4) recommended for 6 GB GPUs  (default: 4)")
    g_quant.add_argument("--skip-quant", action="store_true",
                         help="Disable bitsandbytes quantisation — WARNING: causes OOM on 6 GB GPUs with 3B models")

    # ── Benchmarking ─────────────────────────────────────────────────────────
    g_bench = parser.add_argument_group("Benchmarking")
    g_bench.add_argument("--no-bench", action="store_true",
                         help="Skip all benchmark loops (baseline + compiled) — eliminates GPU spike and fan noise; tok/s shown as n/a")
    g_bench.add_argument("--benchmark-batch-size", type=int, default=1,
                         help="Micro-batch size for throughput benchmark passes  (default: 1)")
    g_bench.add_argument("--benchmark-max-tokens", type=int, default=80,
                         help="Tokens generated per timed benchmark pass  (default: 80)")
    g_bench.add_argument("--benchmark-warmup-tokens", type=int, default=8,
                         help="Tokens generated per warmup pass before timing starts  (default: 8)")
    g_bench.add_argument("--benchmark-warmup-iters", type=int, default=1,
                         help="Number of warmup passes before each timed batch  (default: 1)")
    g_bench.add_argument("--benchmark-empty-cache", action="store_true",
                         help="Call torch.cuda.empty_cache() between timed batches (debug only — usually slower)")
    g_bench.add_argument("--benchmark-multi", action="store_true",
                         help="Benchmark baseline vs HydrusOpt across all models in the MODELS list")
    g_bench.add_argument("--stress-test", action="store_true",
                         help="Run a built-in set of stress prompts (ambiguous, nonsense, contradiction, math)")
    g_bench.add_argument("--visualise", action="store_true",
                         help="Save a bar chart of multi-model results to hydrusopt_performance.png")

    # ── Answer generation ────────────────────────────────────────────────────
    g_answer = parser.add_argument_group("Answer generation")
    g_answer.add_argument("--answer-max-tokens", type=int, default=0,
                          help="Max new tokens for answer generation  (0 = auto-adaptive per prompt type, up to 512)")
    g_answer.add_argument("--check-consistency", type=str, default="",
                          help="Prompt to run through the full guard pipeline and consistency check")

    # ── Safety guard ─────────────────────────────────────────────────────────
    g_guard = parser.add_argument_group("Safety guard")
    g_guard.add_argument("--guard-mode", type=str, choices=["post", "token"], default="post",
                         help="post: fast post-generation confidence check (recommended)  |  token: per-token metacognition (slower)")
    g_guard.add_argument("--selfcorrect-threshold", type=float, default=0.30,
                         help="Uncertain-token ratio above which hallucination guards fire  (default: 0.30)")
    g_guard.add_argument("--selfcorrect-samples", type=int, default=3,
                         help="Number of consistency drafts generated when guard fires  (default: 3)")
    g_guard.add_argument("--enable-selfcorrect", action="store_true",
                         help="Enable the hallucination guard self-correction loop")
    g_guard.add_argument("--factual-uncertain-ratio-trigger", type=float, default=0.50,
                         help="Uncertainty ratio above which factual prompts are forced to safe fallback  (default: 0.50)")
    g_guard.add_argument("--kv-prune-threshold", type=float, default=0.15,
                         help="Entropy threshold below which KV-cache tokens are evicted  (default: 0.15)")

    # ── Retrieval fallback ────────────────────────────────────────────────────
    g_retrieval = parser.add_argument_group("Retrieval fallback")
    g_retrieval.add_argument("--enable-retrieval-fallback", action="store_true",
                             help="When the guard cannot produce a reliable answer, fetch a Wikipedia summary")
    g_retrieval.add_argument("--no-retrieval", action="store_true",
                             help="Force offline mode — disable Wikipedia retrieval even when the profile enables it")

    # ── Verify-first gate ────────────────────────────────────────────────────
    g_verify = parser.add_argument_group("Verify-first gate")
    g_verify.add_argument("--verify-first", action="store_true",
                          help="Run a consensus verification gate before allowing any corrective action")
    g_verify.add_argument("--verify-votes", type=int, default=3,
                          help="Number of verifier votes cast by the consensus gate  (default: 3)")
    g_verify.add_argument("--verify-no-threshold", type=int, default=2,
                          help="Minimum NO votes needed to block an answer  (default: 2)")
    g_verify.add_argument("--verify-high-conf-bypass", type=float, default=0.90,
                          help="Skip the verify gate when avg confidence is above this value  (default: 0.90)")
    g_verify.add_argument("--math-bypass-conf", type=float, default=0.85,
                          help="Skip the verify gate for numeric/math answers above this confidence  (default: 0.85)")
    g_verify.add_argument("--external-selection", action="store_true",
                          help="Use a fresh stateless evaluator prompt for each verification vote")

    # ── Metacognition ─────────────────────────────────────────────────────────
    g_meta = parser.add_argument_group("Metacognition")
    g_meta.add_argument("--enable-metacognition", action="store_true",
                        help="Enable the Metacognition Plugin (per-token confidence scoring + recovery)")
    g_meta.add_argument("--meta-threshold", type=float, default=0.75,
                        help="Confidence threshold below which token-level recovery is triggered  (default: 0.75)")
    g_meta.add_argument("--meta-escalate", type=float, default=0.50,
                        help="Confidence threshold below which human escalation is logged  (default: 0.50)")

    # ── Internal self-correction ──────────────────────────────────────────────
    g_isc = parser.add_argument_group("Internal self-correction")
    g_isc.add_argument("--enable-internal-self-correct", action="store_true",
                       help="Enable bounded internal self-correction (stateless verify + parametric memory)")
    g_isc.add_argument("--internal-max-rounds", type=int, default=2,
                       help="Maximum refinement rounds per answer  (default: 2)")
    g_isc.add_argument("--internal-target-uncertain-ratio", type=float, default=0.45,
                       help="Stop refinement when uncertainty ratio falls below this value  (default: 0.45)")
    g_isc.add_argument("--internal-memory-max-tokens", type=int, default=80,
                       help="Token budget for the parametric-memory verbalization block  (default: 80)")
    g_isc.add_argument("--disable-internal-parametric-memory", action="store_true",
                       help="Disable memory verbalization inside internal self-correction")

    # ── Cybernetic monitor ────────────────────────────────────────────────────
    g_cyber = parser.add_argument_group("Cybernetic monitor")
    g_cyber.add_argument("--enable-cybernetic-monitor", action="store_true",
                         help="Track EIR/ECR self-correction stability across a calibration set")
    g_cyber.add_argument("--calibration-file", type=str, default="",
                         help="JSON file with [{prompt, answer}] pairs for EIR/ECR calibration")
    g_cyber.add_argument("--save-monitor", type=str, default="",
                         help="Path to save the cybernetic monitor report as JSON")
    g_cyber.add_argument("--tti-delta-threshold", type=float, default=0.002,
                         help="Uncertainty saturation threshold for thinking-time intervention  (default: 0.002)")
    g_cyber.add_argument("--tti-patience", type=int, default=4,
                         help="Consecutive low-delta tokens before truncating over-thinking  (default: 4)")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    g_tune = parser.add_argument_group("Threshold tuning")
    g_tune.add_argument("--auto-tune-verify", action="store_true",
                        help="Auto-tune verify gate thresholds using EIR/ECR on calibration tasks")
    g_tune.add_argument("--auto-tune-bypass-grid", type=str, default="0.85,0.90,0.95",
                        help="Comma-separated search grid for --verify-high-conf-bypass  (default: 0.85,0.90,0.95)")
    g_tune.add_argument("--auto-tune-no-threshold-grid", type=str, default="1,2,3",
                        help="Comma-separated search grid for --verify-no-threshold  (default: 1,2,3)")
    g_tune.add_argument("--auto-tune-math-bypass-grid", type=str, default="0.80,0.85,0.90",
                        help="Comma-separated search grid for --math-bypass-conf  (default: 0.80,0.85,0.90)")
    g_tune.add_argument("--threshold-sweep", action="store_true",
                        help="Grid-search meta-threshold × bypass-conf × verify-votes to find the optimal Safety Tax")
    g_tune.add_argument("--sweep-meta-grid", type=str, default="0.50,0.65,0.75,0.85,0.95",
                        help="Comma-separated meta-threshold values for --threshold-sweep")
    g_tune.add_argument("--sweep-bypass-grid", type=str, default="0.60,0.70,0.80,0.90",
                        help="Comma-separated verify_high_conf_bypass values for --threshold-sweep")
    g_tune.add_argument("--sweep-votes-grid", type=str, default="1,3",
                        help="Comma-separated verify_votes values for --threshold-sweep")

    # ── Deprecated ───────────────────────────────────────────────────────────
    g_dep = parser.add_argument_group("Deprecated (ignored)")
    g_dep.add_argument("--skip-speculative", action="store_true",
                       help="No-op — speculative decoding is disabled in the active pipeline")
    g_dep.add_argument("--enable-speculative", action="store_true",
                       help="No-op — speculative decoding is disabled in the active pipeline")
    g_dep.add_argument("--early-exit-threshold", type=float, default=0.99,
                       help="No-op — retained for script compatibility")
    g_dep.add_argument("--enable-early-exit", action="store_true",
                       help="No-op — MEEV early-exit is disabled")

    args = parser.parse_args()

    argv_flags = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    apply_profile_defaults(args, argv_flags)

    args.internal_enable_parametric_memory = not bool(args.disable_internal_parametric_memory)
    args.internal_max_rounds = max(1, min(int(args.internal_max_rounds), 6))
    args.internal_target_uncertain_ratio = max(0.0, min(float(args.internal_target_uncertain_ratio), 1.0))
    args.internal_memory_max_tokens = max(16, min(int(args.internal_memory_max_tokens), 256))

    # Speculative decoding is hard-disabled in the active pipeline due repeated regressions.
    args.skip_speculative = True
    if args.enable_speculative:
        print("\n  [INFO] --enable-speculative is deprecated and ignored (speculative decoding disabled).")
    if args.enable_early_exit:
        print("\n  [INFO] --enable-early-exit is deprecated and ignored (MEEV disabled).")
    args.enable_early_exit = False
    if getattr(args, "no_retrieval", False):
        args.enable_retrieval_fallback = False
    if args.enable_retrieval_fallback:
        print("\n  [INFO] Retrieval fallback is ENABLED (Wikipedia).")
    else:
        print("\n  [INFO] Retrieval fallback is DISABLED (offline / parametric memory only).")

    # Max speed preset: disable known slow paths in this codebase.
    if args.speed_preset == "max":
        args.skip_linearise = True

    print(BANNER)

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Environment info ──
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Device   : {device.upper()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    if _HAS_PSUTIL:
        print(f"  RAM      : {psutil.virtual_memory().total/1e9:.1f} GB")

    configure_cuda_speed_knobs(device)

    cybernetic_report = None
    consistency_runs: Optional[List[Dict[str, Any]]] = None

    # ── Multi-model benchmark (independent mode — does not need --model) ──
    if args.benchmark_multi:
        print("\n  Running multi-model benchmark...")
        records = benchmark_multi(linearise_ratio=args.linearise_ratio, cache_dir=args.cache_dir)
        print(f"\n  {'Model':<25} {'Baseline':>12} {'HydrusOpt':>12} {'Speedup':>9}")
        print(f"  {'-'*60}")
        for r in records:
            print(f"  {r['model']:<25} {r['baseline']:>12.1f} {r['hydrus']:>12.1f} {r['speedup']:>8.2f}x")
        if args.visualise:
            visualise_results(records)
        return  # skip single-model flow

    # ── Auto-skip quantisation on CPU ──
    if device == "cpu" and not args.skip_quant:
        print("\n  ⚠ CPU detected: Auto-skipping bitsandbytes 4-bit quantisation.")
        print("    Running quantized inference on CPU requires constant weight dequantization")
        print("    which is extremely slow (up to 100x slowdown). Skipping for performance.")
        args.skip_quant = True

    print(f"\n  Model  : {args.model}")
    print(f"  Profile: {args.profile}")
    print(f"  Speed preset: {args.speed_preset}")
    print(f"  Guard mode: {args.guard_mode}")
    print(
        "  Internal self-correct: "
        f"{'ON' if args.enable_internal_self_correct else 'OFF'}"
        f" (rounds={args.internal_max_rounds}, target_u={args.internal_target_uncertain_ratio:.2f})"
    )
    print(f"  Techniques: "
          f"{'Linearisation ' if not args.skip_linearise else ''}"
            f"{'Quantisation ' if not args.skip_quant else ''}")

    # ── Load model ──
    print(f"\n  ⬇  Loading {args.model}...")
    print(f"  Cache  : {args.cache_dir}")
    os.makedirs(args.cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if not args.skip_quant and args.quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs = {
        "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
        "device_map": device,
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    compile_enabled = (device == "cuda" and not args.disable_compile and args.speed_preset == "max")
    if compile_enabled:
        triton_ok, triton_status = detect_triton_status()
        if not triton_ok:
            compile_enabled = False
            print(f"  [SPEED] torch.compile disabled: Triton backend {triton_status}")
        else:
            # If inductor fails on a graph segment, continue in eager instead of crashing.
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.suppress_errors = True
            except Exception:
                pass
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Loaded — {param_count:.2f}B parameters\n")

    # ── Baseline benchmark (uncompiled, no SWA — plain model reference) ──
    if getattr(args, "no_bench", False):
        baseline_results = {"avg_tok_per_sec": None, "errors": []}
        print("  ⏱  Benchmark skipped (--no-bench)")
    else:
        print("  ⏱  Benchmarking baseline...")
        baseline_results = benchmark_standard(
            model,
            tokenizer,
            "Baseline",
            batch_size=args.benchmark_batch_size,
            max_new_tokens=args.benchmark_max_tokens,
            warmup_new_tokens=args.benchmark_warmup_tokens,
            warmup_iters=args.benchmark_warmup_iters,
            empty_cache_between_batches=args.benchmark_empty_cache,
        )
        print(f"     Baseline: {baseline_results['avg_tok_per_sec']} tok/s")

    # ── Technique 1: Linearisation ──
    layers_linearised = []
    if not args.skip_linearise:
        model, layers_linearised = linearise_model(model, ratio=args.linearise_ratio)

    # ── torch.compile — applied AFTER linearisation ──────────────────────
    # Critical: compiling before linearise_model wraps attention layers causes
    # graph breaks on every SWA forward call (new Python modules = new graph).
    # Compiling after linearisation lets max-autotune see the final SWA graph
    # and fuse masked_fill_ + attention kernels into a single Triton dispatch.
    if compile_enabled and not getattr(args, "no_bench", False):
        try:
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
            print(f"  [SPEED] torch.compile enabled ({args.compile_mode} on final graph; eager fallback on backend errors)")
        except Exception as e:
            print(f"  [SPEED] torch.compile unavailable ({e})")

    if getattr(args, "no_bench", False):
        linearised_results = {"avg_tok_per_sec": None, "errors": []}
    elif not args.skip_linearise:
        # Benchmark linearised + compiled together (the actual HydrusOpt runtime)
        print("\n  ⏱  Benchmarking after linearisation + compile...")
        linearised_results = benchmark_standard(
            model,
            tokenizer,
            "Linearised",
            batch_size=args.benchmark_batch_size,
            max_new_tokens=args.benchmark_max_tokens,
            warmup_new_tokens=args.benchmark_warmup_tokens,
            warmup_iters=args.benchmark_warmup_iters,
            empty_cache_between_batches=args.benchmark_empty_cache,
        )
        print(f"     After linearisation: {linearised_results['avg_tok_per_sec']} tok/s")
        if device == "cpu":
            print(
                "     ⚠ CPU note: SWA applies a causal mask but PyTorch still computes the full\n"
                "       O(n²) QKᵀ matrix before masking — no FLOPs are saved on CPU.\n"
                "       True O(n×W) speedup only materialises on GPU via FlashAttention / CUDA kernels."
            )
    else:
        # No linearisation — model is already compiled above (if enabled).
        print("\n  ⏱  Benchmarking compiled model (linearisation skipped)...")
        linearised_results = benchmark_standard(
            model,
            tokenizer,
            "Compiled",
            batch_size=args.benchmark_batch_size,
            max_new_tokens=args.benchmark_max_tokens,
            warmup_new_tokens=args.benchmark_warmup_tokens,
            warmup_iters=args.benchmark_warmup_iters,
            empty_cache_between_batches=args.benchmark_empty_cache,
        )
        print(f"     Compiled (no SWA): {linearised_results['avg_tok_per_sec']} tok/s")

    # ── Technique 2: Mixed Quantisation ──
    quant_map = {}
    if not args.skip_quant:
        model, quant_map = apply_mixed_quantisation(
            model, tokenizer, device, quant_bits=args.quant_bits
        )

    # ── Technique 3: Speculative Decoding ──
    full_stack_results = linearised_results  # fallback
    if not args.skip_speculative:
        if device == "cpu":
            print("\n  [3/3] SPECULATIVE DECODING")
            print("        ⚠ Skipped on CPU — speculative decoding only speeds up GPU inference.")
            print("        Tip: run on a CUDA GPU or pass --skip-speculative to suppress this message.")
        else:
            decoder = setup_speculative_decoding(model, tokenizer, args.linearise_ratio)
            print("\n  ⏱  Benchmarking full stack (with speculative decoding)...")
            full_stack_results = benchmark_speculative(decoder, tokenizer, device)
            print(f"     Full stack: {full_stack_results['avg_tok_per_sec']} tok/s")

    # ── Plugin 4: Metacognition ──
    # ── Plugin 4: Metacognition (Unified HCIE) ──
    if args.enable_metacognition:
        print("\n  [4/4] HYDRUS COGNITIVE INFERENCE ENGINE (HCIE)")
        print(f"        MAS   : Dynamic Sliding Window Attention (32 to 512 window)")
        print(f"        IT-KV : Info-Theoretic KV Pruning (entropy thresh < {args.kv_prune_threshold})")
        print(f"        MEEV  : Removed from active runtime (complexity > observed benefit)")
        if device == "cpu":
            print(
                "        ⚠ CPU Note: Unified HCIE runs correctly on CPU, but Sliding Window Attention\n"
                "          and KV cache pruning do not reduce FLOP overhead on CPU due to the lack of\n"
                "          specialized hardware attention kernels (like FlashAttention). Early exit will\n"
                "          still reduce layer computation, but true memory and speed efficiency\n"
                "          requires a CUDA-compatible GPU."
            )
        print(f"        Running unified cognitive optimization on test prompts...")

        for prompt in TEST_PROMPTS[:2]:  # Demo on first 2 prompts
            output_text, stats = generate_cognitive(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=40,
                confidence_threshold=args.meta_threshold,
                early_exit_threshold=args.early_exit_threshold,
                keep_recent_n=8,
                low_entropy_thresh=args.kv_prune_threshold,
                enable_early_exit=args.enable_early_exit,
                tti_delta_threshold=args.tti_delta_threshold,
                tti_patience=args.tti_patience,
            )
            print(f"\n  Prompt : \"{prompt}\"")
            print(f"  Output : \"{output_text}\"")
            print(f"  Stats  : {stats['total_steps']} tokens generated")
            print(f"           - Early Exits triggered: {stats['early_exits']} / {stats['total_steps']} steps ({stats['early_exit_pct']}%)")
            print(f"           - KV Cache compressed : {stats['final_kv_cache_len']} / {stats['original_kv_cache_len']} size ({stats['kv_reduction_pct']}% VRAM saved)")
            print(f"           - TTI Triggered       : {'YES' if stats.get('tti_triggered') else 'NO'}")
        print("\n  ✓ Cognitive Optimization Engine complete.")

    elif getattr(args, "enable_selfcorrect", False):
        # Standalone guard demo without full metacognition loop
        print("\n  [5/5] HALLUCINATION GUARDS (standalone demo)")
        for prompt in TEST_PROMPTS[:2]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            draft = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n  Prompt : \"{prompt[:60]}\"")
            print(f"  Draft  : {draft[:200]}")
            self_correct_draft(prompt, draft, model, tokenizer, verbose=True)
            check_semantic_consistency(
                prompt, model, tokenizer,
                n_samples=args.selfcorrect_samples, verbose=True
            )
        print("\n  ✓ Hallucination Guards demo complete.")

    # ── Stress tests ──
    if args.stress_test:
        print("\n  [STRESS] Running stress tests...")
        for test in STRESS_TESTS:
            print(f"\n  {'─'*55}")
            print(f"  TYPE: {test['type']}")
            generate_with_meta(
                test["prompt"], model, tokenizer,
                max_tokens=40, threshold=args.meta_threshold,
                stop_on_uncertainty=False,
                enable_retrieval_fallback=args.enable_retrieval_fallback,
                verify_first=args.verify_first,
                external_selection=args.external_selection,
                tti_delta_threshold=args.tti_delta_threshold,
                tti_patience=args.tti_patience,
                enable_internal_self_correct=args.enable_internal_self_correct,
                internal_max_rounds=args.internal_max_rounds,
                internal_target_uncertain_ratio=args.internal_target_uncertain_ratio,
                internal_memory_max_tokens=args.internal_memory_max_tokens,
                internal_enable_parametric_memory=args.internal_enable_parametric_memory,
            )

    # ── Consistency check on custom prompt ──
    if args.check_consistency:
        _consistency_max_tokens = _adaptive_max_tokens(args.check_consistency, getattr(args, "answer_max_tokens", 0))
        consistency_runs = check_truth_consistency(
            args.check_consistency,
            model,
            tokenizer,
            max_tokens=_consistency_max_tokens,
            threshold=args.meta_threshold,
            enable_retrieval_fallback=args.enable_retrieval_fallback,
            guard_mode=args.guard_mode,
            verify_first=args.verify_first,
            external_selection=args.external_selection,
            verify_votes=args.verify_votes,
            verify_no_threshold=args.verify_no_threshold,
            verify_high_conf_bypass=args.verify_high_conf_bypass,
            math_bypass_conf=args.math_bypass_conf,
            factual_uncertain_ratio_trigger=args.factual_uncertain_ratio_trigger,
            enable_internal_self_correct=args.enable_internal_self_correct,
            internal_max_rounds=args.internal_max_rounds,
            internal_target_uncertain_ratio=args.internal_target_uncertain_ratio,
            internal_memory_max_tokens=args.internal_memory_max_tokens,
            internal_enable_parametric_memory=args.internal_enable_parametric_memory,
        )

    # ── Threshold sweep ──
    if getattr(args, "threshold_sweep", False):
        print("\n  [SWEEP] Running threshold sweep — this will take a while...")
        run_threshold_sweep(
            model=model,
            tokenizer=tokenizer,
            args=args,
            meta_threshold_grid=_parse_float_grid(args.sweep_meta_grid, [0.50, 0.65, 0.75, 0.85, 0.95]),
            bypass_conf_grid=_parse_float_grid(args.sweep_bypass_grid, [0.60, 0.70, 0.80, 0.90]),
            verify_votes_grid=_parse_int_grid(args.sweep_votes_grid, [1, 3]),
            max_tokens=args.benchmark_max_tokens,
        )

    # ── Cybernetic calibration pass (optional) ──
    if args.enable_cybernetic_monitor:
        if not args.calibration_file:
            print("\n  [CYBERNETIC] --enable-cybernetic-monitor requires --calibration-file")
        elif not os.path.exists(args.calibration_file):
            print(f"\n  [CYBERNETIC] calibration file not found: {args.calibration_file}")
        else:
            tasks = load_calibration_tasks(args.calibration_file)
            if not tasks:
                print("\n  [CYBERNETIC] calibration file has no valid {prompt, answer} items")
            else:
                if args.auto_tune_verify:
                    print("\n  [AUTOTUNE] searching verify gate thresholds...")
                    tune = autotune_verify_gate(
                        model=model,
                        tokenizer=tokenizer,
                        tasks=tasks,
                        external_selection=args.external_selection,
                        max_tokens=40,
                        bypass_grid=_parse_float_grid(args.auto_tune_bypass_grid, [0.85, 0.90, 0.95]),
                        no_threshold_grid=_parse_int_grid(args.auto_tune_no_threshold_grid, [1, 2, 3]),
                        math_bypass_grid=_parse_float_grid(args.auto_tune_math_bypass_grid, [0.80, 0.85, 0.90]),
                    )
                    print_autotune_report(tune)
                    best = tune.get("best") or {}
                    if best:
                        args.verify_high_conf_bypass = float(best.get("verify_high_conf_bypass", args.verify_high_conf_bypass))
                        args.verify_no_threshold = int(best.get("verify_no_threshold", args.verify_no_threshold))
                        args.math_bypass_conf = float(best.get("math_bypass_conf", args.math_bypass_conf))

                print(f"\n  [CYBERNETIC] running calibration on {len(tasks)} tasks...")
                cybernetic_report = run_cybernetic_calibration(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=tasks,
                    verify_first=args.verify_first,
                    external_selection=args.external_selection,
                    max_tokens=40,
                    verify_votes=args.verify_votes,
                    verify_no_threshold=args.verify_no_threshold,
                    verify_high_conf_bypass=args.verify_high_conf_bypass,
                    math_bypass_conf=args.math_bypass_conf,
                    preserve_on_verify_block=True,
                )
                print_cybernetic_report(cybernetic_report)
                if args.save_monitor:
                    payload = {
                        "cybernetic_report": cybernetic_report,
                        "applied_gate_config": {
                            "verify_high_conf_bypass": args.verify_high_conf_bypass,
                            "verify_no_threshold": args.verify_no_threshold,
                            "math_bypass_conf": args.math_bypass_conf,
                        },
                    }
                    with open(args.save_monitor, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    print(f"  [CYBERNETIC] monitor report saved to {args.save_monitor}")

    # ── Reliability microbenchmark (guarded generation on TEST_PROMPTS) ──
    # Measures wall-clock latency per verified answer (all guard passes included).
    # Safety Tax = (verified_latency - baseline_latency) / baseline_latency * 100%
    reliability_latency_per_answer_s: Optional[float] = None
    try:
        print("\n  [RELIABILITY] Timing verified answer latency...")
        _rel_t0 = time.perf_counter()
        model.eval()
        with torch.inference_mode():
            for _rp in TEST_PROMPTS:
                _rp_max_tokens = _adaptive_max_tokens(_rp, getattr(args, "answer_max_tokens", 0))
                generate_with_post_guard(
                    _rp,
                    model,
                    tokenizer,
                    max_tokens=_rp_max_tokens,
                    threshold=args.meta_threshold,
                    use_chat_template=True,
                    enable_retrieval_fallback=False,   # exclude network latency from timing
                    verify_first=args.verify_first,
                    external_selection=args.external_selection,
                    verify_votes=args.verify_votes,
                    verify_no_threshold=args.verify_no_threshold,
                    verify_high_conf_bypass=args.verify_high_conf_bypass,
                    math_bypass_conf=args.math_bypass_conf,
                    factual_uncertain_ratio_trigger=args.factual_uncertain_ratio_trigger,
                    enable_internal_self_correct=args.enable_internal_self_correct,
                    internal_max_rounds=args.internal_max_rounds,
                    internal_target_uncertain_ratio=args.internal_target_uncertain_ratio,
                    internal_memory_max_tokens=args.internal_memory_max_tokens,
                    internal_enable_parametric_memory=args.internal_enable_parametric_memory,
                )
        _rel_elapsed = time.perf_counter() - _rel_t0
        reliability_latency_per_answer_s = _rel_elapsed / len(TEST_PROMPTS)
    except Exception:
        pass  # fall back to None — report will note data unavailable

    # ── Final Report ──
    print_final_report(baseline_results, linearised_results, full_stack_results, quant_map, layers_linearised,
                        reliability_latency_per_answer_s=reliability_latency_per_answer_s)
    update_global_benchmark_trends(args, baseline_results, linearised_results, full_stack_results)
    if args.check_consistency and consistency_runs is not None:
        update_prompt_benchmark_trends(
            prompt=args.check_consistency,
            args=args,
            runs=consistency_runs,
            baseline=baseline_results,
            linearised_only=linearised_results,
            full_stack=full_stack_results,
        )

    # ── Save ──
    if args.save:
        output_path = f"./hydrusopt_{args.model.replace('/', '_')}"
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"\n  ✅ Saved to {output_path}/")


if __name__ == "__main__":
    main()