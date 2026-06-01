# HydrusOPT

A local LLM safety and benchmarking layer for Hugging Face models. Wraps inference with consistency checking, uncertainty guards, self-correction, and speed benchmarking — all running fully offline on consumer GPU hardware.

---

## What it does

- Runs a prompt through a local model multiple times and checks whether outputs are **consistent**
- Scores each answer with a **confidence value** and **Shannon entropy (H)**
- Applies a **safety guard** (pre/post) that can block high-uncertainty or factually suspect outputs
- Optionally runs **self-correction** rounds to reduce uncertainty before returning an answer
- Benchmarks **tokens/sec** across baseline and compiled (torch.compile) inference paths
- Tracks **per-prompt speed trends** over time in `benchmark_history/`
- Supports **calibration files** for tuning guard thresholds per prompt type

---

## Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA
- NVIDIA GPU — tested on **RTX 3060 Laptop 6GB**
- `bitsandbytes` INT4 quantization is **mandatory** on 6 GB VRAM (unquantized will OOM)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers bitsandbytes accelerate
```

---

## Quick start

**Fast mode** (no benchmark, laptop-friendly):
```powershell
python Hydrusopt_test.py --model Qwen/Qwen2.5-3B-Instruct --profile fast --skip-linearise --no-bench --no-retrieval --check-consistency "Your question/prompt here" --guard-mode post
```

**Full benchmark run:**
```powershell
python Hydrusopt_test.py --model Qwen/Qwen2.5-3B-Instruct --profile fast --no-retrieval --check-consistency "Your question/prompt here" --guard-mode post
```

Models are cached to `./models/` by default. First run will download the model weights from Hugging Face.

---

## Profiles

| Profile | Speed preset | Self-correct | Verify-first | Guard |
|---------|-------------|--------------|--------------|-------|
| `fast`  | max         | off          | off          | post  |
| `safe`  | balanced    | on           | on           | pre   |
| `eval`  | balanced    | on           | on           | post  |

Select with `--profile fast` / `--profile safe` / `--profile eval`.

---

## Key flags

| Flag | Description |
|------|-------------|
| `--model` | Hugging Face model ID (e.g. `Qwen/Qwen2.5-3B-Instruct`) |
| `--profile` | Preset: `fast`, `safe`, `eval` |
| `--no-bench` | Skip all benchmark passes (faster, no tok/s report) |
| `--skip-linearise` | Skip torch.compile linearisation step |
| `--check-consistency` | Run multi-sample consistency check |
| `--guard-mode` | `pre` (block before answer) or `post` (block after) |
| `--quant_bits` | Quantisation bits: `4` (default) or `8` |
| `--no-retrieval` | Disable Wikipedia retrieval fallback |
| `--calibration-file` | Path to JSON calibration file for threshold tuning |
| `--visualise` | Plot speed trend charts after benchmarking |

Run `python Hydrusopt_test.py --help` for the full grouped flag reference.

---

## Understanding the output

```
CONSISTENT  confidence=0.8252  H=0.175  match=3/3
```

| Field | Meaning |
|-------|---------|
| `CONSISTENT` / `BLOCKED` | Whether the guard passed or rejected the answer |
| `confidence` | 0–1 score; higher = more certain |
| `H` | Shannon entropy across samples; lower = more consistent |
| `match` | How many of the N samples agreed |
| `tok/s` | Tokens per second (baseline vs compiled) |

---

## Tested hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 3060 Laptop (6 GB VRAM) |
| Quantization | INT4 via `bitsandbytes` (required) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| OS | Windows 11 |

---

## Project structure

```
Hydrusopt_test.py       # Main pipeline
Hydrusopt.py            # Supporting utilities
batch_test.py           # Run multiple edge-case prompts in batch
benchmark_history/      # Speed trend CSVs and plots per prompt
calibration_tasks.json  # Example calibration task definitions
```

---

## Notes

- `models/` is excluded from git — weights are re-downloaded automatically via Hugging Face
- Report JSON files (e.g. `hydrus_cybernetic_report_latest.json`) are saved after each run
- The system prompt instructs the model not to add hashtags or social media formatting

---

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
