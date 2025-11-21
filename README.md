# Embedding‑Micro-Benchmark  

A lightweight script that measures latency & throughput of text‑embedding APIs for arbitrary models.

---

## Overview  
* builds a deterministic phrase corpus (default: 1 141 single‑word phrases)  
* tokenises with a HuggingFace tokenizer to obtain exact token counts  
* sends the whole batch in one HTTP POST (`/v1/embeddings`) per model  
* performs warm‑up runs, then records N measured runs  
* reports min / p50 / p90 / max latency, average tokens / second and phrases / second  

---

## Requirements  

| Package | Install |
|---------|--------|
| Python ≥3.8 | `python -m venv .venv && source .venv/bin/activate` |
| requests | `pip install requests` |
| transformers | `pip install transformers` |

The endpoint must follow the OpenAI‑compatible `/v1/embeddings` contract.

---

## Quick Start  

```bash
# clone & enter repo
git clone https://github.com/<user>/embedding-benchmark.git
cd embedding-benchmark

# install deps
pip install -r requirements.txt   # or the two pip commands above

# run (default config)
python benchmark.py
```

The script prints corpus statistics, per‑model latency logs, and a final summary table.

---

## Configuration  

Edit **benchmark.py** to adapt:

| Variable | Meaning |
|----------|---------|
| `URL` | HTTP endpoint (`http://host:port/v1/embeddings`) |
| `MODELS` | List of model identifiers to test |
| `TOKENIZER_NAME` | HuggingFace tokenizer matching the embedding model |
| `BASE_PHRASES_COUNT`, `MULTIPLIER` | Controls total phrase count (`TARGET_COUNT = BASE × MULTIPLIER`) |
| `phrases` block | Replace with a custom corpus (short, medium, or long) – three examples are commented out. |
| `N_WARMUP`, `N_RUNS` | Warm‑up & measured iteration counts |
| `timeout` in `requests.post` | Adjust for slow models |

---

## Output  

Example of actual outpot for ["text-embedding-qwen3-embedding-8b", "text-embedding-qwen3-embedding-4b",] served from LM Studio on an RTX 5090.

```
=== Summary (Short Phrases) ===

Model: text-embedding-qwen3-embedding-8b
  Calls:               10
  Phrases per call:    1141
  Avg tokens/phrase:   1.10
  Tokens per call:     1259
  Min latency:         14.6248 s
  P50 latency:         14.9628 s
  P90 latency:         15.2211 s
  Max latency:         15.2436 s
  Avg latency:         14.9270 s
  Tokens / second:     84.3
  Phrases / second:    76.44

Model: text-embedding-qwen3-embedding-4b
  Calls:               10
  Phrases per call:    1141
  Avg tokens/phrase:   1.10
  Tokens per call:     1259
  Min latency:         13.0931 s
  P50 latency:         13.2373 s
  P90 latency:         13.6234 s
  Max latency:         13.6368 s
  Avg latency:         13.2757 s
  Tokens / second:     94.8
  Phrases / second:    85.95

=== Summary (Medium Phrases) ===

Model: text-embedding-qwen3-embedding-8b
  Calls:               10
  Phrases per call:    163
  Avg tokens/phrase:   9.58
  Tokens per call:     1561
  Min latency:         2.4563 s
  P50 latency:         2.5872 s
  P90 latency:         2.6525 s
  Max latency:         2.6569 s
  Avg latency:         2.5750 s
  Tokens / second:     606.2
  Phrases / second:    63.30

Model: text-embedding-qwen3-embedding-4b
  Calls:               10
  Phrases per call:    163
  Avg tokens/phrase:   9.58
  Tokens per call:     1561
  Min latency:         2.1920 s
  P50 latency:         2.2225 s
  P90 latency:         2.5480 s
  Max latency:         2.5657 s
  Avg latency:         2.2728 s
  Tokens / second:     686.8
  Phrases / second:    71.72

=== Summary (Long Phrases) ===

Model: text-embedding-qwen3-embedding-8b
  Calls:               10
  Phrases per call:    16
  Avg tokens/phrase:   189.94
  Tokens per call:     3039
  Min latency:         0.5378 s
  P50 latency:         0.5444 s
  P90 latency:         0.5640 s
  Max latency:         0.5652 s
  Avg latency:         0.5464 s
  Tokens / second:     5562.1
  Phrases / second:    29.28

Model: text-embedding-qwen3-embedding-4b
  Calls:               10
  Phrases per call:    16
  Avg tokens/phrase:   189.94
  Tokens per call:     3039
  Min latency:         0.3926 s
  P50 latency:         0.4100 s
  P90 latency:         0.4151 s
  Max latency:         0.4152 s
  Avg latency:         0.4086 s
  Tokens / second:     7438.3
  Phrases / second:    39.16
```

All fields are stored in a dictionary; you can export to JSON/CSV by extending `main()`.

---

## Extending  

* **Custom tokeniser** – replace `AutoTokenizer.from_pretrained` with the exact model tokenizer.  
* **Different batch size** – modify `TARGET_COUNT` or supply your own `phrases`.  
* **Additional metrics** – capture response payload size, HTTP status codes, etc., inside `call_embedding`.  
* **Authentication** – add headers to `requests.post` (`headers={"Authorization": "Bearer <token>"}`).

