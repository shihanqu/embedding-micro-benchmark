
# Embedding Micro Bench

Tiny embedding enchmarker. Use it to measure model latency, token load, throughput.  
Compare short/medium/long corpora.  Spot slowdowns, bottlenecks, drift.
Designed for local endpoints but can be easily modded for cloud API services.

## What it does
- Runs warmups + timed runs  
- Benchmarks multiple models × multiple corpora  
- Prints aligned console table  
- Writes CSV for quick analysis  
- Streams verbose progress so you know the exact step

## Key metrics
- min / max / avg latency  
- p50 / p90 tail times  
- tokens per call  
- tokens/sec  
- phrases/sec  

## How to use
```bash
python benchmark.py
````

Outputs:

* **Console table** (fast scan)
* **benchmark_results.csv** (deeper dive)

## Structure

```
benchmark.py
  ├─ CORPORA: short / medium / long
  ├─ MODELS: editable list
  ├─ run_all(): corpus × model loop
  ├─ benchmark_model(): warmups + timed runs
  ├─ print_results_table(): console
  └─ write_results_csv(): csv
```

## Notes

* Designed for local LLM/embedding endpoints
* Verbose logs show heartbeat at all stages
* Keep runs/corpora tight for quick testing
* Expand corpora or models freely; code scales cleanly

## License

MIT

```

