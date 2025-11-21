
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

* **benchmark_results.csv** 
* **Console table:** 

Example output on a 5090 with ["text-embedding-qwen3-embedding-8b","text-embedding-qwen3-embedding-4b","text-embedding-qwen3-embedding-0.6b@f16","text-embedding-qwen3-embedding-0.6b@q8_0",]:

```
corpus   model                                    phrases  tok/phrase tok/call     min_s    p50_s    p90_s    max_s    avg_s    tok/s        phrases/s
----------------------------------------------------------------------------------------------------------------------------------------------------------
short    text-embedding-qwen3-embedding-8b        1141     1.1034     1259         13.9553  14.0505  14.4612  14.3994  14.1372  89.0559      80.7091     
short    text-embedding-qwen3-embedding-4b        1141     1.1034     1259         11.4922  11.6471  11.7626  11.7310  11.6231  108.3183     98.1662     
short    text-embedding-qwen3-embedding-0.6b@f16  1141     1.1034     1259         7.6599   7.6914   7.8926   7.8390   7.7169   163.1483     147.8572    
short    text-embedding-qwen3-embedding-0.6b@q8_0 1141     1.1034     1259         8.1517   8.2319   8.2677   8.2587   8.2215   153.1342     138.7817    
medium   text-embedding-qwen3-embedding-8b        163      9.5767     1561         2.4518   2.4853   2.5039   2.4990   2.4778   630.0052     65.7853     
medium   text-embedding-qwen3-embedding-4b        163      9.5767     1561         2.1725   2.1746   2.2071   2.1992   2.1798   716.1114     74.7765     
medium   text-embedding-qwen3-embedding-0.6b@f16  163      9.5767     1561         1.1251   1.1431   1.1659   1.1618   1.1437   1364.8765    142.5207    
medium   text-embedding-qwen3-embedding-0.6b@q8_0 163      9.5767     1561         1.5243   1.5586   1.6157   1.6019   1.5570   1002.5468    104.6862    
long     text-embedding-qwen3-embedding-8b        16       189.9375   3039         0.5455   0.5587   0.5675   0.5651   0.5564   5461.7966    28.7558     
long     text-embedding-qwen3-embedding-4b        16       189.9375   3039         0.3989   0.4047   0.4322   0.4278   0.4094   7422.8757    39.0806     
long     text-embedding-qwen3-embedding-0.6b@f16  16       189.9375   3039         0.2422   0.2493   0.2603   0.2590   0.2506   12127.5236   63.8501     
long     text-embedding-qwen3-embedding-0.6b@q8_0 16       189.9375   3039         0.2506   0.2516   0.3080   0.2955   0.2626   11573.1881   60.9316     
```

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

