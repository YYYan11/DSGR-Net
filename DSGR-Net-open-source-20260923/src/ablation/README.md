# Exact ablation snapshots

These files preserve the source snapshots used for the paper comparisons. They intentionally remain separate because the variants changed the forward architecture, rather than only a configuration flag.

| File | Paper role |
|---|---|
| `strong_environment.py` | strong environment representation baseline |
| `direct_irrigation.py` | direct-irrigation input baseline |
| `process_gate_v13.py` | sigmoid process-gate ablation |
| `gradient_informed.py` | gradient-informed comparison |
| `temporal_cnn.py` | temporal CNN comparison |
| `multiscale_process.py` | multi-scale process comparison |
| `hybrid_nn_cnn.py` | hybrid NN-CNN comparison |

The main proposed model is maintained only in `src/dsgr_net.py`.
