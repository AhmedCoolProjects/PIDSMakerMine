Here's the per-commit summary of what changed and the results:
Baseline: Vanilla Velox (7fc9e29)
- Standard node-level prediction only, no edge vector
- CS_E3: ADP=0.167, PRE=0.0015, TP=1, FP=666
- CA_E3: ADP=0.343, PRE=1, TP=3, FP=0
---
V1: First edge vector (2a58136)
Config: num_features=6, no edge_vector_proj_dim
Edge vector: concat(src_type, dst_type, src_emb(128d), dst_emb(128d), 6 temporal) = 268 raw dims → fed directly to MLP
6 features: t_norm, min(delta_src,1), min(delta_dst,1), src_freq/global, dst_freq/global, log10(total+1)/6
Key issue: 268-dim vector with redundant full embeddings → MLP bloat, temporal info diluted
---
V2: Causal features + removed embeddings (c80928e)
Config: num_features=8, edge_vector_proj_dim=32
Edge vector: concat(src_type, dst_type, 8 temporal) = 14 raw dims → projected to 32
Key changes:
- Removed src_emb/dst_emb (268→14, no redundancy)
- Added linear projection layer (14→32) before MLP concat
- Causal features: sorted by time, computed online (sliding counters), not global stats
- Zero-filled edge vector for non-edge-vector objectives (decoder re-use fix)
8 features: t_norm, log_delta_prev, log_delta_prev2, log_delta_same_pair, pair_freq_causal, src_freq_causal, src_type_diversity, prev_op_norm
CS_E3: ADP=0.12, PRE=0.02326 (15× vs vanilla), FP=42 (94% reduction), TP=1
CA_E3: TP=0 (first CADETS failure — all edge variants lost detections)
---
V5: Novelty signals + higher projection (42857f0)
Config: num_features=10, edge_vector_proj_dim=64
Key changes:
- Replaced prev_op_norm with 3 novelty signals: edge_type_rarity, is_new_pair, is_new_type_for_src
- Increased projection dim 32→64
10 features: kept 7 from V2 + 3 novelty
CS_E3: ADP=0.333 (best rank) but TP=0 (threshold too strict — loss compressed too much)
CA_E3: ADP up to 0.637 but still TP=0
---
Current: Parallel decoder (uncommitted)
Config: num_features=8, edge_vector_proj_dim=32, edge_loss_lambda=0.3
Key changes:
- Removed log_delta_prev2, src_freq, src_type_diversity (all 3 hurt detection)
- Added is_new_type_for_dst (symmetric src/dst novelty)
- Parallel decoder: loss = loss_node + 0.3 * loss_edge
- Dual MLP heads (node-only vs edge-vector)
CS_E3 so far: ADP=0.125, PRE=0.00007, FP=13K+ — worst ever. Node-only head noise destroyed threshold.
---
Key Patterns
Version	CS_E3 PRE	CS_E3 FP
Vanilla	0.0015	666
V1	?	?
V2	0.02326	42
V5	? (TP=0)	?
Current	0.00007	13649
The fundamental tension: Edge features improve CLEARSCOPE precision but destroy CADETS entirely (TP=0 across ALL edge variants). The parallel decoder approach made things worse by adding noise from the node-only head.
