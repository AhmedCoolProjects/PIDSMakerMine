# PIDSMakerFE — Feature Engineering Living Document

> **Branch**: `feature-engineering` | **Conda Env (HPC)**: `p_fe` | **Last Updated**: 2026-05-12

---

## 1. Project Goal

This project systematically engineers edge-level features to enhance **provenance-based intrusion detection systems (PIDS)**. The goal is to improve detection of Advanced Persistent Threats (APTs) in provenance graph data by providing rich, informative edge-level inputs that complement existing node embeddings.

### Core Tension Discovered

```
Better edge features → lower loss on normal edges → threshold drops → TP↓
```

Edge features that help predict normal edges *too well* compress the loss distribution. Since anomaly detection uses `max_val_loss` as threshold (maximum per-edge loss on validation data), a compressed distribution lowers the threshold below attack-edge detectability.

**Challenge**: Engineer features that widen the normal-edge loss variance *while* keeping attack-edge loss high.

### Architecture Context

- **Framework**: PIDSMaker — modular pipeline:
  `construction → transformation → featurization → feat_inference → batching → training → evaluation → triage`

- **Base System**: Velox (USENIX Security 2025)
  - Node embeddings via Word2Vec
  - Linear encoder
  - Edge MLP decoder
  - Self-supervised: `EdgeTypePrediction` objective

- **Our Variant**: `velox-edge`
  - Inherits from Velox
  - Adds `edge_vector = concat(src_type_onehot, dst_type_onehot, temporal_feats)`
  - Projects via `lin_edge: Linear(16, 32)` before MLP concat
  - Total MLP input: 256 (src_proj) + 256 (dst_proj) + 32 (edge_proj) = 544 dims

---

## 2. What is Velox-Edge?

Velox-edge is an experimental variant of Velox that injects **engineered edge-level feature vectors** into the decoder MLP.

### How It Works (End-to-End)

| Stage | Location | What Happens |
|-------|----------|--------------|
| **Construction** | `build_default_graphs.py:377-419` | Raw edges sorted by time; 10 causal features computed; stored as `edge["temporal_feats"]` |
| **Feat Inference** | `feat_inference.py:26-83` | Features carried to `CollatableTemporalData.temporal_feats`; zero-filled if missing |
| **Batching** | `data_utils.py:290-299` | `edge_vector = concat(src_type(3), dst_type(3), temporal_feats(10))` = 16 dims |
| **Factory** | `factory.py:760-785` | Creates `lin_edge: Linear(16, 32)` projection layer |
| **Decoder** | `custom_edge_mlp_decoder.py` | `h = concat(lin_src(h_src), lin_dst(h_dst), lin_edge(edge_vector))` |
| **Objective** | `predict_edge_type.py` | Cross-entropy loss on edge type prediction |

### Config Inheritance Chain

```
default.yml
  └── orthrus.yml
        └── orthrus_non_snooped.yml
              └── velox.yml
                    └── velox-edge.yml  ← OURS
```

---

## 3. Currently Implemented Features

All features are implemented in `build_default_graphs.py` (helper functions at lines 180-225, computation loop starting at line 425). All are computed **causally**: edges sorted by time, using only information from *before* the current edge.

**Feature selection** is now config-driven via `feature_names` list in `velox-edge.yml`. `num_features` is auto-derived from `len(feature_names)` when set.

### State Variables (Maintained Causally)

```python
# Existing
last_time              # previous edge timestamp (any src/dst)
last_pair_time         # last timestamp per (src, dst)
last_src_time          # last timestamp per src
pair_count             # Counter[(src,dst)]
type_count             # Counter[op_id]
max_type_count         # max(type_count.values())
src_uniq_types         # defaultdict(set) — edge types per src
src_unique_dsts        # defaultdict(set) — unique dsts per src
pair_types             # defaultdict(set) — edge types per (src,dst)
normalizer = e_idx     # causal — edges seen so far

# New
last_dst_time          # last timestamp per dst
dst_unique_srcs        # defaultdict(set) — unique srcs per dst
seen_dst_labels        # set of dst label strings seen so far
last_write_time        # last EVENT_WRITE timestamp per dst
src_recent_new_pairs   # defaultdict(deque(maxlen=50)) — rolling new-pair history per src
```

### Feature Table — All Available Features

| Name | Category | Formula | Range | Grounding |
|------|----------|---------|-------|-----------|
| `t_norm` | Temporal | `(t - start_time) / window_size_ns` | [0,1] | Window position |
| `log_delta_prev` | Temporal | `log1p((t - last_time) / 1e9)` | [0,~16] | Global burstiness |
| `log_delta_same_pair` | Temporal | `log1p((t - last_pair_time.get((s,d), t)) / 1e9)` | [0,~16] | Pair-level timing |
| `log_delta_src_activity` | Temporal | `log1p((t - last_src_time.get(src, t)) / 1e9)` | [0,~16] | Source dormancy→burst |
| `log_delta_dst_activity` | Temporal | `log1p((t - last_dst_time.get(dst, t)) / 1e9)` | [0,~16] | Dest idle time |
| `pair_freq_causal` | Volumetric | `pair_count[(s,d)] / normalizer` | [0,1] | Pair frequency |
| `src_unique_dsts_norm` | Volumetric | `len(src_unique_dsts[src]) / normalizer` | [0,1] | Source spread |
| `pair_type_count_norm` | Volumetric | `len(pair_types[(s,d)]) / normalizer` | [0,1] | Relationship diversity |
| `src_burst_new_pairs` | Volumetric | `sum(src_recent_new_pairs[src]) / 50.0` | [0,1] | Density of new pairs from src |
| `type_rarity` | Novelty | `1 - type_count[op] / max(max_type_count,1)` | [0,1] | Edge type rarity |
| `is_new_pair` | Novelty | `1.0 if (s,d) not in last_pair_time` | {0,1} | First-time pair |
| `is_new_type_for_src` | Novelty | `1.0 if op not in src_uniq_types[src]` | {0,1} | New op type for src |
| `is_new_type_for_pair` | Novelty | `1.0 if op not in pair_types[(s,d)]` | {0,1} | New op type for this pair |
| `is_new_src_for_dst` | Novelty | `1.0 if src not in dst_unique_srcs[dst]` | {0,1} | New src accessing dst |
| `is_new_path_global` | Novelty | `1.0 if dst_label not in seen_dst_labels` | {0,1} | dst path never seen before |
| `is_tmp_dst` | Path-based | `1.0 if '/tmp/' in dst_label` | {0,1} | CADETS: malware staging in /tmp |
| `is_hash_filename_dst` | Path-based | `1.0 if filename ≥32 chars all hex` | {0,1} | CLEARSCOPE: SHA-1 cache entries |
| `is_server_port_outbound` | Path-based | `1.0 if netflow dst AND local_port ≤ 1024` | {0,1} | CADETS: nginx:80 outbound |
| `is_execute_from_tmp` | Path-based | `1.0 if EVENT_EXECUTE AND '/tmp/' in dst_label` | {0,1} | Execute from /tmp |
| `write_then_execute_flag` | Causal | `1.0 if EVENT_EXECUTE AND dst written within 300s` | {0,1} | CADETS: write→execute chain |

### Edge Vector Assembly

```python
edge_vector = torch.cat([
    src_type_onehot,  # 3 dims
    dst_type_onehot,  # 3 dims
    temporal_feats,   # len(feature_names) dims — selected by config
], dim=-1)
```

`num_features` auto-derived from `len(feature_names)` in config; `lin_edge` projects to `edge_vector_proj_dim`.

---

## 4. Datasets

### Primary Datasets (DARPA TC E3/E5)

| Dataset | OS | Attacks (Used) | Size | Attack Types |
|---------|------|-----------------|------|--------------|
| **CADETS_E3** | FreeBSD | 3 (Nginx backdoor variants) | 10 GB | Backdoor, privilege escalation |
| **THEIA_E3** | Linux | 2 (Firefox Drakon, Browser Extension) | 12 GB | Browser exploit, in-memory implant |
| **CLEARSCOPE_E3** | Android | 1 (Firefox Drakon APT) | 4.8 GB | Drive-by download, C2 |
| **FIVEDIRECTIONS_E3** | Windows | 2 (Excel macro, Firefox) | 22 GB | Macro exploit, browser exploit |
| **TRACE_E3** | Linux | 3 (Firefox, phishing, Pine) | 100 GB | Browser exploit, email phishing |
| **CADETS_E5** | FreeBSD | 2 (Nginx Drakon APT) | 276 GB | Mature APT, lateral movement |
| **THEIA_E5** | Linux | 1 (Firefox Drakon) | 36 GB | BinFmt elevation, injection |
| **CLEARSCOPE_E5** | Android | 2 | 49 GB | — |
| **FIVEDIRECTIONS_E5** | Windows | 4 (CopyKatz, BITS, DNS, Drakon) | 280 GB | Credential theft, C2 channels |
| **TRACE_E5** | Linux | 1 (Firefox Drakon) | 710 GB | — |

### Secondary Datasets

| Dataset | OS | Attacks | Size | Notes |
|---------|------|---------|------|-------|
| `optc_h201`, `optc_h501`, `optc_h051` | Windows | 1 each | 6.7-9 GB | OpTC (enterprise telemetry) |
| `ATLASV2_EDR` | Windows | 10 | 1 GB | Red team engagements |
| `CARBANAKV2_EDR` | Windows+Linux | 1 | 6.6 GB | Carbanak APT |

### Node/Edge Schema (Common Across Datasets)

**Node Types** (3):
- `subject` — processes/threads
- `file` — files, pipes, sockets
- `netflow` — network connections

**Edge Types** (10):
| ID | Name | Description |
|----|------|-------------|
| 1 | `EVENT_CONNECT` | Network connect |
| 2 | `EVENT_EXECUTE` | Process execute/load |
| 3 | `EVENT_OPEN` | File open |
| 4 | `EVENT_READ` | Read operation |
| 5 | `EVENT_RECVFROM` | Receive from socket |
| 6 | `EVENT_RECVMSG` | Receive message |
| 7 | `EVENT_SENDMSG` | Send message |
| 8 | `EVENT_SENDTO` | Send to socket |
| 9 | `EVENT_WRITE` | Write operation |
| 10 | `EVENT_CLONE` | Clone/fork process |

**Valid (src_type, dst_type) → edge_type patterns** are learned by the model. Attacks violate these patterns (e.g., browser → EXECUTE → file).

---

## 5. Feature Engineering Taxonomy

Below is the proposed taxonomy with concrete feature ideas. Implemented = [x], Proposed = [ ], Considered = [?].

### 5.1 Temporal Features — When things happen

Captures timing patterns, burstiness, dormancy cycles.

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `t_norm` | [x] | `(t - t_start) / window_ns` | Position in window; attacks may cluster |
| `log_delta_prev` | [x] | `log1p((t - last_time) / 1e9)` | Global event rate; attack burst timing differs |
| `log_delta_same_pair` | [x] | `log1p((t - last_pair_time[(s,d)]) / 1e9)` | Pair-level timing; long gaps = suspicious |
| `log_delta_src_activity` | [x] | `log1p((t - last_src_time[src]) / 1e9)` | Dormancy→burst transitions; malware wakes up |
| `log_delta_dst_activity` | [ ] | `log1p((t - last_dst_time[dst]) / 1e9)` | Symmetric to src; target being accessed after dormancy |
| `log_delta_type` | [ ] | `log1p((t - last_type_time[op]) / 1e9)` | Time since this edge type last seen anywhere |
| `time_since_window_start_sec` | [ ] | `(t - t_start) / 1e9` | Raw seconds, not normalized |
| `hour_of_day` (circular) | [ ] | `sin/cos` encoding of timestamp's hour | Daily activity patterns |
| `edge_rate_rolling` | [ ] | `1 / avg(last_k_interarrivals)` | Recent event density (smoothed) |

### 5.2 Volumetric Features — How much / how many

Captures frequency, counts, diversity of interactions.

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `pair_freq_causal` | [x] | `pair_count[(s,d)] / e_idx` | Dominant pairs = normal; rare = suspicious |
| `src_unique_dsts_norm` | [x] | `len(src_unique_dsts[src]) / e_idx` | Compromised process talks to many new destinations |
| `pair_type_count_norm` | [x] | `len(pair_types[(s,d)]) / e_idx` | Shift in relationship nature (e.g., WRITE→EXECUTE) |
| `src_freq_causal` | [?] | `src_count[src] / e_idx` | How active this source is; removed in V2 (weak?) |
| `dst_freq_causal` | [ ] | `dst_count[dst] / e_idx` | Symmetric to src |
| `dst_unique_srcs_norm` | [ ] | `len(dst_unique_srcs[dst]) / e_idx` | How many sources access this destination |
| `src_out_degree` | [ ] | `src_count[src]` (raw, not normalized) | Absolute activity level |
| `src_type_count_norm` | [?] | `len(src_uniq_types[src]) / e_idx` | Edge type diversity per src; removed (compressed loss too much) |
| `type_freq_causal` | [ ] | `type_count[op] / e_idx` | How common this edge type is |
| `entropy_of_src_types` | [ ] | Shannon entropy of `src_uniq_types[src]` distribution | Behavioral unpredictability |
| `pair_interaction_ratio` | [ ] | `pair_count / (src_count + dst_count)` | How specific this pair is to each other |

### 5.3 Novelty Features — What's new / first-time

Captures first-time occurrences, rarity, behavioral shifts. *Hard binary signals avoid normalization compression issues.*

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `type_rarity` | [x] | `1 - type_count[op] / max_type_count` | Rare types = anomalous; EXECUTE from browser |
| `is_new_pair` | [x] | `(s,d) not in last_pair_time` | **Strongest signal** — never-seen interaction |
| `is_new_type_for_src` | [x] | `op not in src_uniq_types[src]` | Process does new type of syscall |
| `is_new_type_for_dst` | [?] | `op not in dst_uniq_types[dst]` | Symmetric to src; tried in uncommitted version |
| `is_new_src_for_dst` | [ ] | `src not in dst_uniq_srcs[dst]` | First time this source accesses dst |
| `is_new_type_for_pair` | [ ] | `op not in pair_types[(s,d)]` | First time this pair uses this edge type |
| `pair_never_seen_outside_window` | [ ] | Cross-window state (if we implement it) | Truly novel across longer time horizon |
| `type_absolute_rarity` | [ ] | `log(1 + max_type_count / max(type_count[op], 1))` | Log ratio instead of 1-freq |
| `src_becoming_promiscuous` | [ ] | `Δ(src_unique_dsts)` over last k edges | Sharp increase in destinations |

### 5.4 Structural/Relational Features — How entities connect

Captures graph structure, influence, dominance. *These are computed causally from edge stream, not global graph.*

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `pair_dominance` | [ ] | `pair_count[(s,d)] / max(src_count[src], 1)` | What fraction of src's activity goes to this dst |
| `dst_exposure` | [ ] | `pair_count[(s,d)] / max(dst_count[dst], 1)` | What fraction of dst's activity comes from this src |
| `src_specialization` | [ ] | `max(src_pair_freqs) / sum(src_pair_freqs)` | Whether src talks to many or few dsts |
| `degree_ratio_src` | [ ] | `in_degree[src] / max(out_degree[src], 1)` | Receive/send ratio; processes vs files behave differently |
| `betweenness_approx` | [ ] | Approximated from stream | Structural importance (but expensive) |
| `clustering_coeff_approx` | [ ] | Local triangle density approximation | Community structure |
| `is_bridge_candidate` | [ ] | Connecting two otherwise disconnected components | Malware connecting two zones |

### 5.5 Sequential/Contextual Features — What happened nearby

Captures transitions, context windows, operation sequences.

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `prev_op_norm` | [?] | `rel2id[prev_op] / max_rel_id` | Previous edge type; was in V2, replaced by novelty (weak?) |
| `prev_prev_op` | [ ] | Two edges back | Longer context |
| `same_as_prev` | [ ] | `op == prev_op` binary | Bursts of same operation |
| `transition_validity` | [ ] | Is `(prev_type, this_type)` a common sequence? | Unusual transitions = anomalous |
| `last_k_types_histogram` | [ ] | Distribution of last 5-10 edge types | Recent behavioral context |
| `op_sequence_prob` | [ ] | Learned Markov model probability of this sequence | Low-prob sequences = anomalous |

### 5.6 Causal Features — Direct influence relationships

Captures "happened-before" relationships, potential causal chains.

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `is_causal_successor` | [ ] | Same src as previous edge; short delta | Process doing consecutive things |
| `causal_chain_length` | [ ] | How many consecutive edges from same src with small delta | Long chains = program execution loops vs attack |
| `dst_became_src` | [ ] | This edge's src was dst in recent edge | Information flow: file read → process execute |
| `write_read_gap` | [ ] | Time between WRITE to dst and READ from same dst by same src | Suspicious if very short (write-then-read pattern) |

### 5.7 Derived/Composite Features — Combinations

| Feature | Status | Formula/Description | Rationale |
|---------|--------|---------------------|-----------|
| `novelty_timing_interaction` | [ ] | `is_new_pair * log_delta_prev` | New pair after long idle = very suspicious |
| `activity_spread` | [ ] | `src_unique_dsts * log_delta_src_activity` | Active source with many dsts after dormancy |
| `rare_type_new_pair` | [ ] | `type_rarity * is_new_pair` | Both signals = very high anomaly |

---

## 6. Experimental Plan

### 6.1 Principles

1. **One variable at a time**: Change only one feature or parameter per experiment
2. **Control baseline**: Always compare against `velox` (vanilla, no edge vector)
3. **Multiple seeds**: PIDS are unstable; run each config ≥3 times with different seeds
4. **Primary datasets**: Start with `CLEARSCOPE_E3` and `CADETS_E3` (well-understood baselines)
5. **Metrics to track**:
   - `ADP` — Average Detection Precision (ranking quality)
   - `PRE` — Precision (TP / (TP + FP))
   - `TP` — True Positives (actual attacks found)
   - `FP` — False Positives (alarms on normal behavior)
   - `F1` — F1 score
   - `val_loss_distribution` — Min/mean/max/std of validation loss (diagnostic)

### 6.2 Immediate Plan (Next Experiments)

#### Phase 0: Re-establish Baselines (Current)

**Goal**: Get clean, reproducible baselines for both vanilla and current edge vector

| # | Exp ID | System | Dataset | Description | Status |
|---|--------|--------|---------|-------------|--------|
| 0.1 | `baseline_velox_cs` | velox | CLEARSCOPE_E3 | Vanilla, no edge vector | Planned |
| 0.2 | `baseline_velox_ca` | velox | CADETS_E3 | Vanilla, CADETS is sensitive | Planned |
| 0.3 | `baseline_ve_current_cs` | velox-edge | CLEARSCOPE_E3 | Current 10-feature config | Planned |
| 0.4 | `baseline_ve_current_ca` | velox-edge | CADETS_E3 | Current 10-feature on CADETS | Planned |

#### Phase 1: Understand the Compression Problem

**Goal**: Diagnose *why* edge features hurt CADETS TP

Hypotheses to test:
- H1: Normal-edge loss variance is lower with edge features
- H2: Attack-edge loss is actually *lower* (not just threshold dropping)
- H3: Edge features help CLEARSCOPE because it has more stereotyped patterns

| # | Exp ID | Change | Hypothesis Test | Status |
|---|--------|--------|-----------------|--------|
| 1.1 | `ve_v2_repro_cs` | Restore V2: 8 features (no novelty), proj=32 | V2 was best for CS; does it still work? | Planned |
| 1.2 | `ve_v2_repro_ca` | Same V2 config on CADETS | V2 CS=good, CA=bad; confirm this pattern | Planned |
| 1.3 | `ve_proj_16` | `edge_vector_proj_dim: 16` instead of 32 | Lower capacity = less overfitting = less compression? | Planned |
| 1.4 | `ve_proj_8` | `edge_vector_proj_dim: 8` | Even lower capacity | Planned |
| 1.5 | `ve_no_proj` | Remove projection; direct concat | Does forcing lower-dim representation help or hurt? | Planned |

#### Phase 2: Feature Ablation — Which Features Help/Hurt?

**Goal**: Identify individual feature contribution

Start with V2's 8 features, then add/remove systematically:

| # | Exp ID | Feature Set | CS vs CA | Expected Effect | Status |
|---|--------|-------------|----------|-----------------|--------|
| 2.1 | `ve_abl_temporal_only` | Only t_norm, log_delta_prev, log_delta_same_pair, log_delta_src_activity (4) | Both | Pure timing, no normalization | Planned |
| 2.2 | `ve_abl_volumetric_only` | Only pair_freq_causal, src_unique_dsts, pair_type_count (3) | Both | Normalized counts — these may cause compression | Planned |
| 2.3 | `ve_abl_novelty_binary_only` | Only is_new_pair, is_new_type_for_src (2) | Both | Hard binary — no normalization! | Planned |
| 2.4 | `ve_abl_novelty_all` | type_rarity + 2 binary (3) | Both | type_rarity uses normalization | Planned |
| 2.5 | `ve_abl_no_is_new_pair` | Remove is_new_pair from current 10 | CS | is_new_pair is strongest novelty signal; but does it compress most? | Planned |
| 2.6 | `ve_abl_only_is_new_pair` | Only is_new_pair (1) + type vector | Both | Minimal edge vector; test compression hypothesis | Planned |

#### Phase 3: Mitigate Compression

**Goal**: Keep useful signal while widening normal-edge loss distribution

| # | Exp ID | Idea | How It Works | Status |
|---|--------|------|--------------|--------|
| 3.1 | `ve_feature_noise` | Add Gaussian noise during training | `feats += N(0, 0.05)` — widens loss by making features less reliable | Planned |
| 3.2 | `ve_feature_dropout` | Randomly zero features with p=0.1 | Forces model to not rely too much on any single feature | Planned |
| 3.3 | `ve_temp_scaling` | Temperature > 1.0 in softmax | `logits / T` where T > 1 — softens distribution, widens loss range | Planned |
| 3.4 | `ve_weight_decay_edge` | L2 regularization on lin_edge only | Prevents over-learning edge feature patterns | Planned |
| 3.5 | `ve_dynamic_threshold` | Use different threshold method | Instead of max_val_loss, try percentile or other methods | Planned |

#### Phase 4: New Features

**Goal**: Test the newly implemented features from ground truth analysis

All features are now implemented in `build_default_graphs.py`. Config files for each variation created. Run with `--force_restart construction` to recompute graphs.

| # | Exp ID | Config File | Features | Status |
|---|--------|-------------|----------|--------|
| 4.1 | `ve_wte_isolated_ca/cs` | `velox-edge-wte.yml` | is_new_pair + write_then_execute_flag | **Ready** |
| 4.2 | `ve_binary_only_ca/cs` | `velox-edge-binary-only.yml` | 7 binary features only | **Ready** |
| 4.3 | `ve_cadets_targeted_ca/cs` | `velox-edge-cadets-targeted.yml` | 7 CADETS-specific features | **Ready** |
| 4.4 | `ve_clearscope_targeted_ca/cs` | `velox-edge-clearscope-targeted.yml` | 7 CLEARSCOPE-specific features | **Ready** |
| 4.5 | `ve_v2_plus_binary_ca/cs` | `velox-edge-v2-plus-binary.yml` | V2 core + 4 binary extensions | **Ready** |

### 6.3 Longer-term Plan

After understanding CLEARSCOPE_E3 and CADETS_E3:

1. **Test on THEIA_E3** — different attack profile
2. **Test on E5 datasets** — more mature APTs
3. **Test on OpTC** — different data source (enterprise)
4. **Per-dataset feature selection** — different datasets may benefit from different features
5. **Feature importance analysis** — attention weights, SHAP, or ablation-based

---

## 7. Progress Tracking

### 7.1 Feature Decision Log

Every feature addition, test, or discard logged here.

| Date | Feature ID | Action | Dataset | ADP | PRE | TP | FP | Decision | Reasoning |
|------|------------|--------|---------|-----|-----|----|-----|----------|-----------|
| 2026-05-12 | (baseline) | Reference | CS_E3 | 0.167 | 0.00150 | 1 | 666 | Baseline | Vanilla Velox (no edge vector) |
| 2026-05-12 | (baseline) | Reference | CA_E3 | 0.343 | 1.000 | 3 | 0 | Baseline | Vanilla Velox — CADETS sensitive |
| 2026-05-12 | `t_norm` | Implemented | — | — | — | — | — | Keep (tentative) | In V2 and current; need ablations |
| 2026-05-12 | `log_delta_prev` | Implemented | — | — | — | — | — | Keep (tentative) | In V2 (best CS results) |
| 2026-05-12 | `log_delta_same_pair` | Implemented | — | — | — | — | — | Keep (tentative) | In V2 — strong timing signal |
| 2026-05-12 | `log_delta_src_activity` | Implemented | — | — | — | — | — | Keep (tentative) | Dormancy→burst detection |
| 2026-05-12 | `pair_freq_causal` | Implemented | — | — | — | — | — | Keep (tentative) | In V2 — may cause compression |
| 2026-05-12 | `src_unique_dsts_norm` | Implemented | — | — | — | — | — | Evaluate | Uses normalization; suspect for compression |
| 2026-05-12 | `pair_type_count_norm` | Implemented | — | — | — | — | — | Evaluate | Uses normalization |
| 2026-05-12 | `type_rarity` | Implemented | — | — | — | — | — | Evaluate | Uses max normalization; helps ranking but may compress |
| 2026-05-12 | `is_new_pair` | Implemented | — | — | — | — | — | **High priority** | Strongest novelty signal; binary = no normalization |
| 2026-05-12 | `is_new_type_for_src` | Implemented | — | — | — | — | — | Keep (tentative) | Binary novelty |
| 2026-05-12 | `is_new_type_for_pair` | Implemented | — | — | — | — | — | **Pending eval** | Binary: first time this (src,dst) uses this op type. CADETS key: nginx→CLONE never done before |
| 2026-05-12 | `is_new_src_for_dst` | Implemented | — | — | — | — | — | **Pending eval** | Binary: first time src accesses dst. Symmetric to is_new_pair from dst perspective |
| 2026-05-12 | `is_new_path_global` | Implemented | — | — | — | — | — | **Pending eval** | Binary: dst path never seen in window by ANY src. Stronger than is_new_pair; /tmp/vUgefal is globally new |
| 2026-05-12 | `is_tmp_dst` | Implemented | — | — | — | — | — | **Pending eval** | Binary path feature. Direct CADETS signal: /tmp writes = malware staging on FreeBSD server |
| 2026-05-12 | `is_hash_filename_dst` | Implemented | — | — | — | — | — | **Pending eval** | Binary path feature. CLEARSCOPE: 39 of 41 anomalous nodes have 40-char SHA-1 hex names |
| 2026-05-12 | `is_server_port_outbound` | Implemented | — | — | — | — | — | **Pending eval** | Binary: dst=netflow AND local_port≤1024. CADETS: nginx:80 initiates outbound C2 |
| 2026-05-12 | `is_execute_from_tmp` | Implemented | — | — | — | — | — | **Pending eval** | Binary: EVENT_EXECUTE from /tmp. Subset of is_tmp_dst, but more specific |
| 2026-05-12 | `write_then_execute_flag` | Implemented | — | — | — | — | — | **Pending eval** | Binary causal: EVENT_EXECUTE on dst written within 300s. Canonical APT staging pattern |
| 2026-05-12 | `log_delta_dst_activity` | Implemented | — | — | — | — | — | **Pending eval** | Continuous: symmetric to log_delta_src_activity for dst side |
| 2026-05-12 | `src_burst_new_pairs` | Implemented | — | — | — | — | — | **Pending eval** | Continuous: rolling density of is_new_pair in last 50 edges from src. CLEARSCOPE: Firefox writes 39 new files in burst |
| 2026-05-12 | `log_delta_prev2` | Removed | — | — | — | — | — | **Discard** | Redundant with log_delta_prev; removed from uncommitted version |
| 2026-05-12 | `src_freq_causal` | Removed | — | — | — | — | — | **Discard** | Weak signal; removed in V2 pruning |
| 2026-05-12 | `src_type_diversity` | Removed | — | — | — | — | — | **Discard** | Normalizes too much, compresses loss; removed in uncommitted version |
| 2026-05-12 | `prev_op_norm` | Replaced | — | — | — | — | — | **Discard** | Weak signal; replaced by novelty in V5 |

### 7.2 Experiment Log

Full experiment history. Use `Exp ID` to reference in W&B or notes.

| Date | Exp ID | Config | Dataset | Seed | ADP | PRE | TP | FP | Notes |
|------|--------|--------|---------|------|-----|-----|----|-----|-------|
| (historic) | — | Vanilla Velox | CS_E3 | — | 0.167 | 0.00150 | 1 | 666 | Baseline (7fc9e29) |
| (historic) | — | Vanilla Velox | CA_E3 | — | 0.343 | 1.000 | 3 | 0 | Baseline (7fc9e29) |
| (historic) | — | V1: 268-dim, no proj | CS_E3 | — | — | — | — | — | Redundant embeddings (2a58136) — abandoned |
| (historic) | — | V2: 14-dim, proj=32, 8 features | CS_E3 | — | 0.120 | 0.02326 | 1 | 42 | **Best CS result**: 15× PRE, 94% fewer FP (c80928e) |
| (historic) | — | V2: same | CA_E3 | — | — | — | 0 | — | **CADETS broken**: TP=0 |
| (historic) | — | V5: 10 features, proj=64, novelty | CS_E3 | — | 0.333 | — | 0 | — | Best ADP ranking, but TP=0 (threshold too strict) |
| (historic) | — | V5: same | CA_E3 | — | 0.637 | — | 0 | — | Best ADP, still TP=0 |
| (historic) | — | Uncommitted: 8 features + parallel decoder | CS_E3 | — | 0.125 | 0.00007 | 1 | 13649 | **Worst ever**: parallel decoder adds noise |

### 7.3 Key Insights to Date

| # | Insight | Evidence | Implication |
|---|---------|----------|-------------|
| 1 | **V2 is sweet spot for CS** | V2: PRE=0.02326 (15× vanilla), FP=42 (94% reduction) | Start by reproducing and understanding V2 |
| 2 | **ALL edge variants break CADETS** | TP=0 across V1, V2, V5, current | Core compression problem needs mitigation |
| 3 | **Novelty improves ranking but not detection** | V5: ADP=0.333 (best) but TP=0 | Novelty compresses threshold further |
| 4 | **Parallel decoder = regression** | FP exploded to 13K+ | Two heads compete for shared weights; node head adds noise |
| 5 | **Normalization = suspect** | Features using `/ e_idx` normalization removed in pruning | Continuous [0,1] may compress; binary {0,1} avoids this |
| 6 | **`is_new_pair` = strongest signal** | High impact in V5; binary = no normalization | Test in isolation; understand its compression effects |

---

## 8. Code Reference

### Key Files for Feature Engineering

| Purpose | File | Lines |
|---------|------|-------|
| Helper functions | `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py` | 180-225 |
| Feature computation loop | `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py` | 425-524 |
| Feature config schema | `pidsmaker/config/config.py` | 815-819 |
| Edge vector assembly | `pidsmaker/utils/data_utils.py` | 290-299 |
| Edge vector decoder | `pidsmaker/decoders/custom_edge_mlp_decoder.py` | 7-29 |
| Feature dim derivation | `pidsmaker/factory.py` | 760-786 |
| Zero-fill for missing feats | `pidsmaker/tasks/feat_inference.py` | 26-32 |
| Model forward pass | `pidsmaker/model.py` | 150 |
| Base config | `config/velox-edge.yml` | All |
| Variation 1 (binary-only) | `config/velox-edge-binary-only.yml` | All |
| Variation 2 (CADETS) | `config/velox-edge-cadets-targeted.yml` | All |
| Variation 3 (CLEARSCOPE) | `config/velox-edge-clearscope-targeted.yml` | All |
| Variation 4 (V2+binary) | `config/velox-edge-v2-plus-binary.yml` | All |
| Variation 5 (WTE only) | `config/velox-edge-wte.yml` | All |

### How to Add a New Feature

1. **Add state variable** (if needed) before the inner loop in `build_default_graphs.py`
2. **Add computation** to `all_feat_vals` dict (inside the loop)
3. **Add state update** at the bottom of the loop (after `e["temporal_feats"] = feats`)
4. **Add to a config file**: include the name in `feature_names` list; `num_features` auto-derives
5. **Document** in this file's Feature Decision Log
6. **Run experiment** with `--force_restart construction` and log results

### Run Commands

```bash
# CLEARSCOPE_E3 (quick for development)
./run.sh velox-edge CLEARSCOPE_E3 \
  --training.encoder.dropout=0.3 \
  --training.lr=0.001 \
  --training.node_hid_dim=64 \
  --training.node_out_dim=64 \
  --training.num_epochs=12 \
  --featurization.emb_dim=128 \
  --construction.time_window_size=15.0 \
  --project=PIDSHyp \
  --exp=<YOUR_EXP_ID> \
  --force_restart construction

# CADETS_E3 (sensitive, for validation)
./run.sh velox-edge CADETS_E3 \
  --training.encoder.dropout=0.3 \
  --training.lr=0.0001 \
  --training.node_hid_dim=256 \
  --training.node_out_dim=256 \
  --training.num_epochs=12 \
  --featurization.emb_dim=256 \
  --project=PIDSHyp \
  --exp=<YOUR_EXP_ID>
```

---

## 9. Related Documents

- `progress.md` — Detailed experiment log with V1/V2/V5 analysis
- `features_math.md` — Mathematical description of the 10 current features
- `notes/versions.md` — Per-commit summary of changes
- `notes/commands.md` — Example run commands
- `config/velox-edge.yml` — Current configuration
- `Ground_Truth/orthrus/readme.md` — Dataset attack details

---

> **Keep this file updated**: Every feature addition, removal, or experimental result should be logged here. This is our single source of truth for the feature engineering journey.
