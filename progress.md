# PIDSMaker — Velox Edge Variants Progress

## Goal

Outperform vanilla Velox by engineering edge-level features and a decoder
architecture that preserves the node-feature loss signal while adding weighted
temporal-novelty features.

## Constraints & Preferences

- All runs use the same base hyperparams:
  `dropout=0.3`, `lr=0.0001`, `node_hid_dim=128`, `node_out_dim=128`,
  `emb_dim=128`, `epochs=12`.
- Feature sets are versioned in git as commits `velox_edge_*`.
- Must not snoop the edge type label (edge vector excludes the label).
- Edge vector = `concat(src_type_onehot, dst_type_onehot, temporal_feats)` —
  no redundant node embeddings.

---

## Architecture Backbone (inherited from `orthrus.yml → orthrus_non_snooped.yml → velox.yml`)

```yaml
construction:
  time_window_size: 15.0
  fuse_edge: True

featurization:
  emb_dim: 128
  used_method: word2vec

batching:
  intra_graph_batching:
    used_methods: edges, tgn_last_neighbor
    edges:
      intra_graph_batch_size: 1024
    tgn_last_neighbor:
      tgn_neighbor_size: 20

training:
  lr: 0.0001
  num_epochs: 12
  patience: 3
  node_hid_dim: 128
  node_out_dim: 128
  encoder:
    dropout: 0.3
    used_methods: none          # just Linear(?, 128)
    x_is_tuple: True
  decoder:
    used_methods: predict_edge_type
    predict_edge_type:
      decoder: edge_mlp
      edge_mlp:
        architecture_str: linear(0.5) | relu
        src_dst_projection_coef: 2

evaluation:
  node_evaluation:
    threshold_method: max_val_loss
    use_dst_node_loss: True
    use_kmeans: False
    best_model_selection: best_adp
```

### How the decoder MLP works (base)

```python
lin_src = Linear(node_out_dim=128, 128 * src_dst_projection_coef=2)  # 128 → 256
lin_dst = Linear(128, 256)                                           # 128 → 256

concat = [lin_src(h_src), lin_dst(h_dst)]           # 256 + 256 = 512

# If edge_vector present:
#   lin_edge = Linear(input_dim, proj_dim)
#   concat += [lin_edge(edge_vector)]                # 512 + proj_dim

MLP: Linear(512+proj_dim, (512+proj_dim)*0.5) → ReLU → Linear(..., 10)
```

### How anomaly detection works

Inference mode: the `cross_entropy` loss function returns **per-edge scores**
(the negative log-probability of the predicted edge type). Higher score =
more surprising = more anomalous.

The threshold is `max_val_loss` — the maximum per-edge score on the validation
set. Any test edge with score > this threshold is flagged as anomalous.

**Core tension**: features that help the model predict edge types too well
compress the loss distribution → `max_val_loss` drops → fewer test edges
clear the threshold → TP goes to 0.

---

## Vanilla Velox (`7fc9e29`) — No Edge Vector

### Architecture

`EdgeTypePrediction` receives only `h_src` and `h_dst` (128-dim node
embeddings each). No edge vector. Decoder MLP input = `concat(lin_src,
lin_dst)` = 512 dims.

### Results

| Dataset | ADP | F1 | PRE | TP | FP |
|---------|-----|----|-----|----|----|
| CS_E3   | 0.167 | 0.00282 | 0.00150 | 1 | 666 |
| CA_E3   | 0.343 | 0.08451 | 1.00000 | 3 | 0 |

---

## V1: First Edge Vector (`2a58136` — "version 1 velox-edge")

### Config

```yaml
num_features: 6
edge_features: edge_type,edge_vector
# NO edge_vector_proj_dim
```

### The 6 Temporal Features (NOT causal, pre-computed from window-wide stats)

```python
src_counts = Counter(all src in window)   # global, before the loop
dst_counts = Counter(all dst in window)
total = len(edge_list)

for e in edge_list:
    t = e["time"]
    feats = [
        (t - start_time) / window_size_ns,                # 0: t_norm
        min((t - last_src_time[src]) / 1e9, 1.0),         # 1: delta_src, capped
        min((t - last_dst_time[dst]) / 1e9, 1.0),         # 2: delta_dst, capped
        src_counts[src] / total,                           # 3: src_frac
        dst_counts[dst] / total,                           # 4: dst_frac
        math.log10(total + 1) / 6.0,                       # 5: window_density
    ]
```

### Edge Vector Assembly

```python
edge_vector = concat(
    src_type_onehot(3),
    dst_type_onehot(3),
    src_emb(128),      # REDUNDANT: model already sees x_src
    dst_emb(128),      # REDUNDANT: model already sees x_dst
    temporal_feats(6),
)   # total = 268 dims
```

### Decoder

```python
lin_edge = Linear(268, 268)   # identity — NO compression

concat = lin_src(256) + lin_dst(256) + lin_edge(268) = 780

MLP: Linear(780, 390) → ReLU → Linear(390, 10)
```

### Problems

- 96% of the 268 edge dims are redundant embeddings (src_emb+dst_emb=256).
- `lin_edge` is identity → temporal features are only 6 / 780 = **0.8%** of
  MLP input → the MLP can easily ignore them.
- Features 1, 2 clipped at 1s → binary for sparse nodes.
- Feature 5 is a window-global constant → zero per-edge discriminative power.

---

## V2: Causal Features + Projection (`c80928e` — "good for cs3, a bit for ca3")

**This was the best CLEARSCOPE variant.**

### Config

```yaml
num_features: 8
edge_vector_proj_dim: 32
```

### The 8 Temporal Features (CAUSAL — sorted by time, computed incrementally)

```python
edge_list.sort(key=lambda e: e["time"])
last_time = None
second_last_time = None
last_pair_time = {}
pair_count = Counter()
src_count = Counter()
src_uniq_types = defaultdict(set)
prev_op = None

for e_idx, e in enumerate(edge_list):
    t = e["time"]
    src, dst = e["src"], e["dst"]
    op = e["label"]
    normalizer = max(e_idx, 1)

    feats = [
        (t - start_time) / window_size_ns,                              # 0: t_norm
        log1p((t - last_time) / 1e9) if last_time else 0.0,             # 1: log_delta_prev
        log1p((t - second_last_time) / 1e9) if second_last_time else 0.0, # 2: log_delta_prev2
        log1p((t - last_pair_time.get((src,dst), t)) / 1e9),            # 3: log_delta_same_pair
        pair_count[(src, dst)] / normalizer,                             # 4: pair_freq_causal
        src_count[src] / normalizer,                                     # 5: src_freq_causal
        len(src_uniq_types[src]) / normalizer,                           # 6: src_type_diversity
        rel2id.get(prev_op, 0) / max_rel_id if prev_op else 0.0,        # 7: prev_op_norm
    ]
```

| # | Name | Math | What it captures |
|---|------|------|------------------|
| 0 | t_norm | `(t - t_start) / window_ns` | Temporal position in window |
| 1 | log_delta_prev | `log1p((t - last_time) / 1e9)` | Log-seconds since any previous edge |
| 2 | log_delta_prev2 | `log1p((t - second_last_time) / 1e9)` | Log-seconds since second-previous edge (redundant) |
| 3 | log_delta_same_pair | `log1p((t - last_pair_time[(s,d)]) / 1e9)` | Log-seconds since this pair last interacted |
| 4 | pair_freq_causal | `pair_count[(s,d)] / e_idx` | Causal frequency of this pair |
| 5 | src_freq_causal | `src_count[src] / e_idx` | Causal frequency of this src |
| 6 | src_type_diversity | `len(src_uniq_types[src]) / e_idx` | Causal type diversity of this src |
| 7 | prev_op_norm | `rel2id[prev_op] / max_rel_id` | Previous edge type [0,1] |

### Edge Vector Assembly

```python
edge_vector = concat(
    src_type_onehot(3),
    dst_type_onehot(3),
    temporal_feats(8),
)   # total = 14 dims
```

### Decoder

```python
lin_edge = Linear(14, 32)   # projection!

concat = lin_src(256) + lin_dst(256) + lin_edge(32) = 544

MLP: Linear(544, 272) → ReLU → Linear(272, 10)
```

### Key changes from V1

| Change | Effect |
|--------|--------|
| **Removed** src_emb + dst_emb from edge vector | 268 → 14 dims. Temporal features go from 0.8% → 57% of edge vector signal |
| **Added** `lin_edge: Linear(14 → 32)` | Compresses and learns a latent representation of the edge vector |
| **Causal** computation (sorted, incremental state) | Each feature uses only info from **before** the current edge. `pair_count / e_idx` = genuine novelty (new pair → high loss) |
| **Log transforms** instead of caps | Preserves resolution for both small and large deltas |
| **Per-edge normalizer** `e_idx` instead of global `total` | Reflects causal density at this point in time |

### Why V2 worked on CLEARSCOPE

Temporal features (especially `log_delta_same_pair`, `pair_freq_causal`,
`log_delta_prev`) are highly predictive of normal edge types in CLEARSCOPE's
stereotyped system call patterns. Normal edges get low loss; attack edges
deviate → high loss → good detection. The 14→32 projection prevents the MLP
from overfitting while still giving useful signal.

### Why V2 failed on CADETS

The same mechanism hurts: temporal features predict CADETS normal edges too
well → loss distribution compresses → `max_val_loss` threshold drops to near
zero → attack edges barely clear it → TP=0.

### Results

| Dataset | ADP | PRE | FP | TP |
|---------|-----|-----|----|----|
| CS_E3   | 0.120 | **0.02326** (15× vs vanilla) | **42** (94% fewer vs vanilla) | 1 |
| CA_E3   | — | — | — | **0** |

---

## V5: Novelty Signals + Higher Capacity (`42857f0` — "version 5 velox edge")

### Config

```yaml
num_features: 10
edge_vector_proj_dim: 64
```

### The 10 Temporal Features (V2's 7 kept + 3 novelty replacing prev_op_norm)

```python
type_count = Counter()
max_type_count = 0

for e_idx, e in enumerate(edge_list):
    op_id = rel2id.get(op, 0)
    max_type_count = max(max_type_count, max(type_count.values(), default=0))
    type_rarity = 1 - (type_count.get(op_id, 0) / max(max_type_count, 1))

    feats = [
        (t - start_time) / window_size_ns,                         # 0: t_norm
        log1p((t - last_time) / 1e9) if last_time else 0.0,        # 1: log_delta_prev
        log1p((t - second_last_time) / 1e9) if ... else 0.0,       # 2: log_delta_prev2
        log1p((t - last_pair_time[(s,d)]) / 1e9),                   # 3: log_delta_same_pair
        pair_count[(s,d)] / normalizer,                              # 4: pair_freq_causal
        src_count[src] / normalizer,                                 # 5: src_freq_causal
        len(src_uniq_types[src]) / normalizer,                       # 6: src_type_diversity
        type_rarity,                                                 # 7: type_rarity (NEW)
        0.0 if (src,dst) in last_pair_time else 1.0,                 # 8: is_new_pair (NEW)
        0.0 if op_id in src_uniq_types[src] else 1.0,                # 9: is_new_type_for_src (NEW)
    ]
```

| # | Name | Math | What it captures |
|---|------|------|------------------|
| 7 | type_rarity | `1 - type_count[op] / max_type_count` | Rarity of this edge type in the window. Rare types get high values |
| 8 | is_new_pair | `1.0` if never seen (src,dst), else `0.0` | Brand-new interaction |
| 9 | is_new_type_for_src | `1.0` if op_id unseen for this src, else `0.0` | Novel edge type for this source |

### Key changes from V2

| Change | Effect |
|--------|--------|
| **Replaced** `prev_op_norm` with `type_rarity` | `prev_op_norm` was weak (just last type). `type_rarity` captures genuine rarity — edge types seen fewer times → higher value → possible attack |
| **Added** `is_new_pair`, `is_new_type_for_src` | Binary novelty. Attacks inherently do novel things → strongly flagged |
| **Hard binary** 0/1 instead of continuous | Preserves the novelty signal without normalization compression |
| `edge_vector_proj_dim: 32 → 64` | More capacity for the 10-dim temporal space, but risks overfitting |

### Results

| Dataset | ADP | TP | Notes |
|---------|-----|----|-------|
| CS_E3   | **0.333** (best rank) | **0** | Threshold too strict — novelty compressed normal-edge loss further |
| CA_E3   | up to **0.637** | **0** | Good ranking but zero detections |

---

## Current Working State (uncommitted): Pruned Features + Parallel Decoder

### Config

```yaml
num_features: 8
edge_vector_proj_dim: 32

# Parallel decoder
training:
  decoder:
    predict_edge_type:
      edge_loss_lambda: 0.3
```

### Feature pruning from V5

```
Removed:  log_delta_prev2 (redundant with log_delta_prev)
Removed:  src_freq_causal   (weak — doesn't discriminate)
Removed:  src_type_diversity (normalizes too much, kills detection)
Added:    is_new_type_for_dst (symmetric with is_new_type_for_src)
```

### The 8 Features

```python
feats = [
    (t - start_time) / window_size_ns,                         # 0: t_norm
    log1p((t - last_time) / 1e9) if last_time else 0.0,        # 1: log_delta_prev
    log1p((t - last_pair_time[(s,d)]) / 1e9),                   # 2: log_delta_same_pair
    pair_count[(s,d)] / normalizer,                              # 3: pair_freq_causal
    type_rarity,                                                 # 4: type_rarity
    0.0 if (src,dst) in last_pair_time else 1.0,                 # 5: is_new_pair
    0.0 if op_id in src_uniq_types[src] else 1.0,                # 6: is_new_type_for_src
    0.0 if op_id in dst_uniq_types[dst] else 1.0,                # 7: is_new_type_for_dst (NEW)
]
```

### Parallel Decoder Architecture

```python
# CustomEdgeMLP.forward returns (logits_node, logits_edge)
#   logits_node  = MLP_node(concat(lin_src, lin_dst))           # 512 → 256 → 10
#   logits_edge  = MLP(concat(lin_src, lin_dst, lin_edge))      # 544 → 272 → 10

# EdgeTypePrediction
#   loss = loss_fn(logits_node, target) + edge_loss_lambda * loss_fn(logits_edge, target)
```

### Results (CS_E3)

| ADP | F1 | PRE | TP | FP |
|-----|----|-----|----|----|
| 0.125 | 0.00015 | **0.00007** | 1 | **13649** |

Worst version. The node-only head produces noisy high losses that dominate the
combined signal, blowing up false positives.

---

## Summary: Key Patterns Across Versions

| Version | CS_E3 PRE | CS_E3 FP | CA_E3 TP | What changed |
|---------|-----------|----------|----------|-------------|
| **Vanilla** | 0.00150 | 666 | **3** | No edge vector |
| V1 | ? | ? | ? | 268-dim edge vector, no projection, redundant embeddings |
| **V2** | **0.02326** | **42** | 0 | 14-dim causal features + 32-dim projection |
| V5 | ? (TP=0) | ? | 0 | 10 features, novelty signals, 64-dim projection |
| **Current** | 0.00007 | 13649 | ? | 8 features, parallel decoder |

### Key Insights

1. **V2 hits the sweet spot for CLEARSCOPE**: causal temporal features +
   projection + no redundant embeddings. The 15× precision gain over vanilla
   with 94% fewer FPs is real.

2. **ALL edge-vector variants kill CADETS (TP=0)**: The temporal features
   compress the loss distribution below detectability. CADETS has fewer edges,
   less data → threshold calibration is tighter.

3. **Novelty signals improve ADP ranking but not detection**: V5's `is_new_pair`
   and `type_rarity` ranked attacks higher (ADP 0.333) but pushed the threshold
   too low to make a detection.

4. **Parallel decoder was a regression**: The node-only head adds noise, not
   signal. The two heads compete for the shared `lin_src`/`lin_dst` weights.

### Core Tension

```
Better edge features → lower loss on normal edges → threshold drops → TP↓
```

The problem is *not* that edge features are uninformative — it's that they are
**too informative** and compress the loss distribution. Any fix needs to widen
the loss distribution for normal edges while keeping attack edge loss high.

---

## CLEARSCOPE E3 Domain Analysis

### Edge Types (DARPA TC)

| ID | Name | Description |
|----|------|-------------|
| 1 | EVENT_CONNECT | Network connect |
| 2 | EVENT_EXECUTE | Process execute/load |
| 3 | EVENT_OPEN | File open |
| 4 | EVENT_READ | Read operation |
| 5 | EVENT_RECVFROM | Receive from socket |
| 6 | EVENT_RECVMSG | Receive message |
| 7 | EVENT_SENDMSG | Send message |
| 8 | EVENT_SENDTO | Send to socket |
| 9 | EVENT_WRITE | Write operation |
| 10 | EVENT_CLONE | Clone/fork process |

### Valid (src_type, dst_type) → edge type combinations

| Src→Dst | Allowed edge types |
|---------|-------------------|
| subject→subject | READ, WRITE, OPEN, CONNECT, RECVFROM, SENDTO, CLONE, SENDMSG, RECVMSG (9 types) |
| subject→file | WRITE, CONNECT, SENDMSG, SENDTO, CLONE (5 types) |
| subject→netflow | WRITE, SENDTO, CONNECT, SENDMSG (4 types) |
| file→subject | READ, OPEN, RECVFROM, EXECUTE, RECVMSG (5 types) |
| netflow→subject | OPEN, READ, RECVFROM, RECVMSG (4 types) |

### Attack profile (Firefox Drakon APT)

The browser (subject) is exploited via malicious website → Drakon implant
downloaded → C2 established. Key behavioral changes in the browser process:
- Subject→File: WRITE (writing malware binary to disk)
- File→Subject: EXECUTE (launching the malware)
- Subject→Netflow: CONNECT/SENDTO/SENDMSG (C2 communication)
- Subject→Subject: CLONE (forking)

### What makes a good feature for this attack

A feature is useful if:
1. For **normal** edges: feature correlates with edge type (helps predict it → low loss)
2. For **attack** edges: feature does NOT match the learned pattern (→ high loss)
3. The feature does NOT collapse normal-edge loss variance (so threshold stays meaningful)

---

## Feature Design for CLEARSCOPE E3

### Feature Categories

**Category A: Temporal** (when things happen)
- `t_norm`: temporal position in window
- `log_delta_prev`: time since any previous edge (global event rate)
- `log_delta_same_pair`: time since this pair last interacted
- `log_delta_src_activity`: time since this src last did anything

**Category B: Volumic** (how much / how many)
- `pair_freq_causal`: edges between this (src,dst) so far
- `src_freq_causal`: total edges from this src so far
- `src_unique_dsts / normalizer`: how many unique destinations src talks to
- `pair_type_count / normalizer`: how many different edge types between this pair

**Category C: Novelty** (what's new)
- `type_rarity`: how rare this edge type is in the window
- `is_new_pair`: binary, first time seeing (src,dst)
- `is_new_type_for_src`: binary, first time using this edge type for src
- `is_new_type_for_dst`: binary, first time seeing this edge type for dst

**Category D: Structural/Relational** (connections between entities)
- `pair_dominance = pair_freq / max(src_freq, 1)`: how much of src's activity goes to this dst
- `dst_exposure = pair_freq / max(dst_freq, 1)`: how much of dst's activity comes from this src
- `degree_ratio_src = in_degree[src] / max(out_degree[src], 1)`: src's receive/send ratio

### What I think would work best for CLEARSCOPE

V2's success (PRE 0.02326, FP 42) suggests the proven core should be preserved.
V5's improved ADP (0.333) suggests novelty signals help ranking but need
moderation. The parallel decoder experiment shows that splitting the signal
destroys performance.

**Proposed feature set** (10 features, 16 raw dims → 32 proj dim):

| # | Feature | Category | Rationale |
|---|---------|----------|-----------|
| 0 | `t_norm` | Temporal | Proven in V2 |
| 1 | `log_delta_prev` | Temporal | Proven in V2 — captures burstiness |
| 2 | `log_delta_same_pair` | Temporal | Proven in V2 — pair-level timing |
| 3 | `log_delta_src_activity` | Temporal | NEW: time since src last emitted ANY edge. Attack makes browser active after idle → high delta → unusual |
| 4 | `pair_freq_causal` | Volumic | Proven in V2 — how often this pair interacts |
| 5 | `src_unique_dsts / e_idx` | Volumic | NEW: during attack, browser talks to new files/sockets → unique dsts increase → behavioral shift |
| 6 | `pair_type_count / e_idx` | Volumic | NEW: during attack, browser may use different edge types on same pair → diversity increases |
| 7 | `type_rarity` | Novelty | From V5 — but without the over-compression since balanced by other features |
| 8 | `is_new_pair` | Novelty | From V5 — best novelty signal. If never seen before, likely suspicious |
| 9 | `is_new_type_for_src` | Novelty | From V5 — captures browser doing something it never did before |

**Not included** (and why):
- `log_delta_prev2`: redundant with log_delta_prev
- `src_freq_causal`: weak — in V2 it didn't add much discrimination
- `src_type_diversity`: normalizes too much, compresses loss
- `prev_op_norm`: weak signal (just encodes the previous edge type)
- `is_new_type_for_dst`: too aggressive, already covered by is_new_type_for_src and type_rarity
- `pair_dominance` / `dst_exposure`: too coupled to other features, may cause multicollinearity

**Decoder**: Single head (no parallel decoder), `lin_edge: Linear(16, 32)`,
MLP input = 512 + 32 = 544 dims.

### Alternative approaches worth considering

1. **Feature noise**: Add gaussian noise (std=0.05) to temporal features during
   training to prevent over-reliance. This widens the loss distribution by
   making features slightly less reliable.

2. **Feature dropout**: Randomly zero out temporal features with p=0.1 during
   training. Forces the model to be robust and not overly dependent on any
   single feature.

3. **Lower projection dim**: Proj_dim=24 instead of 32. Less capacity means the
   model can't memorize temporal patterns as easily → less compression.

4. **Temperature scaling**: In the cross-entropy loss, use a temperature > 1.0
   to soften the probability distribution → wider loss range → less compression.
