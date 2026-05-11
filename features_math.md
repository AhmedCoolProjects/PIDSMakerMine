# Feature Math — Velox Edge (10 Features)

## Setup

All features are computed **causally**: edges in a time window are sorted by
timestamp, then iterated once. Only information from edges seen **before** the
current edge is used to compute its features.

```python
edge_list.sort(key=lambda e: e["time"])
window_size_ns = max(window_end_ns - window_start_ns, 1)

last_time = None                     # timestamp of the previous edge (any src/dst)
last_pair_time = {}                  # last timestamp per (src, dst)
last_src_time = {}                   # last timestamp per src
pair_count = Counter()               # how many times (src, dst) seen so far
type_count = Counter()               # how many times this edge type seen so far
max_type_count = 0                   # max of type_count values at this point
src_uniq_types = defaultdict(set)    # set of edge types used by each src
src_unique_dsts = defaultdict(set)   # set of unique dsts per src
pair_types = defaultdict(set)        # set of edge types used between (src, dst)

for e_idx, e in enumerate(edge_list):
    t = e["time"]
    src, dst = e["src"], e["dst"]
    op_id = rel2id.get(e["label"], 0)
    normalizer = max(e_idx, 1)

    # compute features here...

    # update state after computing
    last_time = t
    last_src_time[src] = t
    last_pair_time[(src, dst)] = t
    pair_count[(src, dst)] += 1
    type_count[op_id] += 1
    max_type_count = max(max_type_count, max(type_count.values(), default=0))
    src_uniq_types[src].add(op_id)
    src_unique_dsts[src].add(dst)
    pair_types[(src, dst)].add(op_id)
```

---

## Feature 0 — `t_norm` (Temporal position)

```python
(t - start_time) / window_size_ns
```

- **Range**: `[0.0, 1.0]`
- **What**: Where in the time window this edge falls. First edge → 0.0, last
  edge in a 15min window → 1.0.
- **Why**: Separates early-window edges from late-window edges. Attack edges
  may cluster at a specific time offset from the window start.

---

## Feature 1 — `log_delta_prev` (Global inter-arrival time)

```python
math.log1p((t - last_time) / 1e9) if last_time is not None else 0.0
```

- **Range**: `[0.0, ~16.1]` (log1p of up to ~10⁷s)
- **What**: `log(1 + seconds_since_the_previous_edge_in_the_window)`. First
  edge → 0.0. If edges are 0.1s apart → `log1p(0.1) ≈ 0.095`. If 10s apart
  → `log1p(10) ≈ 2.40`.
- **`log1p(x) = ln(1 + x)`**: smooth for small x, wide dynamic range for
  large x. Same value whether gap is 0.01s or 0.02s (both ≈0.01–0.02),
  differentiates 1s vs 10s.
- **Why**: Captures global burstiness. Attack bursts have different timing
  than normal background noise.

---

## Feature 2 — `log_delta_same_pair` (Pair-level inter-arrival time)

```python
math.log1p((t - last_pair_time.get((src, dst), t)) / 1e9)
```

- **Range**: `[0.0, ~16.1]`
- **What**: `log(1 + seconds_since_this_(src,dst)_pair_last_interacted)`. If
  first time → `log1p(0) = 0.0` (because `last_pair_time.get((s,d), t)` falls
  back to `t`, making `t - t = 0`).
- **Why**: A pair that interacts every 0.01s has low delta (normal loop). A
  pair that hasn't interacted in hours then suddenly does → high delta
  (suspicious — first contact in a long time, or brand-new relationship).

---

## Feature 3 — `log_delta_src_activity` (Source-level idle time)

```python
math.log1p((t - last_src_time.get(src, t)) / 1e9)
```

- **Range**: `[0.0, ~16.1]`
- **What**: `log(1 + seconds_since_this_src_last_emitted_any_edge)`. First
  edge from this src → 0.0.
- **Why**: A src that was idle for 5 minutes then suddenly emits edges (e.g.,
  dormant malware waking up) gets a high delta. A constantly-active process
  gets low deltas. Captures **dormancy → burst** transitions that attacks
  often trigger.

---

## Feature 4 — `pair_freq_causal` (Pair frequency)

```python
pair_count[(src, dst)] / normalizer
```

- **Range**: `[0.0, 1.0]`
- **What**: How many times this (src,dst) pair has been seen so far, divided
  by total edges seen so far (`e_idx`). First time → 0.0. If this is the 50th
  time they interact by edge 1000 → `50 / 1000 = 0.05`.
- **Why**: Dominant pairs (frequently interacting processes/files) get high
  values, which the model learns as "normal interaction pattern." Novel or
  rare pairs get low values → higher loss → potentially anomalous.

---

## Feature 5 — `src_unique_dsts / normalizer` (Source behavioral spread)

```python
len(src_unique_dsts[src]) / normalizer
```

- **Range**: `[0.0, 1.0]`
- **What**: How many **unique destinations** this src has connected to so far,
  divided by `e_idx`. First edge from src → `1 / e_idx ≈ 0`. As src talks to
  more different dsts, this grows.
- **Why**: A process that suddenly spreads activity to many new destinations
  (e.g., compromised browser writing to many files, connecting to many IPs)
  sees a sharp rise in this feature. Normal processes tend to talk to a
  consistent, limited set of destinations.

---

## Feature 6 — `pair_type_count / normalizer` (Relationship diversity)

```python
len(pair_types[(src, dst)]) / normalizer
```

- **Range**: `[0.0, 1.0]`
- **What**: How many **different edge types** have been used between this
  (src,dst) pair, divided by `e_idx`. First interaction → `1 / e_idx ≈ 0`.
  If a pair normally uses only WRITE but suddenly starts using READ too →
  `2 / e_idx`.
- **Why**: Captures shifts in the nature of the relationship. A browser that
  normally only WRITEs to a config file but then EXECUTEs it (launching
  malware) → the pair_type_count rises, signaling a behavioral change.

---

## Feature 7 — `type_rarity` (Edge type rarity)

```python
max_type_count = max(max_type_count, max(type_count.values(), default=0))
type_rarity = 1 - (type_count.get(op_id, 0) / max(max_type_count, 1))
```

- **Range**: `[0.0, 1.0]`
- **What**: How rare this edge type is among all edges seen so far. The most
  common edge type → 0.0. A type that has appeared only once while other types
  have appeared 1000 times → `1 - 1/1000 = 0.999`.
- **Why**: Rare edge types are more likely to be anomalous. In CLEARSCOPE,
  `EVENT_EXECUTE` might be rare compared to `EVENT_READ`/`EVENT_WRITE`.
  An attack process doing `EVENT_EXECUTE` would get a high type_rarity →
  high loss.

---

## Feature 8 — `is_new_pair` (Novel pair indicator)

```python
0.0 if (src, dst) in last_pair_time else 1.0
```

- **Range**: `{0.0, 1.0}` (binary)
- **What**: `1.0` if this (src,dst) pair has never been seen before in this
  window, `0.0` otherwise.
- **Why**: Strongest novelty signal. A process talking to a destination it has
  never communicated with is inherently suspicious. Hard binary avoids the
  normalization compression that continuous features suffer from. But because
  it's binary, it's also the most aggressive — every new pair is flagged.

---

## Feature 9 — `is_new_type_for_src` (Novel edge-type for source)

```python
0.0 if op_id in src_uniq_types[src] else 1.0
```

- **Range**: `{0.0, 1.0}` (binary)
- **What**: `1.0` if this src has never used this edge type before in this
  window, `0.0` otherwise.
- **Why**: A process that suddenly performs a type of system call it has never
  made before is suspicious. For example, a browser that normally only does
  READ/WRITE but suddenly does EXECUTE → flagged.


## Summary

| Feature | What it Captures | Why it's Useful for Detection | Math |
|---------|------------------|------------------------------|------|
| `t_norm` | Temporal position in the window | Attack edges may cluster at specific time offsets | `(t - start_time) / window_size_ns` |
| `log_delta_prev` | Global inter-arrival time | Captures burstiness; attacks may have different timing patterns than normal noise | `log1p((t - last_time) / 1e9)` |
| `log_delta_same_pair` | Pair-level inter-arrival time | A long gap followed by interaction → suspicious | `log1p((t - last_pair_time.get((src, dst), t)) / 1e9)` |
| `log_delta_src_activity` | Source-level idle time | A dormant process waking up → suspicious | `log1p((t - last_src_time.get(src, t)) / 1e9)` |
| `pair_freq_causal` | Pair frequency so far | Novel or rare pairs get low values → potentially anomalous | `pair_count[(src, dst)] / normalizer` |
| `src_unique_dsts / normalizer` | Source behavioral spread | A process suddenly talking to many new destinations → suspicious | `len(src_unique_dsts[src]) / normalizer` |
| `pair_type_count / normalizer` | Relationship diversity | A shift in the nature of interaction (e.g., new edge type between a pair) → suspicious | `len(pair_types[(src, dst)]) / normalizer` |
| `type_rarity` | Edge type rarity | Rare edge types are more likely to be anomalous | `1 - (type_count.get(op_id, 0) / max(max_type_count, 1))` |
| `is_new_pair` | Novel pair indicator | A process talking to a never-before-seen destination is inherently suspicious | `0.0 if (src, dst) in last_pair_time else 1.0` |
| `is_new_type_for_src` | Novel edge-type for source | A process performing a system call type it has never made before is suspicious | `0.0 if op_id in src_uniq_types[src] else 1.0` |

## Results

### Original Velox

|Dataset| TP | FP | Pre | Rec | F1 |
|----|----|----|-----|-----|----|
|CA E3| 8  | 0  | 1.0 | 0.1176 | 0.2105 |
|CS E3 | 1 | 913  | 0.0011 | 0.0244 | 0.0021 |

### V6 Velox Edge

|Dataset| TP | FP | Pre | Rec | F1 |
|----|----|----|-----|-----|----|
|CA E3| 0  | 8  | 0.0 | 0.0 | 0.0 |
|CS E3 | 1  | 64  | 0.0154 | 0.0244 | 0.0189 |

