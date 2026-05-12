# PIDSMakerMA — Model Architecture Living Document

- **Branch**: `model-architecture`
- **Conda env (HPC)**: `p_ma`
- **Last updated**: 2026-05-12

---

## 1. Project Goal & Role of Model Architecture

**PIDSMaker** is a framework for building provenance-based intrusion detection systems (PIDSs) using deep learning on system-level provenance graphs (DARPA TC, OpTC, EDR). The **model architecture** layer sits at the core of the pipeline:

```
Provenance Graph → Graph Construction → Featurization → ENCODER → DECODER → Objective → Anomaly Scores
```

The **encoder** consumes node features (embeddings + types) and graph structure to produce node embeddings. The **decoder** consumes those embeddings (and optional edge vectors) to produce predictions for one or more **objectives** (edge type classification, node feature reconstruction, etc.). During inference, the loss/per-edge surprise serves as the anomaly score.

**Our goal**: Systematically explore the architecture space — different GNN variants, pooling strategies, message-passing schemes, decoder designs, multi-task objectives — to find configurations that maximize detection performance (ADP, precision, recall) across diverse datasets.

---

## 2. Current Architectures

### 2.1 Encoders

All encoders accept `(x, edge_index, ...)` and return `{"h": node_embeddings}`.

| Name | Class | File | Notes |
|------|-------|------|-------|
| `none` | `LinearEncoder` | `encoders/linear_encoder.py` | Single `Linear(in_dim, out_dim)`. No message passing. Used by Velox/Orthrus |
| `custom_mlp` | `CustomMLPEncoder` | `encoders/custom_mlp_encoder.py` | Configurable MLP via architecture string (e.g. `linear(0.5) \| relu`) |
| `sage` | `SAGE` | `encoders/sage.py` | `SAGEConv` layers with mean aggregation |
| `gat` | `GAT` | `encoders/gat.py` | `GATConv` layers with multi-head attention |
| `gin` | `GIN` | `encoders/gin.py` | `GINConv`/`GINEConv` with learnable MLP + edge features |
| `graph_attention` | `GraphAttentionEmbedding` | `encoders/graph_attention.py` | `TransformerConv` — transformer-style attention with edge features |
| `glstm` | `GLSTM` | `encoders/glstm.py` | LSTM cells per node, topological ordering, edge-type-specific transforms (NodLink) |
| `rcaid_gat` | `RCaidGAT` | `encoders/rcaid_encoder.py` | 3-layer GAT + skip to MLP aggregator |
| `magic_gat` | `MagicGAT` | `encoders/magic_encoder.py` | Multi-layer GAT + residual + all-layer concat + projection (MAGIC) |
| `sum_aggregation` | `SumAggregation` | `encoders/sum_aggregation.py` | Simple `MessagePassing(aggr="sum")` + 2-layer MLP |

All encoders can be **wrapped with TGN** (`TGNEncoder` in `encoders/tgn_encoder.py`) which adds:
- Optional **memory** (TGNMemory or TimeEncodingMemory) for temporal state tracking
- Optional **time encoding** via learned cosine time features
- Optional **time-order encoding** via GRU on edge sequences
- Neighbor sampling via `LastNeighborLoader`

### 2.2 Decoders

Decoders consume encoder outputs and produce task-specific predictions.

| Name | Class | Input | Output | Notes |
|------|-------|-------|--------|-------|
| `edge_mlp` | `CustomEdgeMLP` | `(h_src, h_dst, edge_vector?)` | `(E, num_classes)` | Project src/dst via `lin_src`/`lin_dst`, optionally add projected `edge_vector`, then MLP |
| `node_mlp` | `CustomMLPDecoder` | `h` | `(N, out_dim)` | Configurable MLP on node embeddings |
| `none` | identity | `h` | `h` | Pass-through |
| `nodlink` | `NodLinkDecoder` | `h` | `(N, out_dim)` | VAE-style: encoder → Gaussian sampling → decoder |
| `magic_gat` | `MagicGat` (as decoder) | `(h, edge_index)` | `(N, out_dim)` | GAT layers, `is_decoder=True` (no concat) |
| `inner_product` | `EdgeInnerProductDecoder` | `(h_src, h_dst)` | `(E,)` | Element-wise product + sum = scalar score |
| `edge_linear` | `EdgeLinearDecoder` | `(h_src, h_dst)` | `(E, 1)` | `lin_src(h_src) + lin_dst(h_dst)` → ReLU → Linear |

The **`edge_mlp` decoder** is the current focus for the Velox Edge experiments:

```python
proj_dim = in_dim * 2 * src_dst_projection_coef  # e.g. 128*2*2 = 512
if edge_vector_proj_dim > 0:
    proj_dim += edge_vector_proj_dim                # +32 = 544

lin_src = Linear(128, 256)
lin_dst = Linear(128, 256)
lin_edge = Linear(14, 32)  # if edge vector present

concat = [lin_src(h_src), lin_dst(h_dst), lin_edge(edge_vector)]
MLP: Linear(544, 272) → ReLU → Linear(272, 10)  # 10 edge types
```

### 2.3 Objectives & Losses

Each objective produces per-edge loss during inference (anomaly score). Training combines objectives additively.

| Objective | Class | Loss | Inputs | Detection mechanism |
|-----------|-------|------|--------|-------------------|
| `predict_edge_type` | `EdgeTypePrediction` | Cross-entropy | `(h_src, h_dst, edge_vector?)` vs edge type | Negative log-prob of predicted type |
| `predict_node_type` | `NodeTypePrediction` | Cross-entropy | `h` vs node type | Surprise in node type prediction |
| `reconstruct_node_features` | `NodeFeatReconstruction` | MSE/SCE/MAE | `h` vs original node features | Reconstruction error |
| `reconstruct_node_embeddings` | `NodeEmbReconstruction` | MSE/SCE | `h` vs self | Auto-encoding error |
| `reconstruct_edge_embeddings` | `EdgeEmbReconstruction` | MSE | `cat(h_src, h_dst)` vs self | Edge auto-encoding |
| `reconstruct_masked_features` | `GMAEFeatReconstruction` | SCE | Masked node features | Mask reconstruction error (MAGIC) |
| `predict_masked_struct` | `GMAEStructPrediction` | BCE | Edge presence prediction | Structure prediction error (MAGIC) |
| `predict_edge_contrastive` | `EdgeContrastivePrediction` | BCE | Positive vs negative edges | Contrastive score |
| `detect_edge_few_shot` | `FewShotEdgeDetection` | Cross-entropy | Edge classification | Supervised anomaly classification |

### 2.4 Current Velox Edge Results (model-architecture branch)

| Version | CS_E3 ADP | CS_E3 PRE | CS_E3 FP | CA_E3 TP | Key change |
|---------|-----------|-----------|----------|----------|------------|
| Vanilla (no edge vector) | 0.167 | 0.00150 | 666 | 3 | Baseline |
| V1 (268-dim edge vector) | — | — | — | — | 96% redundant embeddings |
| V2 (14 causal features, proj 32) | 0.120 | **0.02326** | **42** | 0 | Best CS precision |
| V5 (10 feats, novelty, proj 64) | 0.333 | — | — | 0 | Best ADP (ranking) |
| Current (8 feats, parallel decoder) | 0.125 | 0.00007 | 13649 | — | Regression |

**Core tension**: Better edge features → lower loss on normal edges → `max_val_loss` threshold drops → fewer detections. Solutions must **widen** normal-edge loss distribution while keeping attack-edge loss high.

---

## 3. All Datasets

| Dataset | OS | Attacks | Edges | Subset used | Evaluation sub-dataset |
|---------|----|---------|-------|-------------|----------------------|
| **CADETS_E3** | FreeBSD | 3 | ~2.5M | CA_E3 | CA_E3 |
| **THEIA_E3** | Linux | 2 | ~3M | TH_E3 | TH_E3 |
| **CLEARSCOPE_E3** | Android | 1 | ~1.2M | CS_E3 | CS_E3 |
| **FIVEDIRECTIONS_E3** | Windows | 2 | ~5M | FD_E3 | FD_E3 |
| **TRACE_E3** | Linux | 3 | ~25M | TR_E3 | TR_E3 |
| **CADETS_E5** | FreeBSD | 2 | ~70M | CA_E5 | CA_E5 |
| **THEIA_E5** | Linux | 1 | ~9M | TH_E5 | TH_E5 |
| **CLEARSCOPE_E5** | Android | 2 | ~12M | CS_E5 | CS_E5 |
| **FIVEDIRECTIONS_E5** | Windows | 4 | ~70M | FD_E5 | FD_E5 |
| **TRACE_E5** | Linux | 1 | ~180M | TR_E5 | TR_E5 |
| **optc_h201** | Windows | 1 | — | — | — |
| **optc_h501** | Windows | 1 | — | — | — |
| **optc_h051** | Windows | 1 | — | — | — |
| **ATLASV2_EDR** | Windows | 10 | — | — | — |
| **CARBANAKV2_EDR** | Win+Linux | 1 | — | — | — |

Primary test datasets (current focus): **CS_E3**, **CA_E3**, **TH_E3**, **FD_E3**, **TR_E3**.

---

## 4. Taxonomy of Architectural Choices

### 4.1 GNN Encoder Variants

| Variant | Message-passing scheme | Edge features | Multi-head | Temporal | When to use |
|---------|----------------------|---------------|------------|----------|-------------|
| **Linear** (none) | None | N/A | N/A | No | Fast baseline; isolates decoder effects |
| **SAGE** | Mean-pool neighbor | No | No | No | Scalable; good for large graphs |
| **GAT** | Attention-weighted | No | Yes | No | When neighbor importance varies |
| **TransformerConv** | Transformer attention | Yes | Yes | No | Rich features with edge types |
| **GIN/GINE** | MLP + sum | Yes (GINE) | No | No | Theoretically most expressive |
| **SumAggregation** | Simple sum | No | No | No | Baselines, NodLink |
| **GLSTM** | LSTM over topo order | Yes (typed) | No | Ordering | Sequential provenance (NodLink) |
| **RCaidGAT** | GAT + skip-MLP | No | Yes | No | Attack investigation |
| **MagicGAT** | GAT + residual + concat | Yes | Yes | No | Masked feature learning |
| **TGN-wrapped** | Any above + memory | Via encoder | Per-encoder | Yes (memory) | Temporal graph learning |

### 4.2 Temporal Encoding Strategies

| Strategy | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **TGN memory** | GRU-updated memory per node | Rich temporal state; SOTA for dynamic graphs | Stateful; requires careful reset; large memory footprint |
| **TimeEncoding** | Cosine time features per edge | Lightweight; no memory | No long-term state |
| **Time-order GRU** | GRU over edge sequences | Captures event ordering | Vanishing gradient over long sequences |
| **Causal features** (edge vector) | Hand-engineered temporal stats | Interpretable; dataset-tunable | Manual design effort; feature selection critical |
| **Positional encoding** | Learnable temporal position | Simple; basis for transformers | Limited expressivity alone |

### 4.3 Pooling & Readout Strategies

| Strategy | Mechanism | Use case |
|----------|-----------|----------|
| **Per-edge** (no pooling) | Use `h_src`, `h_dst` directly for edge tasks | Edge type prediction (current) |
| **Global mean/max/sum** | Pool all node embeddings to graph-level | Graph classification |
| **Attention pooling** | Learn weights per node | Graph-level tasks with variable importance |
| **Sort pooling (DGCNN)** | Sort nodes by feature, then conv1d | Graph classification |
| **DiffPool** | Learn hierarchical clustering | Hierarchical graph repr |
| **SAGPool** | Self-attention graph pooling | Hierarchical; learn which nodes to keep |

### 4.4 Decoder Architectures

| Architecture | Structure | Edge vector support | When to use |
|-------------|-----------|---------------------|-------------|
| **CustomEdgeMLP** | `proj(src) + proj(dst) [+ proj(edge_vec)] → MLP` | Yes | **Current default** |
| **CustomMLPDecoder** | `h → MLP` | No | Node-level tasks |
| **EdgeLinearDecoder** | `lin_src(h_src) + lin_dst(h_dst) → ReLU → Linear` | No | Simple additive scoring |
| **EdgeInnerProduct** | `h_src · h_dst` | No | Contrastive learning |
| **Bilinear** | `h_src^T · W · h_dst` | No | Expressive pairwise scoring |
| **Cross-attention** | `attn(h_src, h_dst)` | Yes | Rich edge representation |
| **MagicGAT (decoder)** | GAT on node embeddings | Via edge_dim | Graph-level reconstruction |
| **NodLink VAE** | Encoder → Gaussian → Decoder | No | Probabilistic reconstruction |
| **Multi-head decoder** | Multiple decoder heads summed | Per-head config | Multi-task learning |

### 4.5 Multi-Task & Auxiliary Objectives

| Objective combination | Intuition | When useful |
|----------------------|-----------|-------------|
| `predict_edge_type` alone | Pure edge-type classification | When type is discriminative |
| `predict_edge_type + reconstruct_node_features` | Edge type + node content | Rich self-supervision |
| `predict_edge_type + predict_edge_contrastive` | Classification + contrastive | Better embeddings via negatives |
| `reconstruct_masked_features + predict_masked_struct` | MAGIC-style | Strong SSL for GNN |
| `predict_edge_type + predict_node_type` | Edge + node joint | Heterogeneous graph learning |
| `predict_edge_contrastive` alone | Contrastive only | When no labels available |

### 4.6 Edge Vector Design (Feature Engineering)

| Category | Features | Signal strength | Compression risk |
|----------|----------|----------------|-----------------|
| **Temporal** | `t_norm`, `log_delta_prev`, `log_delta_same_pair`, `log_delta_src_activity` | Medium | High |
| **Volumic** | `pair_freq_causal`, `src_unique_dsts`, `pair_type_count`, `src_freq_causal` | Medium | Medium |
| **Novelty** | `type_rarity`, `is_new_pair`, `is_new_type_for_src`, `is_new_type_for_dst` | High | High (binary flags) |
| **Structural** | `pair_dominance`, `dst_exposure`, `degree_ratio` | Low-Medium | Low |
| **Regularization** | Gaussian noise, feature dropout, lower proj_dim | N/A | N/A (widens loss) |
| **Temperature scaling** | `softmax(logits / T)` with T > 1 | N/A | N/A (softens distribution) |

### 4.7 Message-Passing Schemes (beyond GNN layer choice)

| Scheme | Description | Benefit |
|--------|-------------|---------|
| **Target-to-source** | Messages flow from dst to src | Default for many PyG convs |
| **Source-to-target** | Messages flow from src to dst | Bidirectional variants possible |
| **Directed** | Edge direction matters | Captures causality in provenance |
| **Undirected** | Add reverse edges | More connectivity; simpler |
| **Heterogeneous** | Type-specific message functions | Matches provenance heterogeneity |
| **With edge features** | Edge features in message computation | Richer messages (GINE, TransformerConv) |
| **Skip connections** | Residual/add connections per layer | Deeper GNNs without oversmoothing |

### 4.8 Regularization & Loss Modulation Techniques

| Technique | Mechanism | Effect |
|-----------|-----------|--------|
| Dropout | Randomly zero activations | Prevents co-adaptation |
| MC Dropout | Dropout at inference too | Uncertainty quantification |
| Label smoothing | Soften target distribution | Prevents overconfidence |
| Feature noise (input) | Add Gaussian noise to edge vector | Prevents over-reliance on features |
| Feature dropout (input) | Randomly zero edge vector dims | Forces robustness |
| Loss temperature | `softmax(logits / T)` | Wider loss distribution |
| Gradient clipping | Cap gradient norm | Stable training |
| Weight decay | L2 penalty on weights | Prevents overfitting |
| Balanced loss | Class-weighted cross-entropy | Handles class imbalance |

---

## 5. Step-by-Step Experimental Plan

### Phase 0: Baselines & Infrastructure

- [ ] **P0.1** Reproduce vanilla Velox (no edge vector) on CS_E3, CA_E3, TH_E3
- [ ] **P0.2** Reproduce V2 causal-feature configuration from progress.md
- [ ] **P0.3** Set up standardized evaluation config (fixed hyperparams, seeds, metrics tracking)
- [ ] **P0.4** Confirm W&B logging captures: ADP, PRE, REC, F1, TP, FP, per-epoch loss curves

### Phase 1: Feature Engineering & Decoder Tuning

- [ ] **P1.1** Test V2 feature set + projection dim sweep: {16, 24, 32, 48, 64} on CS_E3
- [ ] **P1.2** Test feature regularization: Gaussian noise std ∈ {0.01, 0.05, 0.1}, feature dropout p ∈ {0.05, 0.1, 0.2}
- [ ] **P1.3** Test temperature scaling: T ∈ {1.0, 1.5, 2.0, 3.0} in cross-entropy
- [ ] **P1.4** Test lower `src_dst_projection_coef`: {1, 2, 3}
- [ ] **P1.5** Test decoder depth: `linear(0.5)|relu` vs `linear(1.0)|relu|linear(0.5)|relu` vs `linear(2.0)|relu|linear(0.5)|relu`
- [ ] **P1.6** Evaluate best config from sweep on CA_E3, TH_E3, FD_E3

### Phase 2: Encoder Variants (no TGN)

- [ ] **P2.1** LinearEncoder (current baseline) — complete
- [ ] **P2.2** SAGE: 2-layer, hid_dim=128
- [ ] **P2.3** GAT: 2-layer, 8 heads, hid_dim=128
- [ ] **P2.4** TransformerConv: 2-layer, 8 heads, with edge_dim
- [ ] **P2.5** GIN/GINE: 2-layer, with edge_dim
- [ ] **P2.6** RCaidGAT: 3-layer GAT + MLP skip
- [ ] **P2.7** For each: evaluate on CS_E3, CA_E3; compare convergence speed, ADP, TP/FP

### Phase 3: Temporal Encoding Strategies

- [ ] **P3.1** TGN memory + LinearEncoder (no GNN underneath)
- [ ] **P3.2** TGN memory + SAGE
- [ ] **P3.3** TGN memory + TransformerConv
- [ ] **P3.4** TimeEncodingMemory (no GRU, just time) + best encoder from P2
- [ ] **P3.5** Time-order GRU encoding + best encoder from P2
- [ ] **P3.6** Compare: does TGN memory help for CADETS (where temporal features hurt)?

### Phase 4: Decoder Variants

- [ ] **P4.1** Bilinear decoder: `h_src^T · W · h_dst`
- [ ] **P4.2** Cross-attention decoder: `attn(h_src, h_dst, edge_vector)`
- [ ] **P4.3** EdgeLinearDecoder: `lin_src(h_src) + lin_dst(h_dst)`
- [ ] **P4.4** InnerProduct (contrastive only)
- [ ] **P4.5** Compare with CustomEdgeMLP baseline on CS_E3

### Phase 5: Multi-Task Objectives

- [ ] **P5.1** `predict_edge_type + reconstruct_node_features` (λ ∈ {0.1, 0.3, 0.5, 1.0})
- [ ] **P5.2** `predict_edge_type + predict_edge_contrastive` (λ ∈ {0.1, 0.3, 0.5})
- [ ] **P5.3** `predict_edge_type + predict_node_type`
- [ ] **P5.4** MAGIC-style: `reconstruct_masked_features + predict_masked_struct`
- [ ] **P5.5** Three-way: `predict_edge_type + reconstruct_node_features + predict_edge_contrastive`
- [ ] **P5.6** Ablation: does multi-task widen the loss distribution?

### Phase 6: Pooling & Hierarchical Architectures

- [ ] **P6.1** SAGPool: learn to keep K nodes per graph
- [ ] **P6.2** DiffPool: hierarchical graph coarsening
- [ ] **P6.3** Global attention pooling over all nodes
- [ ] **P6.4** Compare with per-edge (no pooling) baseline

### Phase 7: Cross-Dataset Evaluation

- [ ] **P7.1** Best architecture from each phase evaluated on all 5 E3 datasets
- [ ] **P7.2** Best architecture evaluated on E5 datasets (CADETS_E5, THEIA_E5, CLEARSCOPE_E5)
- [ ] **P7.3** Ablation: which components generalize vs dataset-specific

### Phase 8: Ablation Studies & Analysis

- [ ] **P8.1** Remove each feature category (temporal, volumic, novelty) individually
- [ ] **P8.2** Vary edge_vector_proj_dim: {0, 8, 16, 32, 64, 128}
- [ ] **P8.3** Vary node_hid_dim/node_out_dim: {32, 64, 128, 256}
- [ ] **P8.4** Vary dropout: {0.0, 0.1, 0.3, 0.5}
- [ ] **P8.5** Vary learning rate: {1e-5, 5e-5, 1e-4, 5e-4}
- [ ] **P8.6** Loss distribution analysis: plot histograms of normal vs attack edge losses

### Phase 9: Advanced Architectures (Exploratory)

- [ ] **P9.1** Heterogeneous GNN: type-specific message functions per (src_type, dst_type)
- [ ] **P9.2** Graph Transformer: full transformer over nodes with edge-type-aware attention
- [ ] **P9.3** Temporal Graph Transformer (TGT): combine TGN memory with transformer attention
- [ ] **P9.4** GraphRNN: sequence of graph snapshots with RNN
- [ ] **P9.5** Contrastive pre-training: learn encoder via contrastive loss, fine-tune with edge type

---

## 6. Progress Tracker

### Experiments Log

| Date | Phase | Experiment | Config | CS_E3 ADP | CS_E3 PRE | CS_E3 TP/FP | CA_E3 TP | Notes |
|------|-------|------------|--------|-----------|-----------|-------------|----------|-------|
| — | P0.1 | Vanilla Velox | no edge vector | 0.167 | 0.00150 | 1/666 | 3 | Baseline from progress.md |
| — | P0.2 | V2 causal features | 14-dim, proj 32 | 0.120 | 0.02326 | 1/42 | 0 | Best CS precision |
| — | P1.x | V5 novelty | 10-dim, proj 64 | 0.333 | — | 0/? | 0 | Best ADP |
| — | P1.x | Parallel decoder | 8-dim, λ=0.3 | 0.125 | 0.00007 | 1/13649 | — | Regression |

### Architecture Decision Record

| Date | Decision | Rationale | Outcome |
|------|----------|-----------|---------|
| — | Edge vector should exclude node embeddings | 96% redundancy in V1 | V2's 14-dim vector |
| — | Causal feature computation | Prevents data leakage | V2+ |
| — | Remove parallel decoder | Node-only head adds noise | Confirmed regression |
| — | Single-head decoder | Two heads compete for shared weights | Simpler, better |

### Checklist

- [ ] Phase 0: Baselines
- [ ] Phase 1: Feature & Decoder Sweeps
- [ ] Phase 2: Encoder Variants
- [ ] Phase 3: Temporal Encoding
- [ ] Phase 4: Decoder Variants
- [ ] Phase 5: Multi-Task Objectives
- [ ] Phase 6: Pooling & Hierarchical
- [ ] Phase 7: Cross-Dataset
- [ ] Phase 8: Ablation Studies
- [ ] Phase 9: Advanced Architectures

---

## 7. Running Experiments

### Standard command pattern

```bash
# Train + evaluate
./run.sh velox-edge DATASET \
  --training.encoder.dropout=0.3 \
  --training.lr=0.0001 \
  --training.node_hid_dim=128 \
  --training.node_out_dim=128 \
  --featurization.emb_dim=128 \
  --construction.time_window_size=15.0 \
  --project=PIDSHyp \
  --exp=EXP_NAME \
  --force_restart construction
```

### Key metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| ADP | Average Detection Precision (ranking) | Higher is better |
| PRE | Precision = TP / (TP + FP) | Higher is better |
| REC | Recall = TP / (TP + FN) | Higher is better |
| F1 | Harmonic mean of PRE and REC | Higher is better |
| TP | True positives (edges) | Maximize |
| FP | False positives (edges) | Minimize |
| Val loss (epoch) | Training convergence | Should decrease steadily |
| Loss distribution spread | `std(loss_val)` on validation set | Wider is better for threshold |

---

## 8. Appendix: Adding a New Architecture

1. **Encoder**: Create class in `pidsmaker/encoders/`, add import to `__init__.py`, add factory branch in `factory.py:encoder_factory()`
2. **Decoder**: Create class in `pidsmaker/decoders/`, add import to `__init__.py`, add factory branch in `factory.py:decoder_factory()`
3. **Objective**: Create class in `pidsmaker/objectives/`, add import to `__init__.py`, add factory branch in `factory.py:objective_factory()`
4. **Config**: Add relevant YAML block to `config/default.yml` and system-specific config
5. **Test**: Run on CS_E3 first (smallest), then expand
6. **Log**: Update this document with results and ADR
