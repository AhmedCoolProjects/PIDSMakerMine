# PIDSMakerDO — Detection Objectives (Living Planning Document)

- **Branch**: `detection-objectives`
- **Conda env (HPC)**: `p_do`
- **Last updated**: 2026-05-12

---

## 1. Project Goal

This project systematically explores **detection objectives** for unsupervised anomaly detection in provenance graphs. The core idea: train a graph neural network to solve a pretext task (reconstructing or predicting some property of the graph), then use the **per-edge reconstruction/prediction loss as an anomaly score** — edges (and their incident nodes) that are hard to reconstruct are flagged as suspicious.

This branch (`detection-objectives`) exists to:
1. Formulate, implement, and test a broad taxonomy of detection objectives
2. Evaluate each objective on multiple real-world security datasets
3. Track results in a single source of truth so we can iteratively add, test, compare, and decide on objectives

---

## 2. Project Architecture

### Pipeline (task DAG)

```
construction → transformation → featurization → feat_inference → batching → training → evaluation → triage
```

Each task produces cached artifacts. If the config hasn't changed, tasks are skipped.

### Core Model

```
Model(encoder, objectives=[obj1, obj2, ...])
```

- **Encoder** (`pidsmaker/encoders/`): GNN producing node embeddings `h` (shape `[N, d]`)
- **Decoders** (`pidsmaker/decoders/`): MLP heads that decode embeddings to the target space
- **Objectives** (`pidsmaker/objectives/`): Each computes a loss = reconstruction/prediction error
- **Inference**: Per-edge loss → aggregated node-level anomaly score
- **Evaluation**: AUC-ROC, AP, ADP, Precision/Recall, F1, MCC, Discrimination

### Training Loop

1. Self-supervised pretraining: encoder + SSL objectives, all train graphs
2. (Optional) Few-shot fine-tuning: freeze encoder, train a detection head on labelled validation edges
3. Inference every 2 epochs on val/test sets
4. Best-epoch selection: `best_adp` (highest ADP) or `best_discrimination`

---

## 3. Current State of Detection Objectives

### 3.1 Implemented Objectives (9 classes)

| # | Objective | File | Level | Target | Loss | Used By |
|---|-----------|------|-------|--------|------|---------|
| 1 | `NodeFeatReconstruction` | `reconstruct_node_feat.py` | Node | Input node features `x` | MSE / SCE / MAE | NodLink |
| 2 | `NodeEmbReconstruction` | `reconstruct_node_emb.py` | Node | Node embeddings `h` (self) | MSE / SCE / MAE | — |
| 3 | `EdgeEmbReconstruction` | `reconstruct_edge_emb.py` | Edge | Concatenated src/dst embeddings `cat(h_src, h_dst)` | MSE / SCE / MAE | — |
| 4 | `EdgeTypePrediction` | `predict_edge_type.py` | Edge | Edge type (system call, 10/33 classes) | Cross-entropy | Orthrus, Kairos |
| 5 | `NodeTypePrediction` | `predict_node_type.py` | Node | Node type (subject/file/netflow, 3 classes) | Cross-entropy | Flash, ThreaTrace, R-Caid |
| 6 | `EdgeContrastivePrediction` | `predict_edge_contrastive.py` | Edge | Positive vs negative edge scores | BCE contrastive | — |
| 7 | `GMAEFeatReconstruction` | `reconstruct_masked_feat.py` | Node | Masked node features (MAGIC-style) | SCE | MAGIC |
| 8 | `GMAEStructPrediction` | `predict_masked_struct.py` | Edge | Edge existence (link pred, binary) | BCE | MAGIC |
| 9 | `FewShotEdgeDetection` | `detect_few_shot.py` | Edge | Edge anomaly (benign/malicious, 2 classes) | Cross-entropy | — |

### 3.2 Available Encoders (13)

| Encoder | File | Used By |
|---------|------|---------|
| `GraphAttentionEmbedding` | `graph_attention.py` | Orthrus, Velox |
| `TGNEncoder` | `tgn_encoder.py` | Orthrus, Kairos, Velox |
| `SAGE` | `sage.py` | Flash, ThreaTrace |
| `GAT` | `gat.py` | General |
| `GIN` | `gin.py` | General |
| `MagicGAT` | `magic_encoder.py` | MAGIC |
| `RCaidGAT` | `rcaid_encoder.py` | R-Caid |
| `GLSTM` | `glstm.py` | NodLink |
| `SumAggregation` | `sum_aggregation.py` | NodLink |
| `GRU` | `gru.py` | General |
| `LinearEncoder` | `linear_encoder.py` | Velox |
| `CustomMLPEncoder` | `custom_mlp_encoder.py` | Custom |
| `CustomMLPAbstract` | `custom_mlp.py` | Shared base |

### 3.3 Available Decoders (5)

| Decoder | Level | Description |
|---------|-------|-------------|
| `CustomMLPDecoder` | Node | Configurable MLP for node-level tasks |
| `CustomEdgeMLP` | Edge | MLP with separate src/dst projections + optional edge vector |
| `EdgeLinearDecoder` | Edge | Simple linear → ReLU → linear(1) |
| `EdgeInnerProductDecoder` | Edge | Dot product of src/dst embeddings |
| `NodLinkDecoder` | Node | Variational autoencoder (mu+sigma → reparameterize → decode) |

### 3.4 Loss Functions

| Loss | Function | Used For |
|------|----------|----------|
| `sce_loss` | `(1 - cos_sim)^α` | Feature reconstruction |
| `mse_loss` | MSE | Reconstruction |
| `mae_loss` | L1 | Reconstruction |
| `bce_contrastive` | BCE on pos/neg scores | Contrastive |
| `cross_entropy` | Cross-entropy | Classification |
| `binary_cross_entropy` | BCE | Binary classification |

### 3.5 System Config ↔ Objective Mapping

| System | Objective(s) | Encoder |
|--------|-------------|---------|
| Orthrus | `predict_edge_type` | TGN + GraphAttention |
| Orthrus Fixed | `predict_edge_type` (edge-level eval) | TGN + GraphAttention |
| Velox | `predict_edge_type` | Linear (no GNN) |
| Velox Edge | `predict_edge_type` (+ engineered edge features) | Linear |
| Flash | `predict_node_type` | SAGE |
| Kairos | `predict_edge_type` | GraphAttention + TGN |
| NodLink | `reconstruct_node_features` | SumAggregation |
| MAGIC | `reconstruct_masked_features` + `predict_masked_struct` | MagicGAT |
| R-Caid | `predict_node_type` | RCaidGAT |
| ThreaTrace | `predict_node_type` | SAGE |

---

## 4. Datasets

| Dataset | Year | # Edge Types | # GT Files | Attacks | Scope |
|---------|------|-------------|------------|---------|-------|
| **DARPA TC E3** (5 sub-datasets) | 2018 | 10 | Varies | Drakon, Backdoor, Phishing | System-call provenance |
| **DARPA TC E5** (5 sub-datasets) | 2019 | 10 | Varies | Drakon APT, Nginx Backdoor | System-call provenance |
| **OpTC** (3 sub-datasets) | 2019 | 10 | 1 each | Unauthorized access | System-call provenance |
| **ATLASV2_EDR** | 2022 | 33 | 10 | Multi-stage attack (10 phases) | EDR alerts |
| **CARBANAKV2_EDR** | 2024 | 33 | 1 | Long-duration APT campaign | EDR alerts |

Total: **16 dataset configurations** across DARPA TC (E3 + E5), OpTC, ATLAS, and CARBANAK.

### Dataset partitioning (per dataset)
- **Train**: benign-only dates (no attacks, no malicious nodes)
- **Validation**: used for threshold calibration (no attacks)
- **Test**: contains the attack(s) — ground-truth labels used only for evaluation

---

## 5. Taxonomy of Detection Objectives

### 5.1 Reconstruction Objectives

Reconstruct a masked or corrupted signal. Anomalous edges/nodes have higher reconstruction error.

| ID | Objective | Target | Level | Loss | Notes |
|----|-----------|--------|-------|------|-------|
| R1 | Node feature reconstruction | `x` (input features) | Node | MSE/SCE/MAE | ✅ Implemented |
| R2 | Node embedding reconstruction | `h` (self) | Node | MSE/SCE/MAE | ✅ Implemented |
| R3 | Edge embedding reconstruction | `cat(h_src, h_dst)` | Edge | MSE/SCE/MAE | ✅ Implemented |
| R4 | GMAE masked feature reconstruction | `x[masked]` | Node | SCE | ✅ Implemented (MAGIC) |
| R5 | **Subgraph reconstruction** | Subgraph adjacency around each node | Node | BCE/SCE | 🆕 Reconstruct neighborhood structure |
| R6 | **Edge feature reconstruction** | Engineered edge vector | Edge | MSE | 🆕 Use `edge_vector` as target |
| R7 | **Temporal edge attribute recon.** | Time deltas, inter-arrival times | Edge | MSE/MAE | 🆕 Predict temporal features |

### 5.2 Prediction / Classification Objectives

Predict a discrete property of the graph. Anomalies violate the predicted pattern.

| ID | Objective | Target | Level | Loss | Notes |
|----|-----------|--------|-------|------|-------|
| P1 | Edge type prediction | System call type (10/33 classes) | Edge | Cross-entropy | ✅ Implemented |
| P2 | Node type prediction | Entity type (3 classes) | Node | Cross-entropy | ✅ Implemented |
| P3 | GMAE structure prediction | Edge existence (binary) | Edge | BCE | ✅ Implemented (MAGIC) |
| P4 | **Edge direction prediction** | Direction src→dst vs dst→src | Edge | Cross-entropy | 🆕 For undirected graphs |
| P5 | **Node degree prediction** | Degree bucket (categorical) | Node | Cross-entropy | 🆕 Anomalous nodes have unexpected degree |
| P6 | **Edge type + direction joint** | Combined (type × 2) | Edge | Cross-entropy | 🆕 Joint prediction |
| P7 | **Node role prediction** | Graph role (hub, bridge, leaf) | Node | Cross-entropy | 🆕 Unsupervised role via community detection |

### 5.3 Contrastive Objectives

Learn to distinguish positive pairs (existing edges) from negative pairs (corrupted edges).

| ID | Objective | Positives | Negatives | Level | Notes |
|----|-----------|-----------|-----------|-------|-------|
| C1 | Edge contrastive | Existing edges | Randomly corrupted edges | Edge | ✅ Implemented |
| C2 | **Node contrastive (InfoNCE)** | Same node across views | Different nodes | Node | 🆕 Graph augmentation views |
| C3 | **Node-subgraph contrastive** | Node ↔ its subgraph | Node ↔ other subgraphs | Node | 🆕 Deep Graph Infomax-style |
| C4 | **Temporal contrastive** | Edges in same time window | Edges across windows | Edge | 🆕 Time-aware negatives |
| C5 | **Type-aware contrastive** | Same-type edges | Different-type edges | Edge | 🆕 Negatives sampled from diff edge types |

### 5.4 Masked Modeling Objectives

Mask a portion of the graph and predict the masked content.

| ID | Objective | Mask | Predict | Level | Notes |
|----|-----------|------|---------|-------|-------|
| M1 | GMAE feature masking | Nodes (50%) | Masked features | Node | ✅ Implemented (MAGIC) |
| M2 | GMAE structure masking | Random edges | Edge existence | Edge | ✅ Implemented (MAGIC) |
| M3 | **Edge type masking** | Edge features | Masked edge type | Edge | 🆕 |
| M4 | **Path/random walk masking** | Subpaths | Masked node/edge | Node/Edge | 🆕 Mask continuous paths |
| M5 | **Node feature corruption** | Add noise to x | Recover original | Node | 🆕 Denoising autoencoder variant |

### 5.5 Multi-Task Combinations

| ID | Combination | Objectives | Notes |
|----|------------|------------|-------|
| T1 | MAGIC | R4 + P3 | ✅ Implemented |
| T2 | **Node recon + edge type pred** | R1 + P1 | 🆕 |
| T3 | **Node recon + edge contrastive** | R1 + C1 | 🆕 |
| T4 | **Edge type + edge embedding** | P1 + R3 | 🆕 |
| T5 | **Node type + node feat + edge type** | P2 + R1 + P1 | 🆕 Triple objective |

### 5.6 Few-Shot / Supervised Objectives

| ID | Objective | Target | Level | Notes |
|----|-----------|--------|-------|-------|
| F1 | Few-shot edge detection | Benign/malicious | Edge | ✅ Implemented (needs updating) |

---

## 6. Experimental Plan

### 6.1 Evaluation Framework

Each objective is evaluated with a **fixed experimental protocol**:

| Component | Setting |
|-----------|---------|
| **Encoder** | GraphAttentionEmbedding (2 layers, 8 heads, 64-dim out) |
| **TGN** | No TGN memory (unless temporal) |
| **Featurization** | word2vec (128-dim), node_type one-hot |
| **Edge features** | edge_type (10-dim one-hot) |
| **Node features** | node_emb + node_type |
| **Batch size** | 1024 edges (intra-graph) |
| **Epochs** | 12 |
| **Learning rate** | 1e-5 |
| **Optimizer** | Adam |
| **Seed** | 0 |
| **Val threshold** | max_val_loss |
| **Evaluation** | Node-level (src node loss aggregation) |
| **Best model** | best_adp |

### 6.2 Evaluation Metrics (per dataset)

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Area under ROC curve |
| **AP** | Average Precision (precision-recall AUC) |
| **ADP** | Average Detection Precision (precision vs detected attacks) |
| **Precision / Recall** | At threshold |
| **F1** | Harmonic mean of precision & recall |
| **FPR** | False positive rate |
| **MCC** | Matthews correlation coefficient |
| **Discrimination** | Rank separation of attack vs benign nodes |
| **FP-in-malicious-TW** | False positives occurring in attack time windows |

### 6.3 Datasets to Test (prioritized)

| Priority | Dataset | Rationale |
|----------|---------|-----------|
| P0 | CADETS_E3 | 4 attacks, simple, fast to iterate |
| P0 | THEIA_E3 | 2 attacks, moderate size |
| P1 | THEIA_E5 | 2 attacks, E5 ground truth |
| P1 | CADETS_E5 | 2 attacks |
| P1 | optc_h201 | 1 attack, OpTC data |
| P2 | FIVEDIRECTIONS_E5 | 4 attacks |
| P2 | CLEARSCOPE_E5 | 3 attacks |
| P3 | ATLASV2_EDR | 33 edge types, 10 attack phases |
| P3 | CARBANAKV2_EDR | 33 edge types, long-duration |

### 6.4 Experiment Tracking

Each experiment run records:
- Objective name + config fingerprint
- Dataset
- Encoder architecture
- All metrics (AUC, AP, ADP, P, R, F1, FPR, MCC, Discrimination)
- Per-attack recall
- Best epoch and threshold
- W&B run link
- Date and branch commit hash

Results are logged in `DETECTION_OBJECTIVES_RESULTS.md` (to be created).

---

## 7. Implementation Roadmap

### Phase 1: Baseline (Current)
- [ ] R1 (NodeFeatReconstruction) — on baseline encoder
- [ ] R2 (NodeEmbReconstruction) — on baseline encoder
- [ ] R3 (EdgeEmbReconstruction) — on baseline encoder
- [ ] P1 (EdgeTypePrediction) — already runs as Orthrus
- [ ] P2 (NodeTypePrediction) — already runs as Flash/ThreaTrace
- [ ] C1 (EdgeContrastive) — on baseline encoder

### Phase 2: New Objectives
- [ ] R5 — Subgraph reconstruction
- [ ] R6 — Edge feature reconstruction
- [ ] R7 — Temporal edge attribute reconstruction
- [ ] P4 — Edge direction prediction
- [ ] P5 — Node degree prediction
- [ ] P6 — Edge type + direction joint prediction
- [ ] P7 — Node role prediction
- [ ] C2 — Node contrastive (InfoNCE)
- [ ] C3 — Node-subgraph contrastive (DGI)
- [ ] C4 — Temporal contrastive
- [ ] C5 — Type-aware contrastive
- [ ] M3 — Edge type masking
- [ ] M4 — Path/random walk masking
- [ ] M5 — Node feature corruption (denoising)

### Phase 3: Multi-objective Combinations
- [ ] T2 — Node recon + edge type pred
- [ ] T3 — Node recon + edge contrastive
- [ ] T4 — Edge type + edge embedding
- [ ] T5 — Node type + node feat + edge type (triple)

### Phase 4: Analysis & Selection
- [ ] Compare all single objectives head-to-head on CADETS_E3
- [ ] Ablation: effect of encoder choice
- [ ] Ablation: effect of loss function (MSE vs SCE vs MAE)
- [ ] Ablation: effect of edge features (type only vs +triplet vs +vector)
- [ ] Identify best 3-5 objectives for each dataset family
- [ ] Write analysis / paper section

---

## 8. Progress Table

| # | Objective | Implemented? | CADETS_E3 | THEIA_E3 | THEIA_E5 | optc_h201 | ATLASV2 | Notes |
|---|-----------|:-----------:|:--------:|:--------:|:--------:|:--------:|:-------:|-------|
| R1 | NodeFeatReconstruction | ✅ | | | | | | Baseline |
| R2 | NodeEmbReconstruction | ✅ | | | | | | Baseline |
| R3 | EdgeEmbReconstruction | ✅ | | | | | | Baseline |
| R4 | GMAEFeatReconstruction | ✅ | | | | | | MAGIC baseline |
| R5 | SubgraphReconstruction | 🔲 | | | | | | |
| R6 | EdgeFeatReconstruction | 🔲 | | | | | | |
| R7 | TemporalAttrReconstruction | 🔲 | | | | | | |
| P1 | EdgeTypePrediction | ✅ | | | | | | Orthrus baseline |
| P2 | NodeTypePrediction | ✅ | | | | | | Flash/TR baseline |
| P3 | GMAEStructPrediction | ✅ | | | | | | MAGIC baseline |
| P4 | EdgeDirectionPrediction | 🔲 | | | | | | |
| P5 | NodeDegreePrediction | 🔲 | | | | | | |
| P6 | EdgeTypeDirectionJoint | 🔲 | | | | | | |
| P7 | NodeRolePrediction | 🔲 | | | | | | |
| C1 | EdgeContrastive | ✅ | | | | | | Baseline |
| C2 | NodeContrastiveInfoNCE | 🔲 | | | | | | |
| C3 | NodeSubgraphContrastive | 🔲 | | | | | | |
| C4 | TemporalContrastive | 🔲 | | | | | | |
| C5 | TypeAwareContrastive | 🔲 | | | | | | |
| M3 | EdgeTypeMasking | 🔲 | | | | | | |
| M4 | PathMasking | 🔲 | | | | | | |
| M5 | NodeFeatureCorruption | 🔲 | | | | | | |
| T2 | R1 + P1 | 🔲 | | | | | | |
| T3 | R1 + C1 | 🔲 | | | | | | |
| T4 | P1 + R3 | 🔲 | | | | | | |
| T5 | P2 + R1 + P1 | 🔲 | | | | | | |

Legend: ✅ = implemented, 🔲 = not yet implemented. Empty cells = results to fill in.

---

## 9. Adding a New Objective: Checklist

When adding a new detection objective, follow these steps:

1. **Define the objective class** in `pidsmaker/objectives/<name>.py`
   - Follow the existing pattern: `class MyObjective(nn.Module)` with `forward()` returning `{"loss": ...}`
   - Support both `inference=False` (scalar loss) and `inference=True` (per-edge/per-node loss tensor)
   - The loss tensor must be edge-indexed (length = num edges) for node-level evaluation

2. **Register in `__init__.py`** — add import to `pidsmaker/objectives/__init__.py`

3. **Add to factory** — extend `objective_factory()` in `pidsmaker/factory.py`

4. **Add to config** — add schema entry in `pidsmaker/config/config.py` (`OBJECTIVES_CFG`, `OBJECTIVES_*_LEVEL` lists)

5. **Add default config** — add default params in `config/default.yml`

6. **Test** — run `python -m pytest tests/test_framework.py -k TestEncoderObjective` to verify

7. **Run experiments** — use existing config (inherit from orthrus/velox/etc.) or create a new one

8. **Log results** — update this document's progress table and create a results entry

---

## 10. Key Architectural Notes

- **Node-level evaluation** aggregates per-edge losses to nodes via `max(src_loss, dst_loss)`.
- **Per-edge losses during inference**: Each objective returns a 1D tensor of length `E` (num edges). Sum across objectives → final anomaly score per edge.
- **MAGIC objectives return zero losses during inference** because they compute global masked loss (not per-edge). This means node-level eval is not straightforward with these objectives alone — a workaround (e.g., nearest-neighbor distance in embedding space) is used.
- **The `ValidationWrapper`** wraps every objective to track validation scores. It's transparent in the forward pass but collects scores for few-shot.
- **TGN memory** can be enabled (`encoder: tgn`) or disabled. Most experiments should start without TGN for speed, then re-run best objectives with TGN.

---

*This document is the single source of truth for detection objectives. Update it whenever a new objective is added, an experiment is run, or results are collected.*
