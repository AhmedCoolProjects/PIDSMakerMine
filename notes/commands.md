## Velox Hyp CS 3
```bash
python pidsmaker/main.py velox CLEARSCOPE_E3 --wandb --database_host localhost --artifact_dir ./artifacts --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1.0
```

```bash
./run.sh velox CLEARSCOPE_E3 --database_host localhost --artifact_dir ./artifacts --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1.0 --project=PIDSHyp --exp=velox_cs_e3_runs
```

## Velox Hyp CA 3
```bash
python pidsmaker/main.py velox CADETS_E3 --wandb --database_host localhost --artifact_dir ./artifacts --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256 --project=PIDSHyp --exp=velox_ca_e3
```

python pidsmaker/main.py velox-edge CADETS_E3 --wandb --database_host localhost --artifact_dir ./artifacts --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256 --project=PIDSHyp --exp=velox_edge_ca_e3_2

```bash
./run.sh velox CADETS_E3 --database_host localhost --artifact_dir ./artifacts --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256 --project=PIDSHyp --exp=velox_ca_e3_runs
```

## Velox Edge CS 3
```bash title="Version 6"
./run.sh velox-edge CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=15.0 --project=PIDSHyp --exp=velox_edge_6_cs_e3_runs --force_restart construction
```

## Velox Edge CA 3
```bash title="Version 6"
./run.sh velox-edge CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=15.0 --project=PIDSHyp --exp=velox_edge_6_ca_e3 --force_restart construction
```

