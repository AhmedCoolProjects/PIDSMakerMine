"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation.
"""

import math
import os
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

import networkx as nx
import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    get_split_to_files,
    init_database_connection,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    stringtomd5,
)


def compute_indexid2msg(cfg):
    """Compute mapping from node index IDs to node types and feature labels.

    Queries PostgreSQL database for all nodes (netflow, subject/process, file) and
    extracts their attributes to create feature labels based on configuration.

    Args:
        cfg: Configuration with database connection and feature settings

    Returns:
        dict: Mapping {index_id: [node_type, label_string]} where:
            - index_id: Database node identifier
            - node_type: One of 'netflow', 'subject', 'file'
            - label_string: Feature label (hashed or plaintext depending on config)
    """
    cur, connect = init_database_connection(cfg)

    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    indexid2msg = {}

    def get_label_str_from_features(attrs, node_type):
        """Extract feature label from node attributes based on configured features.

        Args:
            attrs: Dictionary of node attributes
            node_type: Type of node ('netflow', 'subject', 'file')

        Returns:
            str: Space-separated feature string, optionally hashed
        """
        label_str = " ".join([attrs[label_used] for label_used in node_label_features[node_type]])
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of netflow nodes: {len(records)}")

    for i in records:
        attrs = {
            "type": "netflow",
            "local_ip": str(i[2]),
            "local_port": str(i[3]),
            "remote_ip": str(i[4]),
            "remote_port": str(i[5]),
        }
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of process nodes: {len(records)}")

    for i in records:
        attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of file nodes: {len(records)}")

    for i in records:
        attrs = {"type": "file", "path": str(i[2])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]

    return indexid2msg  # {index_id: [node_type, msg]}


def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """Save filtered node index-to-feature mapping to disk.

    Filters out nodes not used in any train/val/test graphs (due to excluded edge types)
    before saving to avoid downstream errors during featurization.

    Note: Must be called after graph construction to ensure only used nodes are saved.

    Args:
        indexid2msg: Full node mapping from compute_indexid2msg()
        split2nodes: Mapping of splits to their node sets
        cfg: Configuration with output directory path
    """
    all_nodes = set().union(*(split2nodes[split] for split in ["train", "val", "test"]))
    indexid2msg = {k: v for k, v in indexid2msg.items() if k in all_nodes}

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg):
    """Compute and save mapping of dataset splits to their node sets.

    Loads all graphs from train/val/test splits and collects unique node IDs
    appearing in each split. Used to filter node features and track split membership.

    Args:
        cfg: Configuration with graph directory and split file paths

    Returns:
        dict: Mapping of split names to node sets:
            {'train': {node_ids}, 'val': {node_ids}, 'test': {node_ids}}
    """
    split_to_files = get_split_to_files(cfg, cfg.construction._graphs_dir)
    split2nodes = defaultdict(set)

    for split, files in split_to_files.items():
        graph_list = [torch.load(path) for path in files]
        for G in log_tqdm(graph_list, desc=f"Check nodes in {split} set"):
            for node in G.nodes():
                split2nodes[split].add(node)
    split2nodes = dict(split2nodes)

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


_HEX_CHARS = frozenset("0123456789abcdefABCDEF")

_SENSITIVE_PREFIXES = frozenset([
    "/etc/", "/dev/", "/sys/", "/proc/", "/bin/", "/sbin/",
    "/lib/", "/usr/", "/boot/",
    "/system/", "/data/system/",  # Android (CLEARSCOPE E3)
])

_NETWORK_OPS = frozenset([
    "EVENT_SENDTO", "EVENT_SENDMSG", "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_CONNECT",
])


def _get_dir_prefix(path: str) -> str:
    """Return the first two path components as a directory-level key."""
    if not path:
        return "/"
    parts = path.split("/")
    if len(parts) >= 3:
        return "/" + parts[1] + "/" + parts[2]
    elif len(parts) >= 2:
        return "/" + parts[1]
    return "/"


def _is_sensitive_path_fn(path: str) -> float:
    """1.0 if path starts with a known sensitive system prefix (Linux or Android)."""
    for prefix in _SENSITIVE_PREFIXES:
        if path.startswith(prefix):
            return 1.0
    return 0.0


def _is_hash_filename(label: str) -> float:
    """1.0 if the last path segment looks like a hex hash (len >= 32, all hex chars).

    Targets CLEARSCOPE E3: attack cache entries are 40-char SHA-1 hex strings.
    """
    if not label:
        return 0.0
    filename = label.rstrip("/").rsplit("/", 1)[-1]
    if len(filename) >= 32 and all(c in _HEX_CHARS for c in filename):
        return 1.0
    return 0.0


def _is_server_port_outbound(dst_node_type: str, dst_label: str, op: str) -> float:
    """1.0 if this is an outbound event from a privileged local port (<= 1024).

    Targets CADETS E3: nginx:80 initiating outbound connections is anomalous.
    Netflow label format: 'local_ip:local_port->remote_ip:remote_port'
    """
    if dst_node_type != "netflow" or op not in ("EVENT_CONNECT", "EVENT_SENDTO", "EVENT_SENDMSG"):
        return 0.0
    try:
        arrow_idx = dst_label.find("->")
        if arrow_idx == -1:
            return 0.0
        local_part = dst_label[:arrow_idx]
        local_port = int(local_part.rsplit(":", 1)[-1])
        return 1.0 if local_port <= 1024 else 0.0
    except (ValueError, IndexError):
        return 0.0


def _write_then_execute_flag(op: str, dst, t: int, last_write_time: dict, window_ns: int = 300_000_000_000) -> float:
    """1.0 if this is EVENT_EXECUTE on a dst that was written to within the last 300s.

    Targets CADETS E3: malware written to /tmp then immediately executed.
    """
    if op != "EVENT_EXECUTE":
        return 0.0
    last_write = last_write_time.get(dst)
    if last_write is None:
        return 0.0
    return 1.0 if (t - last_write) <= window_ns else 0.0


def gen_edge_fused_tw(indexid2msg, cfg):
    """Generate time-windowed provenance graphs from database events.

    Main graph construction function that:
    1. Queries database for events in time windows
    2. Optionally fuses consecutive edges of same type between node pairs
    3. Optionally adds attack mimicry events for data augmentation
    4. Builds NetworkX MultiDiGraphs with node attributes and edge metadata
    5. Saves graphs to disk organized by day and time window

    Args:
        indexid2msg: Node index to [type, label] mapping from compute_indexid2msg()
        cfg: Configuration with:
            - Database connection settings
            - Time window parameters (size, dates)
            - Edge type filtering (rel2id)
            - Mimicry settings (mimicry_edge_num)
            - Output directory paths
    """
    cur, connect = init_database_connection(cfg)
    rel2id = get_rel2id(cfg)
    include_edge_type = rel2id

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        attack_mimicry_events = mimicry.gen_mimicry_edges(cfg)
    else:
        attack_mimicry_events = defaultdict(list)

    def get_batches(arr, batch_size):
        """Yield consecutive batches of specified size from array.

        Args:
            arr: Input array to batch
            batch_size: Number of elements per batch

        Yields:
            list: Batches of size batch_size (last batch may be smaller)
        """
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]

    # In test mode, we ensure to get 1 TW in each set
    dates = get_dates_from_cfg(cfg)

    log("Building graphs...")
    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start)
            end_ns_timestamp = datetime_to_ns_time_US(stop)

            attack_index = 0
            mimicry_events = []
            for attack_tuple in cfg.dataset.attack_to_time_window:
                attack = attack_tuple[0]
                attack_start_time = datetime_to_ns_time_US(attack_tuple[1])
                attack_end_time = datetime_to_ns_time_US(attack_tuple[2])

                if mimicry_edge_num > 0 and (
                    attack_start_time >= start_ns_timestamp and attack_end_time <= end_ns_timestamp
                ):
                    log(
                        f"Insert mimicry events into attack {attack_index} when building graphs from {date_start} to {date_stop}"
                    )
                    mimicry_events.extend(attack_mimicry_events[attack_index])
                attack_index += 1

            sql = """
            select * from event_table
            where
                  timestamp_rec>'%s' and timestamp_rec<'%s'
                   ORDER BY timestamp_rec, event_uuid;
            """ % (start_ns_timestamp, end_ns_timestamp)
            cur.execute(sql)
            events = cur.fetchall()

            if len(events) == 0:
                continue

            events_list = []
            for (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in events:
                if operation in include_edge_type:
                    event_tuple = (
                        src_node,
                        src_index_id,
                        operation,
                        dst_node,
                        dst_index_id,
                        event_uuid,
                        timestamp_rec,
                        _id,
                    )
                    events_list.append(event_tuple)

            for (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in mimicry_events:
                if operation in include_edge_type:
                    event_tuple = (
                        src_node,
                        src_index_id,
                        operation,
                        dst_node,
                        dst_index_id,
                        event_uuid,
                        timestamp_rec,
                        _id,
                    )
                    events_list.append(event_tuple)

            start_time = events_list[0][-2]
            temp_list = []
            BATCH = 1024
            window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

            last_batch = False
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
                    last_batch = True

                if (batch_edges[-1][-2] > start_time + window_size_in_ns) or last_batch:
                    time_interval = (
                        ns_time_to_datetime_US(start_time)
                        + "~"
                        + ns_time_to_datetime_US(batch_edges[-1][-2])
                    )

                    # log(f"Start create edge fused time window graph for {time_interval}")

                    node_info = {}
                    raw_edges = []

                    # Step 1: Build raw (unfused) edge list from temp_list
                    for (
                        src_node,
                        src_index_id,
                        operation,
                        dst_node,
                        dst_index_id,
                        event_uuid,
                        timestamp_rec,
                        _id,
                    ) in temp_list:
                        if src_index_id not in node_info:
                            node_type, label = indexid2msg[src_index_id]
                            node_info[src_index_id] = {
                                "label": label,
                                "node_type": node_type,
                            }
                        if dst_index_id not in node_info:
                            node_type, label = indexid2msg[dst_index_id]
                            node_info[dst_index_id] = {
                                "label": label,
                                "node_type": node_type,
                            }

                        raw_edges.append({
                            "src": src_index_id,
                            "dst": dst_index_id,
                            "time": timestamp_rec,
                            "label": operation,
                            "event_uuid": event_uuid,
                        })

                    # log(f"Start creating graph for {time_interval}")
                    graph = nx.MultiDiGraph()

                    for node, info in node_info.items():
                        graph.add_node(node, node_type=info["node_type"], label=info["label"])

                    # Step 2: Compute causal temporal features on RAW (unfused) edges
                    temporal_cfg = cfg.construction.get("temporal_features", {})
                    if temporal_cfg.get("enabled", False):
                        raw_edges.sort(key=lambda e: e["time"])
                        window_size_ns = max(batch_edges[-1][-2] - start_time, 1)
                        feature_names = temporal_cfg.get("feature_names", None)

                        # State for existing features
                        last_time = None
                        last_pair_time = {}
                        last_src_time = {}
                        pair_count = Counter()
                        type_count = Counter()
                        max_type_count = 0
                        src_uniq_types = defaultdict(set)
                        src_unique_dsts = defaultdict(set)
                        pair_types = defaultdict(set)

                        # State for new features
                        last_dst_time = {}
                        dst_unique_srcs = defaultdict(set)
                        seen_dst_labels = set()
                        last_write_time = {}
                        src_recent_new_pairs = defaultdict(lambda: deque(maxlen=50))

                        # State for v2 semantic groups
                        max_pair_count_so_far = 0
                        dst_type_count = defaultdict(Counter)
                        src_netflow_count = Counter()
                        src_file_write_count = Counter()
                        src_total_count = Counter()
                        dst_types_seen = defaultdict(set)
                        src_dirs = defaultdict(set)
                        src_last_op = {}
                        last_op_same_pair = {}
                        op_transition_count = Counter()
                        max_transition_count = 0

                        for e_idx, e in enumerate(raw_edges):
                            t = e["time"]
                            src, dst = e["src"], e["dst"]
                            op = e["label"]
                            op_id = rel2id.get(op, 0)
                            normalizer = max(e_idx, 1)

                            dst_label = node_info[dst]["label"]
                            dst_node_type = node_info[dst]["node_type"]

                            max_type_count = max(max_type_count, max(type_count.values(), default=0))
                            type_rarity = 1 - (type_count.get(op_id, 0) / max(max_type_count, 1))
                            is_new_pair_val = 0.0 if (src, dst) in last_pair_time else 1.0

                            # Pre-compute for Group A4 (dst_op_rarity)
                            _dst_max_tc = max(dst_type_count[dst].values(), default=0)
                            # Pre-compute for Group F (op transitions)
                            _prev_op_src = src_last_op.get(src)
                            _prev_op_pair = last_op_same_pair.get((src, dst))
                            _transition_key = (_prev_op_src, op)

                            all_feat_vals = {
                                # --- existing 10 features (kept for backward compat) ---
                                "t_norm": (t - start_time) / window_size_ns,
                                "log_delta_prev": math.log1p((t - last_time) / 1e9) if last_time is not None else 0.0,
                                # sentinel fix: new pairs get max-window gap, not 0
                                "log_delta_same_pair": (
                                    math.log1p(window_size_ns / 1e9)
                                    if (src, dst) not in last_pair_time
                                    else math.log1p((t - last_pair_time[(src, dst)]) / 1e9)
                                ),
                                "log_delta_src_activity": math.log1p((t - last_src_time.get(src, t)) / 1e9),
                                "pair_freq_causal": pair_count[(src, dst)] / normalizer,
                                "src_unique_dsts_norm": len(src_unique_dsts[src]) / normalizer,
                                "pair_type_count_norm": len(pair_types[(src, dst)]) / normalizer,
                                "type_rarity": type_rarity,
                                "is_new_pair": is_new_pair_val,
                                "is_new_type_for_src": 0.0 if op_id in src_uniq_types[src] else 1.0,
                                # --- existing extended features ---
                                "is_tmp_dst": 1.0 if "/tmp/" in dst_label else 0.0,
                                "is_hash_filename_dst": _is_hash_filename(dst_label),
                                "is_server_port_outbound": _is_server_port_outbound(dst_node_type, dst_label, op),
                                "is_new_path_global": 0.0 if dst_label in seen_dst_labels else 1.0,
                                "is_new_type_for_pair": 0.0 if op_id in pair_types[(src, dst)] else 1.0,
                                "is_new_src_for_dst": 0.0 if src in dst_unique_srcs[dst] else 1.0,
                                "write_then_execute_flag": _write_then_execute_flag(op, dst, t, last_write_time),
                                "log_delta_dst_activity": math.log1p((t - last_dst_time.get(dst, t)) / 1e9),
                                "src_burst_new_pairs": sum(src_recent_new_pairs[src]) / 50.0,
                                "is_execute_from_tmp": 1.0 if op == "EVENT_EXECUTE" and "/tmp/" in dst_label else 0.0,
                                # --- v2: revised V1 (fixed normalizers) ---
                                "pair_freq_revised": (
                                    math.log1p(pair_count[(src, dst)])
                                    / math.log1p(max(max_pair_count_so_far, 1))
                                ),
                                "src_unique_dsts_log": math.log1p(len(src_unique_dsts[src])),
                                # --- v2 Group A: dst-perspective ---
                                "dst_unique_srcs_log": math.log1p(len(dst_unique_srcs[dst])),
                                "dst_op_rarity": 1 - dst_type_count[dst][op_id] / max(_dst_max_tc, 1),
                                # --- v2 Group B: node-type-aware behavioral ratios ---
                                "is_network_op": 1.0 if op in _NETWORK_OPS else 0.0,
                                "src_netflow_ratio": src_netflow_count[src] / max(src_total_count[src], 1),
                                "src_file_write_ratio": src_file_write_count[src] / max(src_total_count[src], 1),
                                "src_unique_dst_types": len(dst_types_seen[src]) / 3.0,
                                # --- v2 Group C: file path semantics ---
                                "dst_path_depth": (
                                    (len(dst_label.split("/")) - 1) / 10.0
                                    if dst_node_type == "file" else 0.0
                                ),
                                "is_sensitive_path": (
                                    _is_sensitive_path_fn(dst_label)
                                    if dst_node_type == "file" else 0.0
                                ),
                                "src_unique_dirs": math.log1p(len(src_dirs[src])),
                                "is_new_dir_for_src": (
                                    1.0 if _get_dir_prefix(dst_label) not in src_dirs[src] else 0.0
                                ) if dst_node_type == "file" else 0.0,
                                # --- v2 Group F: op transition sequences ---
                                "prev_op_id_same_pair": (
                                    (rel2id.get(_prev_op_pair, 0) + 1) / (len(rel2id) + 1)
                                    if _prev_op_pair is not None else 0.0
                                ),
                                "op_transition_rarity": (
                                    1 - op_transition_count[_transition_key] / max(max_transition_count, 1)
                                ),
                                "is_recv_write_pattern": (
                                    1.0 if _prev_op_src in {"EVENT_RECVFROM", "EVENT_RECVMSG"}
                                    and op == "EVENT_WRITE" else 0.0
                                ),
                                "is_open_then_write": (
                                    1.0 if _prev_op_pair == "EVENT_OPEN"
                                    and op == "EVENT_WRITE" else 0.0
                                ),
                            }

                            if feature_names is not None:
                                feats = [all_feat_vals[name] for name in feature_names]
                            else:
                                # Legacy order: original 10 features
                                feats = [
                                    all_feat_vals["t_norm"],
                                    all_feat_vals["log_delta_prev"],
                                    all_feat_vals["log_delta_same_pair"],
                                    all_feat_vals["log_delta_src_activity"],
                                    all_feat_vals["pair_freq_causal"],
                                    all_feat_vals["src_unique_dsts_norm"],
                                    all_feat_vals["pair_type_count_norm"],
                                    all_feat_vals["type_rarity"],
                                    all_feat_vals["is_new_pair"],
                                    all_feat_vals["is_new_type_for_src"],
                                ]

                            e["temporal_feats"] = feats

                            # State updates (always, regardless of feature_names selection)
                            last_time = t
                            last_src_time[src] = t
                            last_dst_time[dst] = t
                            last_pair_time[(src, dst)] = t
                            pair_count[(src, dst)] += 1
                            type_count[op_id] += 1
                            src_uniq_types[src].add(op_id)
                            src_unique_dsts[src].add(dst)
                            dst_unique_srcs[dst].add(src)
                            pair_types[(src, dst)].add(op_id)
                            seen_dst_labels.add(dst_label)
                            src_recent_new_pairs[src].append(is_new_pair_val)
                            if op == "EVENT_WRITE":
                                last_write_time[dst] = t

                            # v2 state updates
                            max_pair_count_so_far = max(max_pair_count_so_far, pair_count[(src, dst)])
                            dst_type_count[dst][op_id] += 1
                            src_total_count[src] += 1
                            if dst_node_type == "netflow":
                                src_netflow_count[src] += 1
                            if dst_node_type == "file" and op == "EVENT_WRITE":
                                src_file_write_count[src] += 1
                            dst_types_seen[src].add(dst_node_type)
                            if dst_node_type == "file":
                                src_dirs[src].add(_get_dir_prefix(dst_label))
                            op_transition_count[_transition_key] += 1
                            max_transition_count = max(max_transition_count, op_transition_count[_transition_key])
                            src_last_op[src] = op
                            last_op_same_pair[(src, dst)] = op

                    # Step 3: Optionally fuse consecutive same-type edges, carry over features
                    if cfg.construction.fuse_edge:
                        edge_list = []
                        edge_groups = defaultdict(list)
                        for e in raw_edges:
                            edge_groups[(e["src"], e["dst"])].append(e)
                        for (src, dst), group in edge_groups.items():
                            group.sort(key=lambda x: x["time"])
                            current_type = None
                            for e in group:
                                if e["label"] != current_type:
                                    edge_list.append(e)
                                    current_type = e["label"]
                    else:
                        edge_list = raw_edges

                    for i, edge in enumerate(edge_list):
                        edge_attrs = {
                            "event_uuid": edge["event_uuid"],
                            "time": edge["time"],
                            "label": edge["label"],
                            "y": 0,
                        }
                        if "temporal_feats" in edge:
                            edge_attrs["temporal_feats"] = edge["temporal_feats"]
                        graph.add_edge(
                            edge["src"],
                            edge["dst"],
                            **edge_attrs,
                        )

                        # For unit tests, we only want few edges
                        NUM_TEST_EDGES = 2000
                        if cfg._test_mode and i >= NUM_TEST_EDGES:
                            break

                    date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    # log(f"Saving graph for {time_interval}")
                    torch.save(graph, graph_name)

                    # log(f"[{time_interval}] Num of edges: {len(edge_list)}")
                    # log(f"[{time_interval}] Num of events: {len(temp_list)}")
                    # log(f"[{time_interval}] Num of nodes: {len(node_info.keys())}")
                    start_time = batch_edges[-1][-2]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break


def main(cfg):
    """Main construction pipeline: build graphs from database and save metadata.

    Execution flow:
    1. Extract node features from database (compute_indexid2msg)
    2. Build time-windowed graphs from events (gen_edge_fused_tw)
    3. Compute dataset split node memberships (compute_and_save_split2nodes)
    4. Save filtered node features (save_indexid2msg)

    Args:
        cfg: Configuration object with all construction parameters
    """
    log_start(__file__)

    indexid2msg = compute_indexid2msg(cfg=cfg)

    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg)

    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)
