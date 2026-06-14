from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from utils.common import json_ready_args, set_seed, split_indices

TZ_OFFSET_SEC = 8 * 3600


def binary_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        probs = torch.sigmoid(logits).numpy()
        y = labels.numpy()
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, probs))
    except ImportError:
        return _simple_auc(torch.sigmoid(logits), labels)


def _simple_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.flatten()
    labels = labels.flatten()
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_indices]

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0
    for idx in range(len(sorted_labels)):
        if sorted_labels[idx].item() == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr
    return auc


def log_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    loss = -(labels.float() * torch.log(probs) + (1 - labels.float()) * torch.log(1 - probs))
    return float(loss.mean().item())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return float((preds == labels).float().mean().item())


def time_split_indices(timestamps: torch.Tensor, val_days: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    timestamps_np = timestamps.cpu().numpy().astype(np.int64)
    day_ids = (timestamps_np + TZ_OFFSET_SEC) // 86400
    unique_days = np.unique(day_ids)
    if len(unique_days) < 2:
        raise ValueError("")
    val_days = max(1, min(val_days, len(unique_days) - 1))
    val_day_set = set(unique_days[-val_days:].tolist())
    train_mask = np.array([day not in val_day_set for day in day_ids], dtype=bool)
    val_mask = ~train_mask

    train_idx = torch.from_numpy(np.nonzero(train_mask)[0]).long()
    val_idx = torch.from_numpy(np.nonzero(val_mask)[0]).long()
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("")

    split_meta = {
        "split_mode": "time",
        "timezone": "Asia",
        "val_days": val_days,
        "train_min_timestamp": int(timestamps[train_idx].min().item()),
        "train_max_timestamp": int(timestamps[train_idx].max().item()),
        "val_min_timestamp": int(timestamps[val_idx].min().item()),
        "val_max_timestamp": int(timestamps[val_idx].max().item()),
    }
    return train_idx, val_idx, split_meta


def random_split_with_metadata(
    size: int,
    timestamps: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    train_idx, val_idx = split_indices(size, val_ratio, seed)
    split_meta = {
        "split_mode": "random",
        "val_ratio": val_ratio,
        "train_min_timestamp": int(timestamps[train_idx].min().item()) if len(train_idx) else 0,
        "train_max_timestamp": int(timestamps[train_idx].max().item()) if len(train_idx) else 0,
        "val_min_timestamp": int(timestamps[val_idx].min().item()) if len(val_idx) else 0,
        "val_max_timestamp": int(timestamps[val_idx].max().item()) if len(val_idx) else 0,
    }
    return train_idx, val_idx, split_meta


def select_evenly_spaced_indices(size: int, max_rows: int | None) -> torch.Tensor | None:
    if max_rows is None or max_rows >= size:
        return None
    indices = np.linspace(0, size - 1, num=max_rows, dtype=np.int64)
    indices = np.unique(indices)
    return torch.from_numpy(indices).long()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels in loader:
        non_seq_sparse = non_seq_sparse.to(device)
        non_seq_dense = non_seq_dense.to(device)
        seq_sparse = seq_sparse.to(device)
        seq_dense = seq_dense.to(device)
        seq_mask = seq_mask.to(device)
        labels = labels.to(device).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask).squeeze(-1)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu().long())

    if total_items == 0:
        return 0.0, float("nan"), float("nan"), 0.0

    concat_logits = torch.cat(all_logits)
    concat_labels = torch.cat(all_labels)
    auc = binary_auc_from_logits(concat_logits, concat_labels)
    ll = log_loss_from_logits(concat_logits, concat_labels)
    acc = accuracy_from_logits(concat_logits, concat_labels)
    return total_loss / total_items, auc, ll, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--num-queries-per-seq", type=int, default=8)
    parser.add_argument("--num-non-seq-tokens", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--field-embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=256)
    parser.add_argument("--hyformer-layers", type=int, default=4)
    parser.add_argument("--seq-encoder-type", choices=("longer", "full_transformer", "swiglu", "identity"), default="identity")
    parser.add_argument("--short-seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-mode", choices=("none", "auto"), default="none")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=("time", "random"), default="time")
    parser.add_argument("--val-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ctr")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("...")
    non_seq_sparse = torch.load(args.data_dir / "non_seq_sparse.pt", weights_only=True).long()
    non_seq_dense = torch.load(args.data_dir / "non_seq_dense.pt", weights_only=True)
    seq_sparse = torch.load(args.data_dir / "seq_sparse.pt", weights_only=True).long()
    seq_dense = torch.load(args.data_dir / "seq_dense.pt", weights_only=True).float()
    seq_mask = torch.load(args.data_dir / "seq_mask.pt", weights_only=True)
    labels = torch.load(args.data_dir / "labels.pt", weights_only=True).long()
    timestamps = torch.load(args.data_dir / "timestamps.pt", weights_only=True).long()

    with open(args.data_dir / "metadata.json", encoding="utf-8") as f:
        data_meta = json.load(f)

    sampled_indices = select_evenly_spaced_indices(len(labels), args.max_rows)
    if sampled_indices is not None:
        non_seq_sparse = non_seq_sparse[sampled_indices]
        non_seq_dense = non_seq_dense[sampled_indices]
        seq_sparse = seq_sparse[sampled_indices]
        seq_dense = seq_dense[sampled_indices]
        seq_mask = seq_mask[sampled_indices]
        labels = labels[sampled_indices]
        timestamps = timestamps[sampled_indices]

    if seq_sparse.size(1) != args.num_sequences:
        raise ValueError(
            f" num_sequences={args.num_sequences}  {seq_sparse.size(1)}。"
        )
    if seq_sparse.size(2) != args.seq_len:
        print(f"seq_len={args.seq_len}， seq_len={seq_sparse.size(2)}")
    expected_non_seq_tokens = len(data_meta["token_groups"])
    if args.num_non_seq_tokens != expected_non_seq_tokens:
        raise ValueError(
            f"num_non_seq_tokens={args.num_non_seq_tokens} "
            f"{expected_non_seq_tokens} "
        )

    if args.split_mode == "time":
        train_idx, val_idx, split_meta = time_split_indices(timestamps, args.val_days)
    else:
        train_idx, val_idx, split_meta = random_split_with_metadata(len(labels), timestamps, args.val_ratio, args.seed)

    train_dataset = TensorDataset(
        non_seq_sparse[train_idx],
        non_seq_dense[train_idx],
        seq_sparse[train_idx],
        seq_dense[train_idx],
        seq_mask[train_idx],
        labels[train_idx],
    )
    val_dataset = TensorDataset(
        non_seq_sparse[val_idx],
        non_seq_dense[val_idx],
        seq_sparse[val_idx],
        seq_dense[val_idx],
        seq_mask[val_idx],
        labels[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TAACHyFormerClassifier(
        sparse_field_cardinalities={key: int(value) for key, value in data_meta["sparse_field_cardinalities"].items()},
        non_seq_sparse_fields=list(data_meta["non_seq_sparse_fields"]),
        non_seq_dense_fields=list(data_meta["non_seq_dense_fields"]),
        seq_sparse_fields=list(data_meta["seq_sparse_fields"]),
        seq_dense_fields=list(data_meta["seq_dense_fields"]),
        token_groups=dict(data_meta["token_groups"]),
        num_classes=1,
        seq_len=int(data_meta["seq_len"]),
        num_sequences=int(data_meta["num_sequences"]),
        num_queries_per_seq=args.num_queries_per_seq,
        num_non_seq_tokens=args.num_non_seq_tokens,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        hyformer_layers=args.hyformer_layers,
        seq_encoder_type=args.seq_encoder_type,
        short_seq_len=args.short_seq_len,
        field_embed_dim=args.field_embed_dim,
    ).to(args.device)

    train_labels = labels[train_idx]
    pos_count = int(train_labels.sum().item())
    neg_count = int(len(train_labels) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = (
        torch.tensor([pos_weight_value], dtype=torch.float32, device=args.device)
        if args.pos_weight_mode == "auto"
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device={args.device}  samples={len(labels)}")
    print(f"  train={len(train_dataset)}  val={len(val_dataset)}")
    print(f"  non_seq_sparse={tuple(non_seq_sparse.shape)}")
    print(f"  non_seq_dense={tuple(non_seq_dense.shape)}")
    print(f"  seq_sparse={tuple(seq_sparse.shape)}")
    print(f"  seq_dense={tuple(seq_dense.shape)}")
    print(f"  seq_mask={tuple(seq_mask.shape)}")
    if pos_weight is None:
        print(f"  train_pos_rate={pos_count / max(len(train_labels), 1) * 100:.2f}%  pos_weight=none")
    else:
        print(f"  train_pos_rate={pos_count / max(len(train_labels), 1) * 100:.2f}%  pos_weight={pos_weight.item():.2f}")
    print(
        f"  d_model={args.d_model}  field_embed_dim={args.field_embed_dim} "
        f"layers={args.hyformer_layers}  encoder={args.seq_encoder_type}"
    )

    best_val_auc = float("-inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    args_payload = json_ready_args(args)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, _, train_acc = run_epoch(model, train_loader, criterion, args.device, optimizer)
        val_loss, val_auc, val_ll, val_acc = run_epoch(model, val_loader, criterion, args.device)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_logloss={val_ll:.4f} val_acc={val_acc:.4f}"
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    neg_sampling_info = data_meta.get("neg_sampling")
    calibration_note = None
    if neg_sampling_info and neg_sampling_info.get("neg_ratio", 0) > 0:
        neg_ratio = neg_sampling_info["neg_ratio"]
        calibration_note = (
            f"Predicted probabilities are biased high due to 1:{neg_ratio} negative sampling. "
            f"To calibrate: p_cal = p / (p + (1 - p) / {neg_ratio})"
        )

    run_meta = {
        **data_meta,
        "args": args_payload,
        "split": {
            **split_meta,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "train_pos_rate": round(pos_count / max(len(train_labels), 1), 6),
            "val_pos_rate": round(float(labels[val_idx].float().mean().item()), 6),
        },
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight_mode": args.pos_weight_mode,
            "pos_weight_value": round(pos_weight_value, 6) if pos_weight is not None else None,
        },
        "calibration": calibration_note,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.save_checkpoint and best_model_state is not None:
        ckpt_path = args.output_dir / f"best_model_epoch{best_epoch:02d}_auc{best_val_auc:.4f}.pt"
        torch.save({"model": best_model_state, "metadata": run_meta}, ckpt_path)
        print(f" best checkpoint: {ckpt_path}")

    print(f"\n best_val_auc={best_val_auc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
