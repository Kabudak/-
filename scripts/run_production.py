from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from utils.common import json_ready_args, set_seed, split_indices
from utils.metrics import binary_auc_from_scores


def binary_log_loss(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.clamp(1e-7, 1 - 1e-7)
    labels = labels.float()
    loss = -(labels * torch.log(scores) + (1 - labels) * torch.log(1 - scores))
    return float(loss.mean().item())


def binary_accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).long()
    return float((preds == labels.long()).float().mean().item())


def make_loader(dataset: TensorDataset, indices: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    grad_context = nullcontext() if is_train else torch.no_grad()
    with grad_context:
        for non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels in loader:
            non_seq_sparse = non_seq_sparse.to(device).long()
            non_seq_sparse_bag = non_seq_sparse_bag.to(device).long()
            non_seq_sparse_bag_mask = non_seq_sparse_bag_mask.to(device)
            non_seq_dense = non_seq_dense.to(device).float()
            seq_sparse = seq_sparse.to(device).long()
            seq_dense = seq_dense.to(device).float()
            seq_mask = seq_mask.to(device)
            labels = labels.to(device).float()

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
                if device.startswith("cuda")
                else nullcontext()
            )
            with autocast_context:
                logits = model(
                    non_seq_sparse,
                    non_seq_sparse_bag,
                    non_seq_sparse_bag_mask,
                    non_seq_dense,
                    seq_sparse,
                    seq_dense,
                    seq_mask,
                ).squeeze(-1)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_items += batch_size
            total_loss += loss.item() * batch_size
            all_scores.append(torch.sigmoid(logits.detach()).cpu())
            all_labels.append(labels.detach().cpu().long())

    if total_items == 0:
        return 0.0, float("nan"), float("nan"), float("nan")

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    auc = binary_auc_from_scores(scores, labels)
    log_loss = binary_log_loss(scores, labels)
    acc = binary_accuracy(scores, labels)
    return total_loss / total_items, auc, log_loss, acc


def load_tensor_bundle(data_dir: Path) -> tuple[TensorDataset, dict]:
    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    non_seq_sparse = torch.load(data_dir / "non_seq_sparse.pt", weights_only=True).long()
    labels = torch.load(data_dir / "labels.pt", weights_only=True).long()
    bag_path = data_dir / "non_seq_sparse_bag.pt"
    bag_mask_path = data_dir / "non_seq_sparse_bag_mask.pt"
    if bag_path.exists() and bag_mask_path.exists():
        non_seq_sparse_bag = torch.load(bag_path, weights_only=True).long()
        non_seq_sparse_bag_mask = torch.load(bag_mask_path, weights_only=True)
    else:
        non_seq_sparse_bag = torch.zeros(len(labels), 0, 1, dtype=torch.long)
        non_seq_sparse_bag_mask = torch.zeros(len(labels), 0, 1, dtype=torch.bool)

    tensors = (
        non_seq_sparse,
        non_seq_sparse_bag,
        non_seq_sparse_bag_mask,
        torch.load(data_dir / "non_seq_dense.pt", weights_only=True).float(),
        torch.load(data_dir / "seq_sparse.pt", weights_only=True).long(),
        torch.load(data_dir / "seq_dense.pt", weights_only=True).float(),
        torch.load(data_dir / "seq_mask.pt", weights_only=True),
        labels,
    )
    return TensorDataset(*tensors), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyFormer on preprocessed production tensors")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "production_sample")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "production")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-mode", choices=("none", "auto"), default="auto")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--field-embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--hyformer-layers", type=int, default=2)
    parser.add_argument("--seq-encoder-type", choices=("longer", "full_transformer", "swiglu", "identity"), default="longer")
    parser.add_argument("--short-seq-len", type=int, default=16)
    parser.add_argument("--num-queries-per-seq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast and GradScaler.")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, metadata = load_tensor_bundle(args.data_dir)
    train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed)

    train_loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(dataset, val_indices, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TAACHyFormerClassifier(
        sparse_field_cardinalities={key: int(value) for key, value in metadata["sparse_field_cardinalities"].items()},
        non_seq_sparse_fields=list(metadata["non_seq_sparse_fields"]),
        non_seq_sparse_bag_fields=list(metadata.get("non_seq_sparse_bag_fields", [])),
        non_seq_dense_fields=list(metadata["non_seq_dense_fields"]),
        seq_sparse_fields=list(metadata["seq_sparse_fields"]),
        seq_dense_fields=list(metadata["seq_dense_fields"]),
        token_groups=dict(metadata["token_groups"]),
        num_classes=1,
        seq_len=int(metadata["seq_len"]),
        num_sequences=int(metadata["num_sequences"]),
        num_non_seq_tokens=len(metadata["token_groups"]),
        num_queries_per_seq=args.num_queries_per_seq,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        hyformer_layers=args.hyformer_layers,
        seq_encoder_type=args.seq_encoder_type,
        short_seq_len=args.short_seq_len,
        field_embed_dim=args.field_embed_dim,
    ).to(args.device)

    train_labels = dataset.tensors[-1][train_indices].float()
    train_pos = int(train_labels.sum().item())
    train_neg = int(len(train_labels) - train_pos)
    pos_weight_value = train_neg / max(train_pos, 1)
    pos_weight = (
        torch.tensor([pos_weight_value], dtype=torch.float32, device=args.device)
        if args.pos_weight_mode == "auto"
        else None
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device.startswith("cuda"))

    print(f"device={args.device} amp={args.amp and args.device.startswith('cuda')}")
    print(
        f"samples={len(dataset)} train={len(train_indices)} val={len(val_indices)} "
        f"pos_rate={metadata.get('pos_rate')}"
    )
    print(
        f"sequences={metadata['sequence_names']} non_seq_tokens={len(metadata['token_groups'])} "
        f"queries_per_seq={args.num_queries_per_seq}"
    )
    print(
        f"d_model={args.d_model} layers={args.hyformer_layers} encoder={args.seq_encoder_type} "
        f"field_embed_dim={args.field_embed_dim}"
    )
    if pos_weight is not None:
        print(f"train_pos={train_pos} train_neg={train_neg} pos_weight={pos_weight.item():.4f}")

    best_val_auc = float("-inf")
    history = []
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, train_log_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=args.device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=args.amp and args.device.startswith("cuda"),
        )
        val_loss, val_auc, val_log_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=args.device,
            optimizer=None,
            scaler=None,
            amp_enabled=False,
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_auc": round(train_auc, 6) if not math.isnan(train_auc) else None,
            "train_log_loss": round(train_log_loss, 6) if not math.isnan(train_log_loss) else None,
            "train_accuracy": round(train_acc, 6) if not math.isnan(train_acc) else None,
            "val_loss": round(val_loss, 6),
            "val_auc": round(val_auc, 6) if not math.isnan(val_auc) else None,
            "val_log_loss": round(val_log_loss, 6) if not math.isnan(val_log_loss) else None,
            "val_accuracy": round(val_acc, 6) if not math.isnan(val_acc) else None,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_auc={train_auc:.4f} train_loss={train_loss:.4f} "
            f"val_auc={val_auc:.4f} val_loss={val_loss:.4f} val_logloss={val_log_loss:.4f}"
        )

    run_metadata = {
        **metadata,
        "args": json_ready_args(args),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_positive_samples": train_pos,
        "train_negative_samples": train_neg,
        "training_history": history,
        "best_val_auc": round(best_val_auc, 6) if best_val_auc != float("-inf") else None,
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight_mode": args.pos_weight_mode,
            "pos_weight_value": round(pos_weight_value, 6) if pos_weight is not None else None,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.save_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auc_tag = "nan" if best_val_auc == float("-inf") else f"{best_val_auc:.4f}"
        ckpt_path = args.output_dir / f"hyformer_production_{timestamp}_val_auc{auc_tag}.pt"
        torch.save(
            {
                "model": best_state if best_state is not None else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "metadata": run_metadata,
            },
            ckpt_path,
        )
        print(f"checkpoint: {ckpt_path}")

    print(f"best_val_auc={best_val_auc:.4f}" if best_val_auc != float("-inf") else "best_val_auc=nan")


if __name__ == "__main__":
    main()
