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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from utils.common import json_ready_args, set_seed
from utils.metrics import binary_auc_from_scores


def binary_log_loss(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.clamp(1e-7, 1 - 1e-7)
    labels = labels.float()
    loss = -(labels * torch.log(scores) + (1 - labels) * torch.log(1 - scores))
    return float(loss.mean().item())


def binary_accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).long()
    return float((preds == labels.long()).float().mean().item())


def compute_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute AUC, log_loss, accuracy from probability scores and labels."""
    auc = binary_auc_from_scores(scores, labels)
    ll = binary_log_loss(scores, labels)
    acc = binary_accuracy(scores, labels)
    return {"auc": auc, "log_loss": ll, "accuracy": acc}


def move_batch_to_device(batch: tuple, device: str):
    """Move a batch of tensors to device, return unpacked tensors."""
    non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels = batch
    return (
        non_seq_sparse.to(device).long(),
        non_seq_sparse_bag.to(device).long(),
        non_seq_sparse_bag_mask.to(device),
        non_seq_dense.to(device).float(),
        seq_sparse.to(device).long(),
        seq_dense.to(device).float(),
        seq_mask.to(device),
        labels.to(device).float(),
    )


def forward_pass(
    model: nn.Module,
    batch_tensors: tuple,
    criterion: nn.Module,
    device: str,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass on a single batch. Returns (loss, logits)."""
    non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels = batch_tensors

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
        if device.startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
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
    return loss, logits


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
    parser = argparse.ArgumentParser(description="Train HyFormer on production tensors with progressive per-batch validation")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "production_sample")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "production")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (progressive eval is most meaningful with 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-mode", choices=("none", "auto"), default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--field-embed-dim", type=int, default=64)
    parser.add_argument("--token-mlp-hidden", type=int, default=320)
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
    amp_enabled = args.amp and args.device.startswith("cuda")

    dataset, metadata = load_tensor_bundle(args.data_dir)

    # Single DataLoader, shuffle=False to preserve order for progressive evaluation
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    num_batches = len(loader)
    num_samples = len(dataset)
    last_batch_size = num_samples - (num_batches - 1) * args.batch_size
    train_sample_count = num_samples - last_batch_size

    # Build model
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
        token_mlp_hidden=args.token_mlp_hidden,
        sequence_fields=dict(metadata.get("sequence_fields", {})) or None,
        sequence_names=list(metadata.get("sequence_names", [])) or None,
    ).to(args.device)

    # Compute pos_weight from training data only (exclude last batch)
    train_labels = dataset.tensors[-1][:train_sample_count].float()
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Print config
    print(f"device={args.device} amp={amp_enabled}")
    print(f"samples={num_samples} train_batches={num_batches - 1} test_batch=1 "
          f"train_samples={train_sample_count} test_samples={last_batch_size} "
          f"pos_rate={metadata.get('pos_rate')}")
    print(
        f"sequences={metadata.get('sequence_names')} non_seq_tokens={len(metadata['token_groups'])} "
        f"queries_per_seq={args.num_queries_per_seq}"
    )
    print(
        f"d_model={args.d_model} layers={args.hyformer_layers} encoder={args.seq_encoder_type} "
        f"field_embed_dim={args.field_embed_dim} token_mlp_hidden={args.token_mlp_hidden}"
    )
    if pos_weight is not None:
        print(f"train_pos={train_pos} train_neg={train_neg} pos_weight={pos_weight.item():.4f}")

    best_test_auc = float("-inf")
    all_progressive_results = []

    for epoch in range(1, args.epochs + 1):
        epoch_results = []

        for batch_idx, batch in enumerate(loader):
            is_last_batch = batch_idx == num_batches - 1
            batch_tensors = move_batch_to_device(batch, args.device)
            batch_n = batch_tensors[-1].size(0)

            # === Step 1: EVALUATE with frozen model ===
            model.eval()
            with torch.no_grad():
                eval_loss, eval_logits = forward_pass(model, batch_tensors, criterion, args.device, amp_enabled)

            eval_scores = torch.sigmoid(eval_logits).cpu()
            eval_labels = batch_tensors[-1].detach().cpu().long()
            eval_m = compute_metrics(eval_scores, eval_labels)

            phase_name = "test" if is_last_batch else "pre_train_eval"
            epoch_results.append({
                "epoch": epoch,
                "batch_index": batch_idx,
                "phase": phase_name,
                "loss": round(float(eval_loss.item()), 6),
                "auc": round(eval_m["auc"], 6) if not math.isnan(eval_m["auc"]) else None,
                "log_loss": round(eval_m["log_loss"], 6) if not math.isnan(eval_m["log_loss"]) else None,
                "accuracy": round(eval_m["accuracy"], 6),
                "num_samples": batch_n,
            })

            phase_tag = "TEST" if is_last_batch else "EVAL"
            print(
                f"[Epoch {epoch} Batch {batch_idx:03d}/{num_batches - 1}] {phase_tag:5s}  "
                f"auc={eval_m['auc']:.4f}  loss={float(eval_loss.item()):.4f}  "
                f"logloss={eval_m['log_loss']:.4f}  acc={eval_m['accuracy']:.4f}  n={batch_n}"
            )

            # === Step 2: TRAIN on this batch (except last batch = test-only) ===
            if not is_last_batch:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_loss, train_logits = forward_pass(model, batch_tensors, criterion, args.device, amp_enabled)

                if scaler is not None and amp_enabled:
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    optimizer.step()

                train_scores = torch.sigmoid(train_logits.detach()).cpu()
                train_labels_cpu = batch_tensors[-1].detach().cpu().long()
                train_m = compute_metrics(train_scores, train_labels_cpu)

                epoch_results.append({
                    "epoch": epoch,
                    "batch_index": batch_idx,
                    "phase": "train",
                    "loss": round(float(train_loss.item()), 6),
                    "auc": round(train_m["auc"], 6) if not math.isnan(train_m["auc"]) else None,
                    "log_loss": round(train_m["log_loss"], 6) if not math.isnan(train_m["log_loss"]) else None,
                    "accuracy": round(train_m["accuracy"], 6),
                    "num_samples": batch_n,
                })
                print(
                    f"[Epoch {epoch} Batch {batch_idx:03d}/{num_batches - 1}] TRAIN  "
                    f"auc={train_m['auc']:.4f}  loss={float(train_loss.item()):.4f}  "
                    f"logloss={train_m['log_loss']:.4f}  acc={train_m['accuracy']:.4f}"
                )
            else:
                print(f"[Epoch {epoch} Batch {batch_idx:03d}/{num_batches - 1}] TEST-ONLY (no training)")

            del batch_tensors

        all_progressive_results.extend(epoch_results)

        # Track best test AUC across epochs
        test_results = [r for r in epoch_results if r["phase"] == "test"]
        if test_results:
            epoch_test_auc = test_results[-1]["auc"]
            if epoch_test_auc is not None and epoch_test_auc > best_test_auc:
                best_test_auc = epoch_test_auc

    # === Compute summary metrics ===
    pre_train_evals = [r for r in all_progressive_results if r["phase"] == "pre_train_eval"]
    test_evals = [r for r in all_progressive_results if r["phase"] == "test"]

    # Average progressive AUC: skip first batch (model is untrained)
    progressive_aucs = [
        r["auc"] for r in pre_train_evals[1:]
        if r["auc"] is not None
    ]
    avg_progressive_auc = (
        sum(progressive_aucs) / len(progressive_aucs)
        if progressive_aucs else float("nan")
    )

    final_test_result = test_evals[-1] if test_evals else None
    final_test_auc = final_test_result["auc"] if final_test_result and final_test_result["auc"] is not None else float("nan")
    final_test_logloss = final_test_result.get("log_loss") if final_test_result else None
    final_test_accuracy = final_test_result.get("accuracy") if final_test_result else None

    # Save results
    final_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    run_metadata = {
        **metadata,
        "args": json_ready_args(args),
        "training_mode": "progressive_per_batch",
        "progressive_results": all_progressive_results,
        "summary": {
            "final_test_auc": round(final_test_auc, 6) if not math.isnan(final_test_auc) else None,
            "final_test_logloss": final_test_logloss,
            "final_test_accuracy": final_test_accuracy,
            "avg_progressive_auc": round(avg_progressive_auc, 6) if not math.isnan(avg_progressive_auc) else None,
            "best_test_auc": round(best_test_auc, 6) if best_test_auc != float("-inf") else None,
            "num_training_batches": num_batches - 1,
            "total_train_samples": train_sample_count,
            "test_samples": last_batch_size,
            "train_pos_rate": round(train_pos / max(train_pos + train_neg, 1), 6),
        },
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
        auc_tag = "nan" if best_test_auc == float("-inf") else f"{best_test_auc:.4f}"
        ckpt_path = args.output_dir / f"hyformer_production_{timestamp}_test_auc{auc_tag}.pt"
        torch.save(
            {
                "model": final_model_state,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "metadata": run_metadata,
            },
            ckpt_path,
        )
        print(f"checkpoint: {ckpt_path}")

    print(f"\nfinal_test_auc={final_test_auc:.4f}")
    print(f"avg_progressive_auc={avg_progressive_auc:.4f}")
    print(f"best_test_auc={best_test_auc:.4f}" if best_test_auc != float("-inf") else "best_test_auc=nan")


if __name__ == "__main__":
    main()
