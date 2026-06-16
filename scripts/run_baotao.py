from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from utils.common import json_ready_args, set_seed

TZ_OFFSET_SEC = 8 * 3600
THRESHOLD = 0.06


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


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = THRESHOLD) -> float:
    preds = (torch.sigmoid(logits) >= threshold).long()
    return float((preds == labels).float().mean().item())


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

    fwd_ctx = torch.no_grad() if not is_train else nullcontext()
    with fwd_ctx:
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


def load_day(data_dir: Path, day_label: str) -> tuple[torch.Tensor, ...]:
    """Load all tensors for a single day from its subdirectory."""
    day_dir = data_dir / day_label
    return (
        torch.load(day_dir / "non_seq_sparse.pt", weights_only=True).long(),
        torch.load(day_dir / "non_seq_dense.pt", weights_only=True),
        torch.load(day_dir / "seq_sparse.pt", weights_only=True).long(),
        torch.load(day_dir / "seq_dense.pt", weights_only=True).float(),
        torch.load(day_dir / "seq_mask.pt", weights_only=True),
        torch.load(day_dir / "labels.pt", weights_only=True).long(),
    )


def make_loader_from_tensors(
    non_seq_sparse: torch.Tensor,
    non_seq_dense: torch.Tensor,
    seq_sparse: torch.Tensor,
    seq_dense: torch.Tensor,
    seq_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTR prediction with progressive validation")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-mode", choices=("none", "auto"), default="none")
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

    # ── Read metadata ────────────────────────────────────────────────────
    with open(args.data_dir / "metadata.json", encoding="utf-8") as f:
        data_meta = json.load(f)

    progressive_split = data_meta.get("progressive_split")
    if progressive_split is None:
        raise ValueError(
            "metadata.json does not contain 'progressive_split'. "
            "Please re-run preprocess_baotao.py to generate per-day split data."
        )
    day_splits = progressive_split["days"]
    num_days = progressive_split["num_days"]

    print(f"Progressive validation: {num_days} days  threshold={THRESHOLD}")
    for ds in day_splits:
        print(f"  {ds['day_label']}: samples={ds['num_samples']:,}")

    # ── Validate model config ────────────────────────────────────────────
    expected_non_seq_tokens = len(data_meta["token_groups"])
    if args.num_non_seq_tokens != expected_non_seq_tokens:
        raise ValueError(
            f"num_non_seq_tokens mismatch: args={args.num_non_seq_tokens} data={expected_non_seq_tokens}"
        )

    # ── Build model ──────────────────────────────────────────────────────
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

    # ── Compute pos_weight from training days (1..N-1) ──────────────────
    train_pos = 0
    train_neg = 0
    for day_idx in range(num_days - 1):
        day_label = day_splits[day_idx]["day_label"]
        day_labels = torch.load(args.data_dir / day_label / "labels.pt", weights_only=True).long()
        train_pos += int(day_labels.sum().item())
        train_neg += int(len(day_labels) - day_labels.sum().item())
        del day_labels

    train_total = train_pos + train_neg
    pos_weight_value = train_neg / max(train_pos, 1)
    pos_weight = (
        torch.tensor([pos_weight_value], dtype=torch.float32, device=args.device)
        if args.pos_weight_mode == "auto"
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Print config ─────────────────────────────────────────────────────
    print(f"device={args.device}")
    print(f"  training_days={num_days - 1}  test_day={day_splits[-1]['day_label']}  "
          f"train_samples={train_total}  test_samples={day_splits[-1]['num_samples']}")
    if pos_weight is None:
        print(f"  train_pos_rate={train_pos / max(train_total, 1) * 100:.2f}%  pos_weight=none")
    else:
        print(f"  train_pos_rate={train_pos / max(train_total, 1) * 100:.2f}%  pos_weight={pos_weight.item():.2f}")
    print(
        f"  d_model={args.d_model}  field_embed_dim={args.field_embed_dim} "
        f"layers={args.hyformer_layers}  encoder={args.seq_encoder_type}"
    )

    args_payload = json_ready_args(args)

    # ── Progressive validation training loop ─────────────────────────────
    # For each day D:
    #   1. Load day D's tensors
    #   2. Evaluate with current model (pre-training metrics)
    #   3. Train on day D (except last day = test-only)
    #   4. Release day D's tensors to free memory
    progressive_results = []

    for day_idx in range(num_days):
        day_info = day_splits[day_idx]
        day_label = day_info["day_label"]
        is_last_day = day_idx == num_days - 1

        # Load this day's data
        non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, day_labels = load_day(
            args.data_dir, day_label,
        )

        # === EVALUATE before training on this day ===
        day_neg_rate = float(1.0 - day_labels.float().mean().item())
        eval_loader = make_loader_from_tensors(
            non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, day_labels,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        model.eval()
        eval_loss, eval_auc, eval_ll, eval_acc = run_epoch(
            model, eval_loader, criterion, args.device, optimizer=None,
        )
        progressive_results.append({
            "day_label": day_label,
            "day_index": day_idx,
            "phase": "pre_train_eval",
            "loss": round(eval_loss, 6),
            "auc": round(eval_auc, 6) if not math.isnan(eval_auc) else None,
            "log_loss": round(eval_ll, 6) if not math.isnan(eval_ll) else None,
            "accuracy": round(eval_acc, 6),
            "neg_rate": round(day_neg_rate, 4),
            "num_samples": len(day_labels),
        })
        phase_tag = "TEST" if is_last_day else "EVAL"
        print(f"[{day_label}] {phase_tag:5s}  auc={eval_auc:.4f}  loss={eval_loss:.4f}  "
              f"logloss={eval_ll:.4f}  acc={eval_acc:.4f}  neg_rate={day_neg_rate:.4f}  n={len(day_labels):,}")

        # === TRAIN on this day (except last day = test-only) ===
        if not is_last_day:
            model.train()
            train_loader = make_loader_from_tensors(
                non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, day_labels,
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            )
            train_loss, train_auc, train_ll, train_acc = run_epoch(
                model, train_loader, criterion, args.device, optimizer=optimizer,
            )
            progressive_results.append({
                "day_label": day_label,
                "day_index": day_idx,
                "phase": "train",
                "loss": round(train_loss, 6),
                "auc": round(train_auc, 6) if not math.isnan(train_auc) else None,
                "log_loss": round(train_ll, 6) if not math.isnan(train_ll) else None,
                "accuracy": round(train_acc, 6),
                "num_samples": len(day_labels),
            })
            print(f"[{day_label}] TRAIN  auc={train_auc:.4f}  loss={train_loss:.4f}  logloss={train_ll:.4f}  acc={train_acc:.4f}")
        else:
            print(f"[{day_label}] TEST-ONLY (no training)")

        # Release this day's tensors to free memory
        del non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, day_labels
        del eval_loader
        if not is_last_day:
            del train_loader

    # ── Compute summary metrics ──────────────────────────────────────────
    pre_train_evals = [r for r in progressive_results if r["phase"] == "pre_train_eval"]
    final_test_result = pre_train_evals[-1]  # Last day = test set

    # Average progressive AUC: days 2..N (skip day 1 since model is untrained)
    progressive_aucs = [
        r["auc"] for r in pre_train_evals[1:]
        if r["auc"] is not None
    ]
    avg_progressive_auc = (
        sum(progressive_aucs) / len(progressive_aucs)
        if progressive_aucs else float("nan")
    )
    final_test_auc = final_test_result["auc"] if final_test_result["auc"] is not None else float("nan")

    # ── Save results ─────────────────────────────────────────────────────
    final_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    run_meta = {
        **data_meta,
        "args": args_payload,
        "training_mode": "progressive",
        "threshold": THRESHOLD,
        "progressive_results": progressive_results,
        "summary": {
            "final_test_auc": round(final_test_auc, 6) if not math.isnan(final_test_auc) else None,
            "final_test_logloss": final_test_result.get("log_loss"),
            "final_test_accuracy": final_test_result.get("accuracy"),
            "avg_progressive_auc": round(avg_progressive_auc, 6) if not math.isnan(avg_progressive_auc) else None,
            "num_training_days": num_days - 1,
            "total_train_samples": train_total,
            "test_samples": day_splits[-1]["num_samples"],
            "train_pos_rate": round(train_pos / max(train_total, 1), 6),
        },
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight_mode": args.pos_weight_mode,
            "pos_weight_value": round(pos_weight_value, 6) if pos_weight is not None else None,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.save_checkpoint:
        ckpt_path = args.output_dir / f"final_model_test_auc{final_test_auc:.4f}.pt"
        torch.save({"model": final_model_state, "metadata": run_meta}, ckpt_path)
        print(f"  checkpoint: {ckpt_path}")

    print(f"\n  final_test_auc={final_test_auc:.4f}")
    print(f"  avg_progressive_auc={avg_progressive_auc:.4f}")


if __name__ == "__main__":
    main()
