"""Train TAACHyFormer on production parquet data streamed from HDFS.

Uses ``ProductionHDFSDataset`` to read parquet files from HDFS (or local),
preprocess each batch with production_data logic, and yield 8-tuple tensors
directly - no offline preprocessing step required.

Progressive per-batch training:
  - First batch:  train only (no validation)
  - Middle batches: validate then train
  - Last batch:   validate only (no training)
"""

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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from production_hdfs_dataset import ProductionHDFSDataset
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


def empty_metrics() -> dict:
    return {"auc": float("nan"), "log_loss": float("nan"), "accuracy": float("nan")}


def maybe_compute_metrics(logits: torch.Tensor, labels: torch.Tensor, enabled: bool) -> dict:
    if not enabled:
        return empty_metrics()
    scores = torch.sigmoid(logits.detach()).cpu()
    labels_cpu = labels.detach().cpu().long()
    return compute_metrics(scores, labels_cpu)


def rounded_metric(value: float) -> float | None:
    return round(value, 6) if not math.isnan(value) else None


def format_metric(value: float) -> str:
    return f"{value:.4f}" if not math.isnan(value) else "nan"


def should_run_every(batch_index: int, interval: int) -> bool:
    return interval > 0 and batch_index % interval == 0


def tensor_nonzero_rate(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float((tensor != 0).float().mean().item())


def print_batch_debug(batch_index: int, batch_tensors: tuple, logits: torch.Tensor | None = None) -> None:
    non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels = batch_tensors
    labels_float = labels.detach().float()
    pos_rate = float(labels_float.mean().item()) if labels_float.numel() else 0.0
    label_unique = sorted(labels.detach().cpu().long().unique().tolist())
    seq_mask_counts = seq_mask.detach().float().sum(dim=2).mean(dim=0).cpu().tolist()
    dense_finite = bool(torch.isfinite(non_seq_dense).all().item()) and bool(torch.isfinite(seq_dense).all().item())

    message = (
        f"[DEBUG Batch {batch_index:03d}] "
        f"pos_rate={pos_rate:.6f} labels={label_unique} "
        f"ns_sparse_nz={tensor_nonzero_rate(non_seq_sparse):.4f} "
        f"ns_bag_mask={tensor_nonzero_rate(non_seq_sparse_bag_mask):.4f} "
        f"ns_dense_abs_mean={float(non_seq_dense.detach().abs().mean().item()) if non_seq_dense.numel() else 0.0:.4f} "
        f"seq_sparse_nz={tensor_nonzero_rate(seq_sparse):.4f} "
        f"seq_dense_abs_mean={float(seq_dense.detach().abs().mean().item()) if seq_dense.numel() else 0.0:.4f} "
        f"seq_mask_mean={seq_mask_counts} "
        f"dense_finite={dense_finite}"
    )
    if logits is not None:
        detached_logits = logits.detach()
        message += (
            f" logits_mean={float(detached_logits.mean().item()):.4f} "
            f"logits_std={float(detached_logits.std(unbiased=False).item()):.4f}"
        )
    print(message)


def move_batch_to_device(batch: tuple, device: str, non_blocking: bool = False):
    """Move a batch of tensors to device, return unpacked tensors."""
    non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels = batch
    return (
        non_seq_sparse.to(device, non_blocking=non_blocking).long(),
        non_seq_sparse_bag.to(device, non_blocking=non_blocking).long(),
        non_seq_sparse_bag_mask.to(device, non_blocking=non_blocking),
        non_seq_dense.to(device, non_blocking=non_blocking).float(),
        seq_sparse.to(device, non_blocking=non_blocking).long(),
        seq_dense.to(device, non_blocking=non_blocking).float(),
        seq_mask.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking).float(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyFormer on HDFS-streamed production parquet data with progressive per-batch validation")

    # ---- Data paths ----
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="HDFS or local directory paths. Use multiple --data-path flags for "
             "multiple data sources. E.g. --data-path hdfs://nn:8020/data/hr=00 "
             "--data-path hdfs://nn:8020/data/hr=01",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "selectedfeaturefinal.txt",
        help="Feature grouping file (selectedfeaturefinal.txt).",
    )
    parser.add_argument(
        "--from-hdfs",
        action="store_true",
        default=True,
        help="Read from HDFS (default). Use --no-from-hdfs for local files.",
    )
    parser.add_argument(
        "--no-from-hdfs",
        action="store_false",
        dest="from_hdfs",
        help="Read from local filesystem instead of HDFS.",
    )

    # ---- Preprocessing ----
    parser.add_argument("--parquet-batch-size", type=int, default=4096, help="Rows per iter_batches call.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for streaming preprocessing.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned host memory for faster CUDA transfers.")
    parser.add_argument("--seq-len", type=int, default=200, help="Global max sequence length.")
    parser.add_argument("--sequence-truncation", choices=("head", "tail"), default="tail")
    parser.add_argument("--non-seq-bag-len", type=int, default=64, help="Max length per non-seq sparse bag feature.")
    parser.add_argument(
        "--sequence-lens",
        default="click_seq=100,impression_seq=200,buy_seq=200",
        help="Per-branch lengths, e.g. click_seq=100,impression_seq=200,buy_seq=200",
    )
    parser.add_argument("--non-seq-array-reduction", choices=("last", "mean"), default="last")
    parser.add_argument("--sample-rate", type=float, default=0, help="Fraction of files to use (0=all).")

    # ---- Training ----
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "production_hdfs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (progressive eval is most meaningful with 1)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=None, help="Manual pos_weight for BCEWithLogitsLoss. None=no weighting.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast and GradScaler.")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--debug-batches", type=int, default=0, help="Print data health diagnostics for the first N batches.")
    parser.add_argument("--eval-every-batches", type=int, default=20, help="Run pre-train eval every N batches. Use 1 for every batch.")
    parser.add_argument("--train-metrics-every", type=int, default=20, help="Compute train AUC/logloss every N batches. Use 1 for every batch.")

    # ---- Model ----
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--field-embed-dim", type=int, default=64)
    parser.add_argument("--token-mlp-hidden", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--hyformer-layers", type=int, default=2)
    parser.add_argument("--seq-encoder-type", choices=("longer", "full_transformer", "swiglu", "identity"), default="longer")
    parser.add_argument("--short-seq-len", type=int, default=16)
    parser.add_argument("--num-queries-per-seq", type=int, default=1)

    return parser.parse_args()


def parse_sequence_lens(raw_value: str | None) -> dict[str, int] | None:
    if raw_value is None or not raw_value.strip():
        return None
    result: dict[str, int] = {}
    for part in raw_value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid sequence length item: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = int(value.strip())
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = args.amp and args.device.startswith("cuda")

    sequence_lens = parse_sequence_lens(args.sequence_lens)

    # ---- Build data paths: --data-path flag groups -> List[List[str]] ----
    data_path: list[list[str]] = args.data_path
    if not data_path:
        print("Error: at least one --data-path is required.", file=sys.stderr)
        sys.exit(1)

    # ---- Create single dataset (no train/valid split) ----
    dataset = ProductionHDFSDataset(
        data_path=data_path,
        feature_file=args.feature_file,
        from_hdfs=args.from_hdfs,
        seq_len=args.seq_len,
        sequence_truncation=args.sequence_truncation,
        non_seq_bag_len=args.non_seq_bag_len,
        sequence_lens=sequence_lens,
        non_seq_array_reduction=args.non_seq_array_reduction,
        batch_size=args.parquet_batch_size,
        sample_rate=args.sample_rate,
        split=None,
        seed=args.seed,
    )

    loader_kwargs = {
        "batch_size": None,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    non_blocking_transfer = bool(loader_kwargs["pin_memory"])

    # ---- Build model from metadata ----
    metadata = dataset.get_metadata()
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

    # ---- Loss ----
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ---- Print config ----
    print(f"device={args.device} amp={amp_enabled}")
    print(
        f"from_hdfs={args.from_hdfs} parquet_batch_size={args.parquet_batch_size} "
        f"num_workers={args.num_workers} pin_memory={loader_kwargs['pin_memory']}"
    )
    print(
        f"sequences={metadata['sequence_names']} non_seq_tokens={len(metadata['token_groups'])} "
        f"queries_per_seq={args.num_queries_per_seq}"
    )
    print(f"seq_len={metadata['seq_len']} sequence_lens={metadata.get('sequence_lens')}")
    print(
        f"d_model={args.d_model} layers={args.hyformer_layers} encoder={args.seq_encoder_type} "
        f"field_embed_dim={args.field_embed_dim} token_mlp_hidden={args.token_mlp_hidden}"
    )
    if pos_weight is not None:
        print(f"pos_weight={pos_weight.item():.4f}")
    else:
        print("pos_weight=None (no class weighting)")
    print(
        "training_mode=progressive_per_batch "
        f"eval_every_batches={args.eval_every_batches} train_metrics_every={args.train_metrics_every}"
    )

    # ---- Progressive training loop ----
    best_test_auc = float("-inf")
    all_progressive_results = []

    for epoch in range(1, args.epochs + 1):
        epoch_results = []
        loader_iter = iter(loader)

        # ================================================================
        # First batch: train only (no validation)
        # ================================================================
        first_raw = next(loader_iter, None)
        if first_raw is None:
            print(f"[Epoch {epoch}] No data available, skipping.")
            break

        batch_tensors = move_batch_to_device(first_raw, args.device, non_blocking=non_blocking_transfer)
        batch_n = batch_tensors[-1].size(0)

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
            
        train_m = maybe_compute_metrics(train_logits, batch_tensors[-1], enabled=True)
        if args.debug_batches > 0:
            print_batch_debug(0, batch_tensors, train_logits)

        epoch_results.append({
            "epoch": epoch,
            "batch_index": 0,
            "phase": "train",
            "loss": round(float(train_loss.item()), 6),
            "auc": rounded_metric(train_m["auc"]),
            "log_loss": rounded_metric(train_m["log_loss"]),
            "accuracy": rounded_metric(train_m["accuracy"]),
            "num_samples": batch_n,
        })
        print(
            f"[Epoch {epoch} Batch 000] TRAIN  "
            f"auc={format_metric(train_m['auc'])}  loss={float(train_loss.item()):.4f}  "
            f"logloss={format_metric(train_m['log_loss'])}  acc={format_metric(train_m['accuracy'])}  n={batch_n}"
        )

        del batch_tensors, first_raw

        # ================================================================
        # Middle + last batches: use look-ahead buffer
        #
        # We buffer one batch ahead. When we see a new batch, the buffered
        # one is guaranteed NOT to be the last -> eval + train.
        # After the loop, the remaining buffered batch IS the last -> eval only.
        # ================================================================
        buffered_raw = None
        buffered_idx = None

        for idx, current_raw in enumerate(loader_iter, start=1):
            if buffered_raw is not None:
                # buffered_raw is NOT the last batch -> eval + train
                batch_tensors = move_batch_to_device(buffered_raw, args.device, non_blocking=non_blocking_transfer)
                batch_n = batch_tensors[-1].size(0)

                run_eval = should_run_every(buffered_idx, args.eval_every_batches)
                if run_eval:
                    # --- Evaluate on buffered batch ---
                    model.eval()
                    with torch.no_grad():
                        eval_loss, eval_logits = forward_pass(model, batch_tensors, criterion, args.device, amp_enabled)

                    eval_m = maybe_compute_metrics(eval_logits, batch_tensors[-1], enabled=True)
                    if args.debug_batches > 0 and buffered_idx < args.debug_batches:
                        print_batch_debug(buffered_idx, batch_tensors, eval_logits)

                    epoch_results.append({
                        "epoch": epoch,
                        "batch_index": buffered_idx,
                        "phase": "pre_train_eval",
                        "loss": round(float(eval_loss.item()), 6),
                        "auc": rounded_metric(eval_m["auc"]),
                        "log_loss": rounded_metric(eval_m["log_loss"]),
                        "accuracy": rounded_metric(eval_m["accuracy"]),
                        "num_samples": batch_n,
                    })
                    print(
                        f"[Epoch {epoch} Batch {buffered_idx:03d}] EVAL   "
                        f"auc={format_metric(eval_m['auc'])}  loss={float(eval_loss.item()):.4f}  "
                        f"logloss={format_metric(eval_m['log_loss'])}  "
                        f"acc={format_metric(eval_m['accuracy'])}  n={batch_n}"
                    )

                # --- Train on buffered batch ---
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

                run_train_metrics = should_run_every(buffered_idx, args.train_metrics_every)
                train_m = maybe_compute_metrics(train_logits, batch_tensors[-1], enabled=run_train_metrics)
                if args.debug_batches > 0 and buffered_idx < args.debug_batches and not run_eval:
                    print_batch_debug(buffered_idx, batch_tensors, train_logits)

                epoch_results.append({
                    "epoch": epoch,
                    "batch_index": buffered_idx,
                    "phase": "train",
                    "loss": round(float(train_loss.item()), 6),
                    "auc": rounded_metric(train_m["auc"]),
                    "log_loss": rounded_metric(train_m["log_loss"]),
                    "accuracy": rounded_metric(train_m["accuracy"]),
                    "num_samples": batch_n,
                })
                if run_train_metrics:
                    print(
                        f"[Epoch {epoch} Batch {buffered_idx:03d}] TRAIN  "
                        f"auc={format_metric(train_m['auc'])}  loss={float(train_loss.item()):.4f}  "
                        f"logloss={format_metric(train_m['log_loss'])}  "
                        f"acc={format_metric(train_m['accuracy'])}"
                    )

                del batch_tensors

            # Buffer current batch for next iteration
            buffered_raw = current_raw
            buffered_idx = idx

        # ================================================================
        # Last batch (buffered_raw): eval only, no training
        # ================================================================
        if buffered_raw is not None:
            batch_tensors = move_batch_to_device(buffered_raw, args.device, non_blocking=non_blocking_transfer)
            batch_n = batch_tensors[-1].size(0)

            model.eval()
            with torch.no_grad():
                eval_loss, eval_logits = forward_pass(model, batch_tensors, criterion, args.device, amp_enabled)

            eval_m = maybe_compute_metrics(eval_logits, batch_tensors[-1], enabled=True)
            if args.debug_batches > 0 and buffered_idx < args.debug_batches:
                print_batch_debug(buffered_idx, batch_tensors, eval_logits)

            epoch_results.append({
                "epoch": epoch,
                "batch_index": buffered_idx,
                "phase": "test",
                "loss": round(float(eval_loss.item()), 6),
                "auc": rounded_metric(eval_m["auc"]),
                "log_loss": rounded_metric(eval_m["log_loss"]),
                "accuracy": rounded_metric(eval_m["accuracy"]),
                "num_samples": batch_n,
            })
            print(
                f"[Epoch {epoch} Batch {buffered_idx:03d}] TEST   "
                f"auc={format_metric(eval_m['auc'])}  loss={float(eval_loss.item()):.4f}  "
                f"logloss={format_metric(eval_m['log_loss'])}  "
                f"acc={format_metric(eval_m['accuracy'])}  n={batch_n}"
            )
            print(f"[Epoch {epoch} Batch {buffered_idx:03d}] TEST-ONLY (no training)")

            del batch_tensors, buffered_raw

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
    train_evals = [r for r in all_progressive_results if r["phase"] == "train"]

    progressive_aucs = [
        r["auc"] for r in pre_train_evals
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

    total_train_samples = sum(r["num_samples"] for r in train_evals)
    total_test_samples = sum(r["num_samples"] for r in test_evals)

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
            "num_training_batches": len(train_evals),
            "total_train_samples": total_train_samples,
            "test_samples": total_test_samples,
        },
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": args.pos_weight,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.save_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auc_tag = "nan" if best_test_auc == float("-inf") else f"{best_test_auc:.4f}"
        ckpt_path = args.output_dir / f"hyformer_hdfs_{timestamp}_test_auc{auc_tag}.pt"
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
