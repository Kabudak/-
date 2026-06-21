from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.production_data import LABEL_MODES, build_tensors, load_parquet_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize production parquet features for HyFormer")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=PROJECT_ROOT / "000000_0_selected_head100.parquet",
        help="Production parquet file to read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "production_sample",
        help="Directory where tensor files and metadata.json will be written.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick checks.")
    parser.add_argument("--seq-len", type=int, default=100, help="Maximum length per inferred behavior sequence.")
    parser.add_argument(
        "--label-mode",
        choices=sorted(LABEL_MODES),
        default="rel_score_present",
        help="How to turn production relevance columns into a binary label.",
    )
    parser.add_argument(
        "--sequence-truncation",
        choices=("head", "tail"),
        default="tail",
        help="Keep the first or last seq-len items when a behavior list is longer than seq-len.",
    )
    parser.add_argument(
        "--max-seq-fields-per-branch",
        type=int,
        default=None,
        help="Optional cap for sequence side fields per branch. Useful for very wide full production data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] reading parquet: {args.input_parquet}")
    columns, arrow_types = load_parquet_columns(args.input_parquet, max_rows=args.max_rows)
    tensors = build_tensors(
        columns=columns,
        arrow_types=arrow_types,
        seq_len=args.seq_len,
        label_mode=args.label_mode,
        sequence_truncation=args.sequence_truncation,
        max_seq_fields_per_branch=args.max_seq_fields_per_branch,
    )
    non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels, metadata = tensors

    torch.save(non_seq_sparse, args.output_dir / "non_seq_sparse.pt")
    torch.save(non_seq_dense, args.output_dir / "non_seq_dense.pt")
    torch.save(seq_sparse, args.output_dir / "seq_sparse.pt")
    torch.save(seq_dense, args.output_dir / "seq_dense.pt")
    torch.save(seq_mask, args.output_dir / "seq_mask.pt")
    torch.save(labels, args.output_dir / "labels.pt")

    metadata["source_parquet"] = str(args.input_parquet)
    metadata["preprocess_args"] = {
        "max_rows": args.max_rows,
        "seq_len": args.seq_len,
        "label_mode": args.label_mode,
        "sequence_truncation": args.sequence_truncation,
        "max_seq_fields_per_branch": args.max_seq_fields_per_branch,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[data] saved tensors")
    print(f"  output_dir:       {args.output_dir}")
    print(f"  labels:           {tuple(labels.shape)}  pos_rate={metadata['pos_rate']:.4f}")
    print(f"  non_seq_sparse:   {tuple(non_seq_sparse.shape)}")
    print(f"  non_seq_dense:    {tuple(non_seq_dense.shape)}")
    print(f"  seq_sparse:       {tuple(seq_sparse.shape)}")
    print(f"  seq_dense:        {tuple(seq_dense.shape)}")
    print(f"  seq_mask:         {tuple(seq_mask.shape)}")
    print(f"  sequence_names:   {metadata['sequence_names']}")
    print(f"  token_groups:     {list(metadata['token_groups'].keys())}")


if __name__ == "__main__":
    main()
